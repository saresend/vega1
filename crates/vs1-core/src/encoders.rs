//! MobileCLIP-S0 image and text encoders run with tract.
//!
//! Expects graphs prepared by `train/prepare_models.py` (pinned input shapes,
//! constant-folded). Both encoders return L2-normalised 512-d embeddings in a
//! shared space.

use anyhow::{anyhow, ensure, Context};
use tract_onnx::prelude::*;

use crate::image::{Frame, Region};

pub const EMBED_DIM: usize = 512;
pub const IMAGE_SIZE: usize = 256;
pub const CONTEXT_LEN: usize = 77;
const BOS: i64 = 49406;
const EOS: i64 = 49407;

type Plan = TypedRunnableModel<TypedModel>;

fn load(bytes: &[u8], what: &str, input: InferenceFact) -> anyhow::Result<Plan> {
    tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .with_context(|| format!("parsing {what} ONNX"))?
        .with_input_fact(0, input)?
        .into_optimized()
        .with_context(|| format!("optimizing {what} graph"))?
        .into_runnable()
        .with_context(|| format!("planning {what} graph"))
}

fn first_output(plan: &Plan, input: Tensor) -> anyhow::Result<Vec<f32>> {
    let out = plan.run(tvec!(input.into()))?;
    let v = out[0].cast_to::<f32>()?;
    let v = v.as_slice::<f32>()?;
    ensure!(v.len() == EMBED_DIM, "expected {EMBED_DIM}-d embedding, got {}", v.len());
    Ok(normalized(v.to_vec()))
}

pub fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

pub struct ImageEncoder {
    plan: Plan,
}

impl ImageEncoder {
    pub fn from_onnx_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let fact = f32::fact([1, 3, IMAGE_SIZE, IMAGE_SIZE]).into();
        Ok(Self { plan: load(bytes, "vision encoder", fact)? })
    }

    pub fn embed(&self, frame: &Frame) -> anyhow::Result<Vec<f32>> {
        self.embed_region(frame, frame.full())
    }

    pub fn embed_region(&self, frame: &Frame, region: Region) -> anyhow::Result<Vec<f32>> {
        let pixels = frame.region_to_clip_input(region, IMAGE_SIZE);
        first_output(&self.plan, Tensor::from_shape(&[1, 3, IMAGE_SIZE, IMAGE_SIZE], &pixels)?)
    }
}

pub struct TextEncoder {
    plan: Plan,
    tokenizer: tokenizers::Tokenizer,
}

impl TextEncoder {
    pub fn from_bytes(onnx: &[u8], tokenizer_json: &[u8]) -> anyhow::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json).map_err(|e| anyhow!("loading tokenizer: {e}"))?;
        let fact = i64::fact([1, CONTEXT_LEN]).into();
        Ok(Self { plan: load(onnx, "text encoder", fact)?, tokenizer })
    }

    fn tokens(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(enc.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// Token ids for `context` followed by `focus`. When the pair exceeds the
    /// context window the *context* is truncated, so the part that
    /// distinguishes one option from another always survives.
    pub fn pair_ids(&self, context: &str, focus: &str) -> anyhow::Result<Vec<i64>> {
        let inner = CONTEXT_LEN - 2;
        let mut focus_ids = self.tokens(focus)?;
        focus_ids.truncate(inner);
        let mut context_ids = self.tokens(context)?;
        context_ids.truncate(inner - focus_ids.len());
        let mut ids = Vec::with_capacity(CONTEXT_LEN);
        ids.push(BOS);
        ids.extend(context_ids);
        ids.extend(focus_ids);
        ids.push(EOS);
        Ok(ids)
    }

    /// Embeds a context/focus pair; also returns the number of real tokens.
    pub fn embed_pair(&self, context: &str, focus: &str) -> anyhow::Result<(Vec<f32>, usize)> {
        let mut ids = self.pair_ids(context, focus)?;
        let used = ids.len();
        ids.resize(CONTEXT_LEN, 0); // zero padding; pooling reads the EOS (max id) position
        let emb = first_output(&self.plan, Tensor::from_shape(&[1, CONTEXT_LEN], &ids)?)?;
        Ok((emb, used))
    }
}
