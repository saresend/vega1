//! Answer heads: turn (frame embeddings, question, option embeddings) into
//! option logits.
//!
//! * Zero-shot: each option's logit is its similarity to the video, pooled
//!   over frames with a log-mean-exp (a soft max), so evidence present in only
//!   some frames still counts.
//! * Trained: a small question-conditioned attention head over frames whose
//!   output is added to the (learned-scale) zero-shot logits. With `gamma = 0`
//!   it reduces exactly to zero-shot, which is how training starts.
//!
//! The trained head runs in plain Rust (it's ~1M parameters) and is mirrored
//! exactly by `train/head.py`; weights load from safetensors.

use anyhow::{bail, ensure, Context};
use safetensors::{Dtype, SafeTensors};

use crate::encoders::EMBED_DIM;

const LN_EPS: f32 = 1e-5;

/// Zero-shot defaults, used when no trained head is loaded.
pub const DEFAULT_SCALE: f32 = 50.0;
pub const DEFAULT_POOL_BETA: f32 = 10.0;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Soft maximum over frames of cosine similarity to `option`.
pub fn pooled_similarity(frames: &[Vec<f32>], option: &[f32], beta: f32) -> f32 {
    let sims: Vec<f32> = frames.iter().map(|f| beta * dot(f, option)).collect();
    let max = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_exp = sims.iter().map(|s| (s - max).exp()).sum::<f32>() / sims.len() as f32;
    (max + mean_exp.ln()) / beta
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Row-major `out x in` weight with bias.
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    inputs: usize,
}

impl Linear {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.inputs);
        self.w.chunks_exact(self.inputs).zip(&self.b).map(|(row, b)| dot(row, x) + b).collect()
    }
}

pub struct TrainedHead {
    pub name: String,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
    w1: Linear,
    w2: Linear,
    wz: Linear,
    a_zs: f32,
    gamma: f32,
    pool_beta: f32,
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

impl TrainedHead {
    pub fn from_safetensors(bytes: &[u8]) -> anyhow::Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parsing head safetensors")?;
        let (_, meta) = SafeTensors::read_metadata(bytes).context("reading head metadata")?;
        let name = meta
            .metadata()
            .as_ref()
            .and_then(|m| m.get("name").cloned())
            .unwrap_or_else(|| "vs1-head".to_string());
        let tensor = |key: &str, shape: &[usize]| -> anyhow::Result<Vec<f32>> {
            let t = st.tensor(key).with_context(|| format!("head is missing `{key}`"))?;
            if t.dtype() != Dtype::F32 {
                bail!("`{key}` must be f32, got {:?}", t.dtype());
            }
            ensure!(t.shape() == shape, "`{key}` has shape {:?}, expected {shape:?}", t.shape());
            Ok(t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
        };
        let hidden = st.tensor("bq").context("head is missing `bq`")?.shape()[0];
        let mlp = st.tensor("b1").context("head is missing `b1`")?.shape()[0];
        let linear = |w: &str, b: &str, out: usize, inp: usize| -> anyhow::Result<Linear> {
            Ok(Linear { w: tensor(w, &[out, inp])?, b: tensor(b, &[out])?, inputs: inp })
        };
        let scalar = |key: &str| -> anyhow::Result<f32> { Ok(tensor(key, &[1])?[0]) };
        Ok(Self {
            name,
            wq: linear("wq", "bq", hidden, EMBED_DIM)?,
            wk: linear("wk", "bk", hidden, EMBED_DIM)?,
            wv: linear("wv", "bv", hidden, EMBED_DIM)?,
            ln_g: tensor("ln_g", &[hidden])?,
            ln_b: tensor("ln_b", &[hidden])?,
            w1: linear("w1", "b1", mlp, hidden)?,
            w2: linear("w2", "b2", hidden, mlp)?,
            wz: linear("wz", "bz", EMBED_DIM, hidden)?,
            a_zs: scalar("a_zs")?,
            gamma: scalar("gamma")?,
            pool_beta: scalar("pool_beta")?,
        })
    }

    fn logits(&self, frames: &[Vec<f32>], question: &[f32], options: &[Vec<f32>]) -> Vec<f32> {
        let hidden = self.ln_g.len();
        let qh = self.wq.forward(question);
        let scale = 1.0 / (hidden as f32).sqrt();
        let keys: Vec<f32> = frames.iter().map(|f| dot(&self.wk.forward(f), &qh) * scale).collect();
        let attn = softmax(&keys);
        let mut h = qh.clone();
        for (f, a) in frames.iter().zip(&attn) {
            for (hi, vi) in h.iter_mut().zip(self.wv.forward(f)) {
                *hi += a * vi;
            }
        }
        let mean = h.iter().sum::<f32>() / hidden as f32;
        let var = h.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        let h: Vec<f32> = h.iter().zip(&self.ln_g).zip(&self.ln_b).map(|((x, g), b)| (x - mean) * inv * g + b).collect();
        let u: Vec<f32> = self.w1.forward(&h).into_iter().map(gelu_tanh).collect();
        let h2: Vec<f32> = h.iter().zip(self.w2.forward(&u)).map(|(a, b)| a + b).collect();
        let z = self.wz.forward(&h2);
        options
            .iter()
            .map(|o| self.a_zs * pooled_similarity(frames, o, self.pool_beta) + self.gamma * dot(&z, o))
            .collect()
    }
}

pub enum Head {
    ZeroShot { scale: f32, pool_beta: f32 },
    Trained(TrainedHead),
}

impl Default for Head {
    fn default() -> Self {
        Head::ZeroShot { scale: DEFAULT_SCALE, pool_beta: DEFAULT_POOL_BETA }
    }
}

impl Head {
    pub fn name(&self) -> &str {
        match self {
            Head::ZeroShot { .. } => "vs1-zeroshot-mobileclip-s0",
            Head::Trained(h) => &h.name,
        }
    }

    pub fn probabilities(&self, frames: &[Vec<f32>], question: &[f32], options: &[Vec<f32>]) -> Vec<f32> {
        let logits = match self {
            Head::ZeroShot { scale, pool_beta } => {
                options.iter().map(|o| scale * pooled_similarity(frames, o, *pool_beta)).collect()
            }
            Head::Trained(h) => h.logits(frames, question, options),
        };
        softmax(&logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; EMBED_DIM];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn pooling_rewards_evidence_in_some_frames() {
        let frames = vec![unit(0), unit(0), unit(0), unit(1)];
        let seen_once = pooled_similarity(&frames, &unit(1), 10.0);
        let never = pooled_similarity(&frames, &unit(2), 10.0);
        let mean = 0.25;
        assert!(seen_once > mean, "soft max exceeds the mean: {seen_once}");
        assert!(never.abs() < 1e-6);
        // Identical frames pool to their similarity exactly.
        assert!((pooled_similarity(&[unit(3), unit(3)], &unit(3), 10.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_shot_prefers_matching_option() {
        let p = Head::default().probabilities(&[unit(4)], &unit(0), &[unit(4), unit(5)]);
        assert!(p[0] > 0.99 && (p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
