//! vs1-core: a general, on-device "System One" model for video.
//!
//! Same contract as TypeSafe's Jev (`POST /v1/systemone`): a state plus a map
//! of typed questions (`noul`, `choice`, `score`) in, calibrated typed answers
//! out — except the state is a video, and everything runs locally (native or
//! WASM).
//!
//! ```text
//! frames ──ImageEncoder──► frame embeddings ─┐
//! questions ──prompts──TextEncoder──► option embeddings ─┴─► Head ─► probabilities ─► Answers
//! ```
//!
//! Questions are compiled once ([`SystemOne::compile`]) and can be reused
//! across videos; frames are embedded once ([`FrameEmbedder`]) and can be
//! re-asked any question set.
//!
//! A video's state is a set of embedding rows: one per sampled frame (the
//! full view) plus six crops of each of a few frames ([`crop_frame_indices`]),
//! so small objects are seen at up to twice the scale. The head pools over
//! rows without regard to order, so rows can be computed in any order or in
//! parallel.

pub mod api;
pub mod encoders;
pub mod head;
pub mod image;
pub mod prompts;

use std::collections::HashMap;

use anyhow::Context;
use indexmap::IndexMap;

use api::{round4, Answer, Question, Request, Response, Usage};
use encoders::{ImageEncoder, TextEncoder};
pub use image::Region;
use head::{Head, TrainedHead};
pub use image::Frame;

/// Raw model files; how they're fetched is the caller's business. Either
/// encoder can be left out and loaded later ([`SystemOne::load_vision`],
/// [`SystemOne::load_text`]): precomputed frame and text embeddings need
/// neither, so a page can answer before the encoders download.
pub struct ModelFiles<'a> {
    pub vision_onnx: Option<&'a [u8]>,
    /// The text encoder and its tokenizer.
    pub text: Option<(&'a [u8], &'a [u8])>,
    /// Optional trained answer head (safetensors). Zero-shot without it.
    pub head: Option<&'a [u8]>,
}

/// A question set, embedded and ready to apply to any video.
pub struct Compiled {
    questions: Vec<CompiledQuestion>,
    input_tokens: usize,
}

struct CompiledQuestion {
    key: String,
    kind: Kind,
    question: Vec<f32>,
    options: Vec<Vec<f32>>,
}

enum Kind {
    /// `negated`: the options describe the statement's positive form.
    Noul { negated: bool },
    Choice { labels: Vec<String> },
    Score { legend: Vec<String> },
}

/// How many of the sampled frames also get crop embeddings, by default.
pub const CROP_FRAMES: usize = 4;

/// Indices of the frames (out of `frames`, sampled uniformly) whose crops are
/// embedded too: `crop_frames` of them, evenly spread (2, 6, 10, 14 of 16).
pub fn crop_frame_indices(frames: usize, crop_frames: usize) -> Vec<usize> {
    let n = crop_frames.min(frames);
    (0..n).map(|j| (2 * j + 1) * frames / (2 * n)).collect()
}

/// The vision half of the model: frames in, embedding rows out. Cheap to run
/// several of (one per worker) to embed frames in parallel.
pub struct FrameEmbedder {
    image: ImageEncoder,
}

impl FrameEmbedder {
    pub fn new(vision_onnx: &[u8]) -> anyhow::Result<Self> {
        Ok(Self { image: ImageEncoder::from_onnx_bytes(vision_onnx)? })
    }

    /// The full-frame embedding.
    pub fn embed(&self, frame: &Frame) -> anyhow::Result<Vec<f32>> {
        self.image.embed(frame)
    }

    /// Embeddings of the frame's crops ([`Frame::crop_regions`]).
    pub fn embed_crops(&self, frame: &Frame) -> anyhow::Result<Vec<Vec<f32>>> {
        frame.crop_regions().into_iter().map(|r| self.image.embed_region(frame, r)).collect()
    }

    /// The full view, then the crops if `crops`.
    pub fn embed_rows(&self, frame: &Frame, crops: bool) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut rows = vec![self.embed(frame)?];
        if crops {
            rows.extend(self.embed_crops(frame)?);
        }
        Ok(rows)
    }
}

/// The text half of the model, for extra workers that encode question text
/// in parallel ([`SystemOne::missing_texts`], [`SystemOne::insert_text`]).
pub struct TextEmbedder {
    text: TextEncoder,
}

impl TextEmbedder {
    pub fn new(text_onnx: &[u8], tokenizer_json: &[u8]) -> anyhow::Result<Self> {
        Ok(Self { text: TextEncoder::from_bytes(text_onnx, tokenizer_json)? })
    }

    /// (embedding, tokens used) of a context/focus pair.
    pub fn embed(&self, context: &str, focus: &str) -> anyhow::Result<(Vec<f32>, usize)> {
        self.text.embed_pair(context, focus)
    }
}

pub struct SystemOne {
    vision: Option<FrameEmbedder>,
    text: Option<TextEncoder>,
    head: Head,
    text_cache: HashMap<(String, String), (Vec<f32>, usize)>,
}

impl SystemOne {
    pub fn new(files: ModelFiles) -> anyhow::Result<Self> {
        let head = match files.head {
            Some(bytes) => Head::Trained(TrainedHead::from_safetensors(bytes)?),
            None => Head::default(),
        };
        Ok(Self {
            vision: files.vision_onnx.map(FrameEmbedder::new).transpose()?,
            text: files.text.map(|(onnx, tokenizer)| TextEncoder::from_bytes(onnx, tokenizer)).transpose()?,
            head,
            text_cache: HashMap::new(),
        })
    }

    pub fn model_name(&self) -> &str {
        self.head.name()
    }

    pub fn load_vision(&mut self, vision_onnx: &[u8]) -> anyhow::Result<()> {
        self.vision = Some(FrameEmbedder::new(vision_onnx)?);
        Ok(())
    }

    pub fn load_text(&mut self, text_onnx: &[u8], tokenizer_json: &[u8]) -> anyhow::Result<()> {
        self.text = Some(TextEncoder::from_bytes(text_onnx, tokenizer_json)?);
        Ok(())
    }

    pub fn has_text(&self) -> bool {
        self.text.is_some()
    }

    pub fn vision(&self) -> anyhow::Result<&FrameEmbedder> {
        self.vision.as_ref().context("the vision encoder is not loaded")
    }

    pub fn embed_frame(&self, frame: &Frame) -> anyhow::Result<Vec<f32>> {
        self.vision()?.embed(frame)
    }

    pub fn embed_text(&mut self, context: &str, focus: &str) -> anyhow::Result<(Vec<f32>, usize)> {
        let key = (context.to_string(), focus.to_string());
        if let Some(hit) = self.text_cache.get(&key) {
            return Ok(hit.clone());
        }
        let value = self.text.as_ref().context("the text encoder is not loaded")?.embed_pair(context, focus)?;
        self.text_cache.insert(key, value.clone());
        Ok(value)
    }

    /// The (context, focus) texts a request needs that aren't embedded yet,
    /// without duplicates: embed them elsewhere (e.g. in parallel with
    /// [`TextEmbedder`]) and [`SystemOne::insert_text`] the results.
    pub fn missing_texts(&self, request: &Request) -> anyhow::Result<Vec<(String, String)>> {
        request.validate()?;
        let mut out: Vec<(String, String)> = Vec::new();
        for q in request.questions.values() {
            let prompts = prompts::build(q);
            let texts = std::iter::once((String::new(), prompts.question.clone()))
                .chain(prompts.options.iter().map(|o| (o.context.clone(), o.focus.clone())));
            for key in texts {
                if !self.text_cache.contains_key(&key) && !out.contains(&key) {
                    out.push(key);
                }
            }
        }
        Ok(out)
    }

    /// Adds a text embedded outside this instance (see [`SystemOne::missing_texts`]).
    pub fn insert_text(&mut self, context: &str, focus: &str, embedding: Vec<f32>, tokens: usize) -> anyhow::Result<()> {
        anyhow::ensure!(embedding.len() == encoders::EMBED_DIM, "text embeddings are {} floats", encoders::EMBED_DIM);
        self.text_cache.insert((context.to_string(), focus.to_string()), (embedding, tokens));
        Ok(())
    }

    /// Embeds one question: (question embedding, option embeddings, tokens used).
    pub fn embed_question(&mut self, q: &Question) -> anyhow::Result<(Vec<f32>, Vec<Vec<f32>>, usize)> {
        self.embed_prompts(&prompts::build(q))
    }

    fn embed_prompts(&mut self, prompts: &prompts::Prompts) -> anyhow::Result<(Vec<f32>, Vec<Vec<f32>>, usize)> {
        let (question, mut tokens) = self.embed_text("", &prompts.question)?;
        let mut options = Vec::with_capacity(prompts.options.len());
        for o in &prompts.options {
            let (emb, n) = self.embed_text(&o.context, &o.focus)?;
            tokens += n;
            options.push(emb);
        }
        Ok((question, options, tokens))
    }

    /// Validates and embeds a question set.
    pub fn compile(&mut self, request: &Request) -> anyhow::Result<Compiled> {
        request.validate()?;
        let mut questions = Vec::with_capacity(request.questions.len());
        let mut input_tokens = 0;
        for (key, q) in &request.questions {
            let prompts = prompts::build(q);
            let (question, options, n) = self.embed_prompts(&prompts)?;
            input_tokens += n;
            let kind = match q {
                Question::Noul { .. } => Kind::Noul { negated: prompts.negated },
                Question::Choice { criteria, .. } => Kind::Choice { labels: criteria.keys().cloned().collect() },
                Question::Score { criteria, .. } => Kind::Score { legend: criteria.iter().map(api::render).collect() },
            };
            questions.push(CompiledQuestion { key: key.clone(), kind, question, options });
        }
        Ok(Compiled { questions, input_tokens })
    }

    /// Answers a compiled question set for a video given its frame embeddings.
    /// `rows` are the video state's embedding rows: one full view for each of
    /// the `frames` sampled frames, first, then any crop rows.
    pub fn answer(&self, rows: &[Vec<f32>], frames: usize, compiled: &Compiled) -> anyhow::Result<Response> {
        anyhow::ensure!(!rows.is_empty(), "the video state has no frames");
        let mut answers = IndexMap::new();
        let video = self.head.prepare(rows, frames);
        for q in &compiled.questions {
            let probs = self.head.probabilities_for(&video, &q.question, &q.options);
            let answer = match &q.kind {
                Kind::Noul { negated } => Answer::Noul { noul: round4(if *negated { 1.0 - probs[0] } else { probs[0] }) },
                Kind::Choice { labels } => {
                    let best = argmax(&probs);
                    Answer::Choice {
                        choice: labels[best].clone(),
                        probabilities: labels.iter().cloned().zip(probs.iter().map(|&p| round4(p))).collect(),
                        confidence: round4(api::choice_confidence(&probs)),
                    }
                }
                Kind::Score { legend } => Answer::Score {
                    score: round4(api::score_value(&probs)),
                    legend: legend.iter().enumerate().map(|(i, l)| (i.to_string(), l.clone())).collect(),
                    probabilities: probs.iter().enumerate().map(|(i, &p)| (i.to_string(), round4(p))).collect(),
                    confidence: round4(api::score_confidence(&probs)),
                },
            };
            answers.insert(q.key.clone(), answer);
        }
        Ok(Response {
            model: self.model_name().to_string(),
            answers,
            usage: Usage { input_frames: frames, input_tokens: compiled.input_tokens, output_tokens: 0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_frames_are_spread_evenly() {
        assert_eq!(crop_frame_indices(16, 4), [2, 6, 10, 14]);
        assert_eq!(crop_frame_indices(8, 4), [1, 3, 5, 7]);
        assert_eq!(crop_frame_indices(32, 4), [4, 12, 20, 28]);
        assert_eq!(crop_frame_indices(2, 4), [0, 1]);
        assert_eq!(crop_frame_indices(16, 0), Vec::<usize>::new());
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, &x)| if x > best.1 { (i, x) } else { best }).0
}
