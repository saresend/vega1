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
//! across videos; frames are embedded once ([`SystemOne::embed_frame`]) and
//! can be re-asked any question set.

pub mod api;
pub mod encoders;
pub mod head;
pub mod image;
pub mod prompts;

use std::collections::HashMap;

use indexmap::IndexMap;

use api::{round4, Answer, Question, Request, Response, Usage};
use encoders::{ImageEncoder, TextEncoder};
use head::{Head, TrainedHead};
pub use image::Frame;

/// Raw model files; how they're fetched is the caller's business.
pub struct ModelFiles<'a> {
    pub vision_onnx: &'a [u8],
    pub text_onnx: &'a [u8],
    pub tokenizer_json: &'a [u8],
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
    Noul,
    Choice { labels: Vec<String> },
    Score { legend: Vec<String> },
}

pub struct SystemOne {
    image: ImageEncoder,
    text: TextEncoder,
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
            image: ImageEncoder::from_onnx_bytes(files.vision_onnx)?,
            text: TextEncoder::from_bytes(files.text_onnx, files.tokenizer_json)?,
            head,
            text_cache: HashMap::new(),
        })
    }

    pub fn model_name(&self) -> &str {
        self.head.name()
    }

    pub fn embed_frame(&self, frame: &Frame) -> anyhow::Result<Vec<f32>> {
        self.image.embed(frame)
    }

    fn embed_text(&mut self, context: &str, focus: &str) -> anyhow::Result<(Vec<f32>, usize)> {
        let key = (context.to_string(), focus.to_string());
        if let Some(hit) = self.text_cache.get(&key) {
            return Ok(hit.clone());
        }
        let value = self.text.embed_pair(context, focus)?;
        self.text_cache.insert(key, value.clone());
        Ok(value)
    }

    /// Embeds one question: (question embedding, option embeddings, tokens used).
    pub fn embed_question(&mut self, q: &Question) -> anyhow::Result<(Vec<f32>, Vec<Vec<f32>>, usize)> {
        let prompts = prompts::build(q);
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
            let (question, options, n) = self.embed_question(q)?;
            input_tokens += n;
            let kind = match q {
                Question::Noul { .. } => Kind::Noul,
                Question::Choice { criteria, .. } => Kind::Choice { labels: criteria.keys().cloned().collect() },
                Question::Score { criteria, .. } => Kind::Score { legend: criteria.iter().map(api::render).collect() },
            };
            questions.push(CompiledQuestion { key: key.clone(), kind, question, options });
        }
        Ok(Compiled { questions, input_tokens })
    }

    /// Answers a compiled question set for a video given its frame embeddings.
    pub fn answer(&self, frames: &[Vec<f32>], compiled: &Compiled) -> anyhow::Result<Response> {
        anyhow::ensure!(!frames.is_empty(), "the video state has no frames");
        let mut answers = IndexMap::new();
        for q in &compiled.questions {
            let probs = self.head.probabilities(frames, &q.question, &q.options);
            let answer = match &q.kind {
                Kind::Noul => Answer::Noul { noul: round4(probs[0]) },
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
            usage: Usage { input_frames: frames.len(), input_tokens: compiled.input_tokens, output_tokens: 0 },
        })
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, &x)| if x > best.1 { (i, x) } else { best }).0
}
