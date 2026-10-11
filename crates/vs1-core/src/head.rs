//! Answer heads: turn (frame embeddings, question, option embeddings) into
//! option logits.
//!
//! * Zero-shot: each option's logit is its similarity to the video, pooled
//!   over frames with a log-mean-exp (a soft max), so evidence present in only
//!   some frames still counts.
//! * Trained: a small question-conditioned attention head over frames whose
//!   output is added to the zero-shot logits, which it also rescales per
//!   question (`s = 1 + 2 tanh(w·x)`, x the attention output or the question
//!   embedding: trust, discount or flip the zero-shot evidence, e.g. for
//!   negated statements). With `gamma = 0` and `w = 0` it reduces exactly to
//!   zero-shot, which is how training starts.
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

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
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

/// Input of the zero-shot scale: the attention output (`ws`/`bs`) or the
/// question embedding (`wsq`/`bsq`).
enum Scale {
    Hidden(Linear),
    Question(Linear),
}

/// Crop handling (heads trained with crop rows): the weights of the crop
/// soft-max and mean similarity terms, and optionally full/crop type vectors
/// added to rows so attention also runs over crops. Heads without it ignore
/// crop rows.
struct Crops {
    /// (full, crop) type vectors; without them attention sees full views only.
    types: Option<(Vec<f32>, Vec<f32>)>,
    a_peak: f32,
    a_mean: f32,
    beta: f32,
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
    /// Per-question zero-shot scale; absent in older heads (scale 1).
    scale: Option<Scale>,
    crops: Option<Crops>,
    a_zs: f32,
    gamma: f32,
    pool_beta: f32,
}

/// IEEE 754 half → single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalise.
            let shift = frac.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((frac << (shift + 1)) & 0x3ff) << 13
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
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
            ensure!(t.shape() == shape, "`{key}` has shape {:?}, expected {shape:?}", t.shape());
            match t.dtype() {
                Dtype::F32 => Ok(t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()),
                // Half-size heads for the web; computation stays in f32.
                Dtype::F16 => Ok(t.data().chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()),
                other => bail!("`{key}` must be f32 or f16, got {other:?}"),
            }
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
            scale: if st.names().contains(&&"ws".to_string()) {
                Some(Scale::Hidden(linear("ws", "bs", 1, hidden)?))
            } else if st.names().contains(&&"wsq".to_string()) {
                Some(Scale::Question(linear("wsq", "bsq", 1, EMBED_DIM)?))
            } else {
                None
            },
            crops: if st.names().contains(&&"a_crop".to_string()) {
                let a = tensor("a_crop", &[2])?;
                Some(Crops {
                    types: if st.names().contains(&&"type_crop".to_string()) {
                        Some((tensor("type_full", &[EMBED_DIM])?, tensor("type_crop", &[EMBED_DIM])?))
                    } else {
                        None
                    },
                    a_peak: a[0],
                    a_mean: a[1],
                    beta: scalar("crop_beta")?,
                })
            } else {
                None
            },
            a_zs: scalar("a_zs")?,
            gamma: scalar("gamma")?,
            pool_beta: scalar("pool_beta")?,
        })
    }

    /// Per-video work shared by every question: rows tagged with their type
    /// (or, without crop support, the full views only) through `wk` and `wv`.
    fn project(&self, full: &[Vec<f32>], crop: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let typed: Vec<Vec<f32>> = match self.crops.as_ref().and_then(|c| c.types.as_ref()) {
            Some((type_full, type_crop)) => full
                .iter()
                .map(|r| add(r, type_full))
                .chain(crop.iter().map(|r| add(r, type_crop)))
                .collect(),
            None => full.to_vec(),
        };
        (typed.iter().map(|r| self.wk.forward(r)).collect(), typed.iter().map(|r| self.wv.forward(r)).collect())
    }

    fn logits(&self, video: &Prepared, question: &[f32], options: &[Vec<f32>]) -> Vec<f32> {
        let (full, crop) = (video.full, video.crop);
        let hidden = self.ln_g.len();
        let qh = self.wq.forward(question);
        let scale = 1.0 / (hidden as f32).sqrt();
        let keys: Vec<f32> = video.keys.iter().map(|k| dot(k, &qh) * scale).collect();
        let attn = softmax(&keys);
        let mut h = qh.clone();
        for (v, a) in video.values.iter().zip(&attn) {
            for (hi, vi) in h.iter_mut().zip(v) {
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
        let scale = match &self.scale {
            None => 1.0,
            Some(Scale::Hidden(w)) => 1.0 + 2.0 * w.forward(&h2)[0].tanh(),
            Some(Scale::Question(w)) => 1.0 + 2.0 * w.forward(question)[0].tanh(),
        };
        options
            .iter()
            .map(|o| {
                let mut logit = self.a_zs * scale * pooled_similarity(full, o, self.pool_beta) + self.gamma * dot(&z, o);
                if let (Some(c), false) = (&self.crops, crop.is_empty()) {
                    let mean = crop.iter().map(|r| dot(r, o)).sum::<f32>() / crop.len() as f32;
                    logit += c.a_peak * pooled_similarity(crop, o, c.beta) + c.a_mean * mean;
                }
                logit
            })
            .collect()
    }
}

/// A video state ready to be asked any number of questions: its rows split
/// into full views and crops, with the per-row projections a trained head
/// would otherwise redo for every question.
pub struct Prepared<'a> {
    full: &'a [Vec<f32>],
    crop: &'a [Vec<f32>],
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
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

    /// `rows`: the video state, its first `n_full` rows full views and the
    /// rest crops. Zero-shot scoring uses the full views only (crops shift
    /// every similarity, generic captions most).
    pub fn prepare<'a>(&self, rows: &'a [Vec<f32>], n_full: usize) -> Prepared<'a> {
        let (full, crop) = rows.split_at(n_full.min(rows.len()));
        let (keys, values) = match self {
            Head::ZeroShot { .. } => (Vec::new(), Vec::new()),
            Head::Trained(h) => h.project(full, crop),
        };
        Prepared { full, crop, keys, values }
    }

    pub fn probabilities(&self, rows: &[Vec<f32>], n_full: usize, question: &[f32], options: &[Vec<f32>]) -> Vec<f32> {
        self.probabilities_for(&self.prepare(rows, n_full), question, options)
    }

    pub fn probabilities_for(&self, video: &Prepared, question: &[f32], options: &[Vec<f32>]) -> Vec<f32> {
        let logits = match self {
            Head::ZeroShot { scale, pool_beta } => {
                options.iter().map(|o| scale * pooled_similarity(video.full, o, *pool_beta)).collect()
            }
            Head::Trained(h) => h.logits(video, question, options),
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
    fn half_floats_convert_exactly() {
        for (h, f) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x3555, 0.333_251_95), (0x0001, 5.960_464_5e-8), (0x7bff, 65504.0), (0, 0.0)] {
            assert_eq!(f16_to_f32(h), f, "{h:#x}");
        }
    }

    #[test]
    fn zero_shot_prefers_matching_option() {
        let p = Head::default().probabilities(&[unit(4), unit(5)], 1, &unit(0), &[unit(4), unit(5)]);
        assert!(p[0] > 0.99 && (p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
