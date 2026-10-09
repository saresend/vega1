//! Jev-compatible request/response types (TypeSafe "System One" API), with a
//! video as the `state`.
//!
//! Wire format follows https://docs.typesafe.ai/api: a map of typed questions
//! (`noul`, `choice`, `score`) in, a map of typed answers out, same keys.
//! `score` and `confidence` use the formulas from docs.typesafe.ai/confidence
//! and /primitives/score.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_CHOICE_OPTIONS: usize = 255;
pub const MIN_SCORE_LEVELS: usize = 2;
pub const MAX_SCORE_LEVELS: usize = 10;

/// The request minus `state`: the video is supplied separately as frames.
#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    #[serde(default)]
    pub model: Option<String>,
    pub questions: IndexMap<String, Question>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    Noul {
        instructions: Value,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        instructions: Value,
        criteria: IndexMap<String, Value>,
    },
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(rename = "true", default)]
    pub yes: Option<Value>,
    #[serde(rename = "false", default)]
    pub no: Option<Value>,
}

impl Question {
    pub fn instructions(&self) -> &Value {
        match self {
            Question::Noul { instructions, .. }
            | Question::Choice { instructions, .. }
            | Question::Score { instructions, .. } => instructions,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_frames: usize,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        probabilities: IndexMap<String, f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        legend: IndexMap<String, String>,
        probabilities: IndexMap<String, f32>,
        confidence: f32,
    },
}

/// A 422-style validation failure.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ValidationError {
    pub question: Option<String>,
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.question {
            Some(q) => write!(f, "questions.{q}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for ValidationError {}

impl Request {
    pub fn parse(json: &str) -> Result<Self, ValidationError> {
        let req: Request = serde_json::from_str(json).map_err(|e| ValidationError { question: None, message: e.to_string() })?;
        req.validate()?;
        Ok(req)
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        let fail = |key: &str, message: String| Err(ValidationError { question: Some(key.to_string()), message });
        if self.questions.is_empty() {
            return Err(ValidationError { question: None, message: "questions must not be empty".into() });
        }
        for (key, q) in &self.questions {
            if render(q.instructions()).trim().is_empty() {
                return fail(key, "instructions must not be empty".into());
            }
            match q {
                Question::Noul { .. } => {}
                Question::Choice { criteria, .. } => {
                    if criteria.is_empty() || criteria.len() > MAX_CHOICE_OPTIONS {
                        return fail(key, format!("choice needs 1 to {MAX_CHOICE_OPTIONS} options, got {}", criteria.len()));
                    }
                }
                Question::Score { criteria, .. } => {
                    if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&criteria.len()) {
                        return fail(
                            key,
                            format!("score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {}", criteria.len()),
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

/// Flattens a string/object/array instruction or criterion into plain text.
pub fn render(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(_) | Value::Number(_) => v.to_string(),
        Value::Array(items) => items.iter().map(render).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("; "),
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| format!("{k}: {}", render(v)))
            .collect::<Vec<_>>()
            .join("; "),
    }
}

/// Choice confidence: how far the top probability sits above uniform.
pub fn choice_confidence(probs: &[f32]) -> f32 {
    let n = probs.len() as f32;
    if probs.len() < 2 {
        return 1.0;
    }
    let p_max = probs.iter().cloned().fold(0.0, f32::max);
    ((p_max - 1.0 / n) / (1.0 - 1.0 / n)).clamp(0.0, 1.0)
}

/// Score value: probability-weighted level number (levels are 0-based).
pub fn score_value(probs: &[f32]) -> f32 {
    probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum()
}

/// Score confidence: 1 − (mean absolute distance from the modal level) /
/// (the same quantity for a uniform distribution), floored at 0.
pub fn score_confidence(probs: &[f32]) -> f32 {
    let n = probs.len();
    let mode = probs
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &p)| if p > best.1 { (i, p) } else { best })
        .0;
    let spread: f32 = probs.iter().enumerate().map(|(i, p)| p * (i as f32 - mode as f32).abs()).sum();
    let center = (n as f32 - 1.0) / 2.0;
    let mad_uniform: f32 = (0..n).map(|i| (i as f32 - center).abs()).sum::<f32>() / n as f32;
    if mad_uniform == 0.0 {
        return 1.0;
    }
    (1.0 - spread / mad_uniform).max(0.0)
}

/// Rounds for the wire so answers read like Jev's (e.g. 0.57, not 0.5700012).
pub fn round4(x: f32) -> f32 {
    (x * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.005
    }

    #[test]
    fn confidence_matches_jev_docs_examples() {
        // docs.typesafe.ai/confidence
        assert!(close(choice_confidence(&[0.6, 0.3, 0.1]), 0.4));
        assert!(close(choice_confidence(&[0.6, 0.2, 0.2]), 0.4));
        assert!(close(score_confidence(&[0.0, 0.5, 0.5]), 0.25));
        assert!(close(score_confidence(&[0.5, 0.0, 0.5]), 0.0));
        assert!((score_confidence(&[0.0, 0.57, 0.43]) - 0.355).abs() < 1e-3, "docs: about 0.35");
        // docs.typesafe.ai/primitives/score
        assert!(close(score_value(&[0.0, 0.57, 0.43]), 1.43));
    }

    #[test]
    fn parses_all_question_types() {
        let req = Request::parse(
            r#"{
              "model": "vs1-latest",
              "questions": {
                "captioned": {"type": "noul", "instructions": "The video is captioned",
                              "criteria": {"true": "Captions on screen", "false": "No captions"}},
                "people": {"type": "choice", "instructions": "How many people?",
                           "criteria": {"0": null, "1": null, "2 or more": "Several people"}},
                "energy": {"type": "score", "instructions": {"question": "How energetic?", "note": "pace"},
                           "criteria": ["Calm", "Moderate", "Hectic"]}
              }
            }"#,
        )
        .unwrap();
        let keys: Vec<_> = req.questions.keys().cloned().collect();
        assert_eq!(keys, ["captioned", "people", "energy"], "order preserved");
        assert_eq!(render(req.questions["energy"].instructions()), "question: How energetic?; note: pace");
    }

    #[test]
    fn rejects_invalid_questions() {
        let one_level = r#"{"questions": {"s": {"type": "score", "instructions": "x", "criteria": ["only"]}}}"#;
        assert_eq!(Request::parse(one_level).unwrap_err().question.as_deref(), Some("s"));
        let empty = r#"{"questions": {"n": {"type": "noul", "instructions": "  "}}}"#;
        assert!(Request::parse(empty).is_err());
        let unknown_type = r#"{"questions": {"n": {"type": "essay", "instructions": "x"}}}"#;
        assert!(Request::parse(unknown_type).is_err());
        assert!(Request::parse(r#"{"questions": {}}"#).is_err());
    }

    #[test]
    fn answers_serialize_like_jev() {
        let a = Answer::Noul { noul: 0.88 };
        assert_eq!(serde_json::to_string(&a).unwrap(), r#"{"type":"noul","noul":0.88}"#);
        let mut probs = IndexMap::new();
        probs.insert("a".to_string(), 0.7);
        probs.insert("b".to_string(), 0.3);
        let c = Answer::Choice { choice: "a".into(), probabilities: probs, confidence: 0.4 };
        assert_eq!(
            serde_json::to_string(&c).unwrap(),
            r#"{"type":"choice","choice":"a","probabilities":{"a":0.7,"b":0.3},"confidence":0.4}"#
        );
    }
}
