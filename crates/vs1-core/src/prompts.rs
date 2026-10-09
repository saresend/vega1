//! Turns a typed question into text the encoders can embed.
//!
//! Every question type reduces to "pick among described options":
//! * noul   → two options, the `true` and `false` descriptions,
//! * choice → one option per criteria entry,
//! * score  → one option per level (order only matters for `score`/`confidence`).
//!
//! Each option is a (context, focus) pair: context is the rendered
//! instructions, focus is what distinguishes this option.

use crate::api::{render, Question};

/// Reference hypothesis for a `noul` with no explicit `false` description.
/// A generic caption works better as a contrast than a negated statement,
/// which CLIP-style text encoders largely ignore.
pub const NOUL_DEFAULT_FALSE: &str = "a video";

#[derive(Debug, Clone, PartialEq)]
pub struct Prompts {
    /// The question itself, for heads that condition on it.
    pub question: String,
    pub options: Vec<OptionText>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptionText {
    pub context: String,
    pub focus: String,
}

pub fn build(q: &Question) -> Prompts {
    let instructions = render(q.instructions()).trim().to_string();
    let option = |context: &str, focus: String| OptionText { context: context.to_string(), focus };
    let options = match q {
        Question::Noul { criteria, .. } => {
            let crit = criteria.clone().unwrap_or_default();
            let yes = crit.yes.as_ref().map(render).filter(|s| !s.trim().is_empty());
            let no = crit.no.as_ref().map(render).filter(|s| !s.trim().is_empty());
            vec![
                option("", yes.unwrap_or_else(|| instructions.clone())),
                option("", no.unwrap_or_else(|| NOUL_DEFAULT_FALSE.to_string())),
            ]
        }
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(label, desc)| {
                let desc = render(desc);
                let focus = if desc.trim().is_empty() { label.clone() } else { format!("{label}: {desc}") };
                option(&instructions, focus)
            })
            .collect(),
        Question::Score { criteria, .. } => criteria.iter().map(|level| option(&instructions, render(level))).collect(),
    };
    Prompts { question: instructions, options }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Request;

    fn question(json: &str) -> Question {
        let req = Request::parse(&format!(r#"{{"questions": {{"q": {json}}}}}"#)).unwrap();
        req.questions["q"].clone()
    }

    #[test]
    fn noul_uses_criteria_or_falls_back() {
        let p = build(&question(r#"{"type": "noul", "instructions": "The video is captioned"}"#));
        assert_eq!(p.options[0].focus, "The video is captioned");
        assert_eq!(p.options[1].focus, NOUL_DEFAULT_FALSE);
        let p = build(&question(
            r#"{"type": "noul", "instructions": "Captioned?", "criteria": {"true": "text on screen", "false": "no text"}}"#,
        ));
        assert_eq!((p.options[0].focus.as_str(), p.options[1].focus.as_str()), ("text on screen", "no text"));
    }

    #[test]
    fn choice_and_score_keep_instructions_as_context() {
        let p = build(&question(r#"{"type": "choice", "instructions": "How many people?", "criteria": {"1": null, "2+": "a group"}}"#));
        assert_eq!(p.options[0], OptionText { context: "How many people?".into(), focus: "1".into() });
        assert_eq!(p.options[1].focus, "2+: a group");
        let p = build(&question(r#"{"type": "score", "instructions": "Pace", "criteria": ["slow", "fast"]}"#));
        assert_eq!(p.options.iter().map(|o| o.focus.as_str()).collect::<Vec<_>>(), ["slow", "fast"]);
    }
}
