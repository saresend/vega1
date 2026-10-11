//! Turns a typed question into text the encoders can embed.
//!
//! Every question type reduces to "pick among described options":
//! * noul   → two options, the `true` and `false` descriptions,
//! * choice → one option per criteria entry,
//! * score  → one option per level (order only matters for `score`/`confidence`).
//!
//! Each option is a (context, focus) pair: context is the rendered
//! instructions, focus is what distinguishes this option.
//!
//! A negated `noul` statement ("there is no dog") is answered as the
//! complement of its positive form ("there is a dog"): CLIP-style text
//! encoders largely ignore negation, so the two would otherwise embed almost
//! identically and get the same answer.

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
    /// `noul` only: the options describe the statement's positive form, so the
    /// answer is the complement of the first option's probability.
    pub negated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptionText {
    pub context: String,
    pub focus: String,
}

pub fn build(q: &Question) -> Prompts {
    let mut instructions = render(q.instructions()).trim().to_string();
    let mut negated = false;
    let option = |context: &str, focus: String| OptionText { context: context.to_string(), focus };
    let options = match q {
        Question::Noul { criteria, .. } => {
            let crit = criteria.clone().unwrap_or_default();
            let yes = crit.yes.as_ref().map(render).filter(|s| !s.trim().is_empty());
            let no = crit.no.as_ref().map(render).filter(|s| !s.trim().is_empty());
            if yes.is_none() && no.is_none() {
                if let Some(positive) = positive_form(&instructions) {
                    negated = positive.negated;
                    instructions = positive.text;
                }
            }
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
    Prompts { question: instructions, options, negated }
}

#[derive(Debug, PartialEq)]
pub struct Positive {
    /// The statement with its negations removed.
    pub text: String,
    /// Whether it had an odd number of negations (double negatives cancel).
    pub negated: bool,
}

/// Rewrites a statement without its negations, or `None` if it has none.
/// The rewrite only needs to be close enough for a text encoder: "There is no
/// dog" → "There is a dog", "Nobody is swimming" → "someone is swimming",
/// "The video isn't captioned" → "The video is captioned".
pub fn positive_form(text: &str) -> Option<Positive> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut negations = 0;
    let mut i = 0;
    while i < words.len() {
        // Split off surrounding punctuation so "dog?" and "(not" still match.
        let raw = words[i];
        let start = raw.find(|c: char| c.is_alphanumeric()).unwrap_or(raw.len());
        let end = raw.rfind(|c: char| c.is_alphanumeric()).map_or(start, |e| e + 1).max(start);
        let (pre, core, post) = (&raw[..start], &raw[start..end], &raw[end..]);
        let lower = core.to_lowercase().replace('’', "'");
        let replacement: Option<String> = match lower.as_str() {
            "not" | "never" => Some(String::new()),
            "cannot" => Some("can".into()),
            "nobody" | "noone" | "no-one" => Some("someone".into()),
            "nothing" => Some("something".into()),
            "nowhere" => Some("somewhere".into()),
            "none" => Some("some".into()),
            "without" => Some("with".into()),
            "no" => {
                let next = words.get(i + 1).map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase());
                match next.as_deref() {
                    Some("one") => {
                        i += 1;
                        Some("someone".into())
                    }
                    Some(n) if n.ends_with('s') || ["people", "children", "men", "women"].contains(&n) => Some(String::new()),
                    Some(_) => Some("a".into()),
                    None => Some(String::new()),
                }
            }
            "can't" => Some("can".into()),
            "won't" => Some("will".into()),
            "ain't" => Some("is".into()),
            w if w.ends_with("n't") && w.chars().count() > 3 => {
                // Drop the last three characters ("n't" / "n’t") of the original spelling.
                let cut = core.char_indices().rev().nth(2).map_or(0, |(i, _)| i);
                Some(core[..cut].to_string())
            }
            _ => None,
        };
        match replacement {
            Some(r) => {
                negations += 1;
                let word = format!("{pre}{r}{post}");
                if !r.is_empty() || !(pre.is_empty() && post.is_empty()) {
                    out.push(word.trim().to_string());
                }
            }
            None => out.push(raw.to_string()),
        }
        i += 1;
    }
    if negations == 0 {
        return None;
    }
    let mut text = out.into_iter().filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
    if text.chars().next().is_some_and(|c| c.is_lowercase()) && words[0].chars().next().is_some_and(|c| c.is_uppercase()) {
        text = text[..1].to_uppercase() + &text[1..];
    }
    Some(Positive { text, negated: negations % 2 == 1 })
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
    fn negated_statements_use_their_positive_form() {
        let pos = |s: &str| positive_form(s).map(|p| (p.text, p.negated));
        assert_eq!(pos("There is no dog"), Some(("There is a dog".into(), true)));
        assert_eq!(pos("There are no people in the video"), Some(("There are people in the video".into(), true)));
        assert_eq!(pos("Nobody is swimming."), Some(("Someone is swimming.".into(), true)));
        assert_eq!(pos("No one is visible"), Some(("Someone is visible".into(), true)));
        assert_eq!(pos("The video isn't captioned"), Some(("The video is captioned".into(), true)));
        assert_eq!(pos("The video doesn’t show a cat?"), Some(("The video does show a cat?".into(), true)));
        assert_eq!(pos("It is not raining"), Some(("It is raining".into(), true)));
        assert_eq!(pos("A man without a hat"), Some(("A man with a hat".into(), true)));
        assert_eq!(pos("It isn't true that nobody is here"), Some(("It is true that someone is here".into(), false)));
        assert_eq!(pos("There is a dog"), None);
        assert_eq!(pos("A knotted rope and nothingness"), None);

        let p = build(&question(r#"{"type": "noul", "instructions": "There is no dog"}"#));
        assert!(p.negated);
        assert_eq!((p.question.as_str(), p.options[0].focus.as_str()), ("There is a dog", "There is a dog"));
        // Explicit criteria are the caller's options: no rewriting.
        let p = build(&question(r#"{"type": "noul", "instructions": "No dog", "criteria": {"true": "no dog", "false": "a dog"}}"#));
        assert!(!p.negated);
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
