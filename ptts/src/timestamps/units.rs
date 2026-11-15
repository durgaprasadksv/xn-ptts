//! Turning a prompt into the units the alignment scores, and mapping each
//! token to one of them.
//!
//! Ported from the parts of `pocket_tts_timestamped/timestamps/text.py` that
//! this tokenizer path needs: `_lexical_word_spans` and the piece-to-word
//! assignment over the tokenizer's own offsets.
//!
//! Two things make this the real mechanism rather than a guess. Words are found
//! **lexically**, so an apostrophe or a hyphen between two word characters
//! keeps "isn't" and "well-worn" whole — splitting on whitespace is close but
//! not the same, and grouping pieces on SentencePiece's `▁` marker is wrong
//! outright, since a contraction is several pieces with punctuation in the
//! middle. And a piece is assigned to a word by **where the tokenizer says it
//! came from**, not by reconstructing it, so a piece that spans a boundary or
//! normalises to something else still lands in the right place.
//!
//! What is not ported is the layer above: `text.py` also maps a prepared chunk
//! back to the caller's original text through normalisation, capitalisation
//! variants and byte-fallback pieces. Here the alignment is against the prompt
//! the model was given, so that layer is not needed.

use crate::{Error, Result};

use super::records::TextUnit;

/// Characters that join two word characters into one word, as
/// `_WORD_CONTINUATIONS` in `text.py`: straight and curly apostrophes, hyphen
/// and its unicode variants.
const WORD_CONTINUATIONS: [char; 5] = ['-', '\u{2010}', '\u{2011}', '\'', '\u{2019}'];

/// `_is_combining_mark`, approximated by the ranges that occur in the
/// languages Pocket serves. Python asks unicode for the general category; that
/// needs a table this crate does not otherwise carry, and the difference would
/// only show on scripts outside these ranges.
fn is_combining_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05BD | 0x0610..=0x061A
        | 0x064B..=0x065F | 0x0670 | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF
        | 0x20D0..=0x20F0 | 0xFE20..=0xFE2F)
}

/// `_lexical_word_spans`: byte spans of the words in `text`.
///
/// A word is a run of alphanumerics, keeping any combining marks with their
/// base character, and continuing across a single `'` or `-` when a word
/// character follows it.
pub fn lexical_word_spans(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let end_of = |i: usize| chars.get(i).map(|&(b, _)| b).unwrap_or(text.len());

    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        if !chars[i].1.is_alphanumeric() {
            i += 1;
            continue;
        }
        let begin = chars[i].0;
        loop {
            while i < chars.len() && (chars[i].1.is_alphanumeric() || is_combining_mark(chars[i].1))
            {
                i += 1;
            }
            // A continuation only continues the word if a word character
            // follows it; a trailing apostrophe ends the word instead.
            if i + 1 < chars.len()
                && WORD_CONTINUATIONS.contains(&chars[i].1)
                && chars[i + 1].1.is_alphanumeric()
            {
                i += 1;
                continue;
            }
            break;
        }
        let end = end_of(i);
        if text[begin..end].chars().any(|c| c.is_alphanumeric()) {
            spans.push((begin, end));
        }
    }
    spans
}

/// The units of `text`, and which unit each token belongs to.
///
/// `spans` are the tokenizer's own byte offsets for `pieces`, which is what
/// makes the assignment reliable. A piece that overlaps a word belongs to that
/// word; one that overlaps none is punctuation, and punctuation is what lets
/// the alignment tell a pause before a comma from the end of the utterance.
///
/// The second return is `token_to_unit`: a frame's attention is one score per
/// token and the alignment wants one per unit, so the caller sums each unit's
/// tokens.
pub fn units_from_layout(
    text: &str,
    pieces: &[String],
    spans: &[(usize, usize)],
) -> Result<(Vec<TextUnit>, Vec<usize>)> {
    if pieces.len() != spans.len() {
        return Err(Error::invalid_data(format!("{} pieces but {} spans", pieces.len(), spans.len())));
    }
    let words = lexical_word_spans(text);
    if words.is_empty() {
        return Err(Error::invalid_data(format!("no words in {text:?}")));
    }

    let mut units: Vec<TextUnit> = Vec::new();
    let mut token_to_unit: Vec<usize> = Vec::with_capacity(pieces.len());
    // Which unit each word became, so a second piece of the same word finds it.
    let mut unit_of_word: Vec<Option<usize>> = vec![None; words.len()];

    for (_piece, &(begin, end)) in pieces.iter().zip(spans) {
        // The word this piece overlaps. A piece covering a boundary is
        // attributed to the first word it touches, as the offsets order them.
        let word = words.iter().position(|&(wb, we)| begin < we && end > wb);

        let unit = match word {
            Some(index) => match unit_of_word[index] {
                Some(unit) => unit,
                None => {
                    let (wb, we) = words[index];
                    units.push(TextUnit::word(&text[wb..we], index, wb, we));
                    unit_of_word[index] = Some(units.len() - 1);
                    units.len() - 1
                }
            },
            None => {
                let body = text.get(begin..end).unwrap_or("").trim();
                if body.is_empty() {
                    // Whitespace or a special token: its attention belongs to
                    // whatever unit is open rather than being dropped.
                    token_to_unit.push(units.len().saturating_sub(1));
                    continue;
                }
                units.push(TextUnit::punctuation(body, begin, end));
                units.len() - 1
            }
        };
        token_to_unit.push(unit);
    }

    let mapped = unit_of_word.iter().filter(|u| u.is_some()).count();
    if mapped != words.len() {
        let missed: Vec<&str> = words
            .iter()
            .zip(&unit_of_word)
            .filter(|(_, u)| u.is_none())
            .map(|(&(b, e), _)| &text[b..e])
            .collect();
        return Err(Error::invalid_data(format!(
            "{mapped} of {} words in {text:?} got no token: {missed:?}. The \
             tokenizer's offsets do not cover them, so their timings would be \
             invented.",
            words.len()
        )));
    }
    Ok((units, token_to_unit))
}

/// Sum a frame's per-token attention into per-unit scores, which is what the
/// alignment reads.
pub fn scores_by_unit(frame: &[f32], token_to_unit: &[usize], unit_count: usize) -> Vec<f32> {
    let mut scores = vec![0.0; unit_count];
    for (token, &unit) in token_to_unit.iter().enumerate() {
        if let Some(v) = frame.get(token) {
            if unit < scores.len() {
                scores[unit] += v;
            }
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words_of(text: &str) -> Vec<&str> {
        lexical_word_spans(text).into_iter().map(|(b, e)| &text[b..e]).collect()
    }

    #[test]
    fn a_contraction_is_one_word() {
        // The case that whitespace splitting gets right and `▁` grouping does
        // not, and the reason this is lexical.
        assert_eq!(words_of("isn't"), vec!["isn't"]);
        assert_eq!(words_of("I'm from Scotland"), vec!["I'm", "from", "Scotland"]);
    }

    #[test]
    fn a_hyphenated_word_is_one_word() {
        assert_eq!(words_of("well-worn"), vec!["well-worn"]);
        assert_eq!(words_of("well\u{2010}worn"), vec!["well\u{2010}worn"]);
    }

    #[test]
    fn a_continuation_needs_a_word_character_after_it() {
        // A trailing apostrophe or a dash between spaces ends the word.
        assert_eq!(words_of("readin' now"), vec!["readin", "now"]);
        assert_eq!(words_of("one - two"), vec!["one", "two"]);
    }

    #[test]
    fn punctuation_and_whitespace_are_not_words() {
        assert_eq!(words_of("Alba here, and yes."), vec!["Alba", "here", "and", "yes"]);
        assert!(words_of("... -- ...").is_empty());
    }

    #[test]
    fn combining_marks_stay_with_their_base_character() {
        let text = "cafe\u{301} noir";
        assert_eq!(words_of(text), vec!["cafe\u{301}", "noir"]);
    }

    /// Spans as a tokenizer would report them: each piece's byte range.
    fn layout(text: &str, pieces: &[&str]) -> (Vec<String>, Vec<(usize, usize)>) {
        let mut spans = Vec::new();
        let mut at = 0usize;
        for piece in pieces {
            let begin = text[at..].find(piece).map(|i| at + i).unwrap_or(at);
            spans.push((begin, begin + piece.len()));
            at = begin + piece.len();
        }
        (pieces.iter().map(|s| s.to_string()).collect(), spans)
    }

    #[test]
    fn a_contraction_split_across_pieces_is_one_unit() {
        // "isn't" arrives as three pieces, one of which is an apostrophe. All
        // three must score the same unit, or the word competes with itself.
        let text = "it isn't here";
        let (pieces, spans) = layout(text, &["it", "isn", "'", "t", "here"]);
        let (units, map) = units_from_layout(text, &pieces, &spans).unwrap();
        let words: Vec<&str> = units.iter().filter(|u| u.is_word()).map(|u| u.text.as_str()).collect();
        assert_eq!(words, vec!["it", "isn't", "here"]);
        assert_eq!(map, vec![0, 1, 1, 1, 2]);
    }

    #[test]
    fn punctuation_gets_its_own_unit() {
        let text = "one two.";
        let (pieces, spans) = layout(text, &["one", "two", "."]);
        let (units, map) = units_from_layout(text, &pieces, &spans).unwrap();
        assert_eq!(units.len(), 3);
        assert!(!units[2].is_word());
        assert_eq!(map, vec![0, 1, 2]);
    }

    #[test]
    fn word_offsets_are_into_the_given_text() {
        let text = "Alba here, and yes.";
        let (pieces, spans) = layout(text, &["Alba", "here", ",", "and", "yes", "."]);
        let (units, _) = units_from_layout(text, &pieces, &spans).unwrap();
        for unit in units.iter().filter(|u| u.is_word()) {
            assert_eq!(&text[unit.chunk_begin..unit.chunk_end], unit.text);
        }
    }

    #[test]
    fn a_word_no_piece_covers_is_an_error() {
        // Emitting timings for a word the tokenizer never produced would be
        // inventing them.
        let text = "one two three";
        let (pieces, spans) = layout(text, &["one", "three"]);
        let err = units_from_layout(text, &pieces, &spans).unwrap_err();
        assert!(format!("{err}").contains("two"), "{err}");
    }
}

#[cfg(test)]
mod reduce_tests {
    use super::*;

    #[test]
    fn a_words_tokens_sum_into_one_score() {
        let frame = [0.1, 0.3, 0.4, 0.2];
        let scores = scores_by_unit(&frame, &[0, 1, 1, 2], 3);
        // Approximately: 0.3f32 + 0.4f32 is 0.70000005, and scores are only
        // ever compared against each other.
        for (got, want) in scores.iter().zip([0.1f32, 0.7, 0.2]) {
            assert!((got - want).abs() < 1e-6, "{scores:?}");
        }
    }

    #[test]
    fn a_short_frame_does_not_panic() {
        assert_eq!(scores_by_unit(&[0.5], &[0, 1, 1, 2], 3), vec![0.5, 0.0, 0.0]);
    }
}
