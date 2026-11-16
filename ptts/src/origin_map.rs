//! Mapping a timing back onto the text the caller passed, before normalization.
//!
//! [`crate::timestamps`] reports a word's span in the *prepared prompt* -- the string the
//! model was actually fed, after normalization and [`prepare_text_prompt`]. A caller holding
//! the original text cannot use that: normalization rewrites `$2` as "2 dollars" and `3:30`
//! as "3 30", so the spans address a string the caller never saw. This module carries each
//! timing the rest of the way, to Unicode scalar offsets into the original.
//!
//! The chain is four steps, and only the third is interesting:
//!
//! 1. prepared byte -> chunk byte, by [`prepared_to_chunk`]. The two differ only by the case
//!    of the first letter and a closing mark the preparation may add.
//! 2. chunk byte -> which whitespace-separated word of the normalized chunk.
//! 3. normalized word -> original word, by [`align_content`].
//! 4. original word -> its scalar offsets, which [`OrigWord`] already carries.
//!
//! Step 3 is an edit-distance alignment over *content characters* -- letters and digits,
//! lowercased, punctuation and spaces dropped. Aligning characters rather than words is what
//! survives a rewrite changing the word count: the letters of "2 dollars" still match the `2`
//! of `$2`, and the spoken words that matched nothing ("dollars") inherit their neighbour.
//! Trailing original text is free in the cost, so a chunk matches its own sentence instead of
//! dragging in the rest of the input.
//!
//! Ported from the fork's earlier `align.rs`, whose walk [`crate::timestamps`] replaced; this
//! mapping half had no counterpart in the reference, which reports spans into the prompt and
//! leaves the rest to its caller. It is kept in its own module, rather than folded into
//! `timestamps/`, so that the ported alignment stays a faithful copy.

use crate::plan::Chunk;
use crate::tts_model::prepare_text_prompt;

/// A word of the caller's text, with both offset kinds the mapping needs.
#[derive(Clone, Debug)]
struct OrigWord {
    word: String,
    start_byte: usize,
    end_byte: usize,
    /// Unicode scalar offsets, which are what a caller is handed.
    start: usize,
    end: usize,
}

/// One chunk's share of the mapping.
struct ChunkOrigins {
    /// Byte in the chunk's normalized text for each byte of its prepared prompt.
    prepared_at: Vec<Option<usize>>,
    /// The chunk's whitespace-separated words, as byte ranges into its normalized text.
    spoken: Vec<(usize, usize)>,
    /// The original word each of those landed on.
    origins: Vec<Option<u32>>,
}

/// Which word of the caller's text each chunk's spoken words came from.
pub struct OriginMap {
    words: Vec<OrigWord>,
    chunks: Vec<ChunkOrigins>,
}

impl OriginMap {
    /// Build the mapping for one utterance.
    ///
    /// `text` is the string the caller passed, before normalization. `chunks` are the chunks
    /// generated from it, in order, whose `text` is the normalized piece each covers.
    pub fn new(text: &str, chunks: &[Chunk]) -> Self {
        let words = words_in(text);
        let orig_content = content_of(text, &words);
        let mut cursor = 0usize;
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let spoken = whitespace_words(&chunk.text);
            let spoken_content = content_of(&chunk.text, &spoken_spans(&chunk.text));
            let (origins, advance) =
                align_content(&orig_content[cursor.min(orig_content.len())..], &spoken_content, spoken.len());
            cursor = (cursor + advance).min(orig_content.len());
            let prepared = prepare_text_prompt(&chunk.text).0;
            let prepared_at = prepared_to_chunk(&chunk.text, &prepared);
            out.push(ChunkOrigins { prepared_at, spoken, origins });
        }
        Self { words, chunks: out }
    }

    /// The caller-text scalar offsets and spelling for a word spanning `begin..end` bytes of
    /// `chunk`'s prepared prompt, or `None` when the span maps to no original word.
    ///
    /// A piece often begins on the space before its word (`▁the`), so the first byte can sit
    /// between words; the span is walked until a byte lands inside one.
    pub fn span_of(&self, chunk: usize, begin: usize, end: usize) -> Option<(usize, usize, &str)> {
        let chunk = self.chunks.get(chunk)?;
        let index = (begin..end).find_map(|byte| {
            let chunk_byte = chunk.prepared_at.get(byte).copied().flatten()?;
            let spoken = word_index(&chunk.spoken, chunk_byte)?;
            chunk.origins.get(spoken).copied().flatten()
        })?;
        let word = self.words.get(index as usize)?;
        Some((word.start, word.end, word.word.as_str()))
    }
}

struct Content {
    ch: char,
    word: u32,
}

fn content_of(text: &str, words: &[OrigWord]) -> Vec<Content> {
    let mut out = Vec::new();
    for (index, word) in words.iter().enumerate() {
        for ch in text[word.start_byte..word.end_byte].chars() {
            if let Some(ch) = fold_char(ch) {
                out.push(Content { ch, word: index as u32 });
            }
        }
    }
    out
}

/// Align `spoken` content onto a window of `orig`, starting at the beginning of `orig`.
///
/// Returns the original word each spoken word landed on, and how many `orig` content
/// characters the match consumed.
fn align_content(
    orig: &[Content],
    spoken: &[Content],
    n_spoken_words: usize,
) -> (Vec<Option<u32>>, usize) {
    let window = spoken.len().saturating_mul(4).saturating_add(48).max(spoken.len());
    let orig = &orig[..window.min(orig.len())];
    let (n, m) = (orig.len(), spoken.len());
    if m == 0 || n_spoken_words == 0 {
        return (vec![None; n_spoken_words], 0);
    }
    // dp[i][j]: cost of consuming orig[..i] and spoken[..j]. Trailing orig is free, so the
    // match can stop at the first sentence instead of dragging in the rest of the text.
    let cols = m + 1;
    let mut dp = vec![0u32; (n + 1) * cols];
    let mut from = vec![0u8; (n + 1) * cols];
    let at = |i: usize, j: usize| i * cols + j;
    for i in 1..=n {
        dp[at(i, 0)] = i as u32;
        from[at(i, 0)] = 1; // delete orig
    }
    for j in 1..=m {
        dp[at(0, j)] = j as u32;
        from[at(0, j)] = 2; // insert spoken
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = dp[at(i - 1, j - 1)] + u32::from(orig[i - 1].ch != spoken[j - 1].ch);
            let del = dp[at(i - 1, j)] + 1;
            let ins = dp[at(i, j - 1)] + 1;
            let (cost, how) = if sub <= del && sub <= ins {
                (sub, 0u8)
            } else if del <= ins {
                (del, 1)
            } else {
                (ins, 2)
            };
            dp[at(i, j)] = cost;
            from[at(i, j)] = how;
        }
    }
    let mut end_i = 0;
    for i in 1..=n {
        if dp[at(i, m)] < dp[at(end_i, m)] {
            end_i = i;
        }
    }
    // Backtrack. `matched[spoken char]` is the original word its matched char landed on.
    let mut matched: Vec<Option<u32>> = vec![None; m];
    let (mut i, mut j) = (end_i, m);
    while i > 0 || j > 0 {
        let how = if i == 0 {
            2
        } else if j == 0 {
            1
        } else {
            from[at(i, j)]
        };
        match how {
            0 => {
                i -= 1;
                j -= 1;
                if orig[i].ch == spoken[j].ch {
                    matched[j] = Some(orig[i].word);
                }
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    let mut origins = vec![None; n_spoken_words];
    let mut counts: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n_spoken_words];
    for (item, word) in spoken.iter().zip(&matched) {
        if let Some(word) = word {
            let slot = &mut counts[item.word as usize];
            if let Some(pair) = slot.iter_mut().find(|pair| pair.0 == *word) {
                pair.1 += 1;
            } else {
                slot.push((*word, 1));
            }
        }
    }
    let mut previous = None;
    for (index, votes) in counts.iter().enumerate() {
        let best = votes.iter().max_by_key(|pair| pair.1).map(|pair| pair.0).or(previous);
        origins[index] = best;
        if best.is_some() {
            previous = best;
        }
    }
    // A spoken word with no matched letter, such as "dollars" read out of "$2", keeps the
    // previous word. Leading ones take the next word that did match.
    let mut following = None;
    for origin in origins.iter_mut().rev() {
        if origin.is_none() {
            *origin = following;
        }
        if origin.is_some() {
            following = *origin;
        }
    }
    (origins, end_i)
}

/// Byte offset in `chunk` for each byte of `prepared`, which is [`prepare_text_prompt`] of it.
///
/// The two differ by the case of the first letter and a closing mark `prepared` may gain.
/// Bytes `prepared` added map to `None`.
fn prepared_to_chunk(chunk: &str, prepared: &str) -> Vec<Option<usize>> {
    let mut map = vec![None; prepared.len()];
    let mut chunk_chars = chunk.char_indices().peekable();
    let mut prepared_at = 0usize;
    if let Some(&(chunk_byte, chunk_char)) = chunk_chars.peek() {
        let upper: String = chunk_char.to_uppercase().collect();
        let mut buf = [0u8; 4];
        let as_written = chunk_char.encode_utf8(&mut buf).to_string();
        let take = if prepared[prepared_at..].starts_with(&upper) {
            Some(upper.len())
        } else if prepared[prepared_at..].starts_with(&as_written) {
            Some(as_written.len())
        } else {
            None
        };
        if let Some(take) = take {
            map[prepared_at..prepared_at + take].fill(Some(chunk_byte));
            prepared_at += take;
            chunk_chars.next();
        }
    }
    while prepared_at < prepared.len() {
        let Some(&(chunk_byte, chunk_char)) = chunk_chars.peek() else { break };
        let prepared_char = prepared[prepared_at..].chars().next().unwrap();
        if prepared_char != chunk_char {
            break;
        }
        let width = prepared_char.len_utf8();
        map[prepared_at..prepared_at + width].fill(Some(chunk_byte));
        prepared_at += width;
        chunk_chars.next();
    }
    map
}

fn word_index(words: &[(usize, usize)], byte: usize) -> Option<usize> {
    let index = words.partition_point(|&(start, _)| start <= byte);
    let index = index.checked_sub(1)?;
    (byte < words[index].1).then_some(index)
}

fn whitespace_words(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(start) = start.take() {
                out.push((start, index));
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(start) = start {
        out.push((start, text.len()));
    }
    out
}

/// Whitespace words as [`OrigWord`]s, so [`content_of`] can name them. The scalar offsets are
/// unused here; the word index is the position in this vec.
fn spoken_spans(text: &str) -> Vec<OrigWord> {
    whitespace_words(text)
        .into_iter()
        .enumerate()
        .map(|(index, (start, end))| OrigWord {
            word: text[start..end].to_string(),
            start_byte: start,
            end_byte: end,
            start: index,
            end: index + 1,
        })
        .collect()
}

fn fold_char(ch: char) -> Option<char> {
    ch.to_lowercase().next().filter(|ch| ch.is_alphanumeric())
}

/// The words of `text` a caller would highlight: letter and number runs, with an internal
/// hyphen, dot, underscore or apostrophe kept when a letter or number follows it.
fn words_in(text: &str) -> Vec<OrigWord> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if !chars[index].1.is_alphanumeric() {
            index += 1;
            continue;
        }
        let mut start_byte = chars[index].0;
        index += 1;
        while index < chars.len() {
            if chars[index].1.is_alphanumeric() {
                index += 1;
                continue;
            }
            let internal = matches!(chars[index].1, '-' | '.' | '_' | '\'' | '\u{2019}');
            if internal && chars.get(index + 1).is_some_and(|(_, ch)| ch.is_alphanumeric()) {
                index += 1;
                continue;
            }
            break;
        }
        let end_byte = chars.get(index).map(|(byte, _)| *byte).unwrap_or(text.len());
        // A currency sign glued to the front is part of the word the caller wants highlighted.
        if let Some((sym_at, sym)) = text[..start_byte].char_indices().next_back()
            && matches!(sym, '$' | '£' | '€' | '¥' | '₹' | '₩' | '#')
            && sym_at + sym.len_utf8() == start_byte
        {
            start_byte = sym_at;
        }
        let start = text[..start_byte].chars().count();
        let word = text[start_byte..end_byte].to_string();
        let end = start + word.chars().count();
        out.push(OrigWord { word, start_byte, end_byte, start, end });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk carrying only the normalized text, which is all this module reads.
    fn chunk(text: &str) -> Chunk {
        Chunk {
            text: text.to_string(),
            tokens: Vec::new(),
            frames_after_eos: 0,
            frame_budget: 0,
        }
    }

    /// Where `word` sits in the prompt the model would be fed for `chunk_text`.
    fn prepared_span(chunk_text: &str, word: &str) -> (usize, usize) {
        let prepared = prepare_text_prompt(chunk_text).0;
        let at = prepared.find(word).expect("word is in the prepared prompt");
        (at, at + word.len())
    }

    #[test]
    fn a_word_maps_to_its_own_scalar_offsets_when_nothing_was_rewritten() {
        let text = "The road climbs east.";
        let map = OriginMap::new(text, &[chunk(text)]);
        let (begin, end) = prepared_span(text, "road");
        let (start, stop, word) = map.span_of(0, begin, end).expect("road maps back");
        assert_eq!(word, "road");
        assert_eq!((start, stop), (4, 8));
        assert_eq!(text.chars().skip(start).take(stop - start).collect::<String>(), "road");
    }

    #[test]
    fn both_words_of_a_rewritten_amount_map_back_to_the_original() {
        // What the normalizer does to `$2`, and the case the offsets exist for: the caller
        // never saw "dollars", so the only honest answer for it is the `$2` it was read from.
        let text = "It cost $2 today.";
        let spoken = "It cost 2 dollars today.";
        let map = OriginMap::new(text, &[chunk(spoken)]);
        let expected = text.find("$2").unwrap();
        let expected = (
            text[..expected].chars().count(),
            text[..expected].chars().count() + 2,
        );
        for word in ["2", "dollars"] {
            let (begin, end) = prepared_span(spoken, word);
            let (start, stop, got) = map.span_of(0, begin, end).expect("maps back");
            assert_eq!((start, stop), expected, "{word} should point at $2");
            assert_eq!(got, "$2");
        }
    }

    #[test]
    fn offsets_count_scalars_and_not_bytes() {
        // `é` is two bytes and one scalar, so a byte offset would report "road" one too late.
        let text = "Café road ahead.";
        let map = OriginMap::new(text, &[chunk(text)]);
        let (begin, end) = prepared_span(text, "road");
        let (start, stop, word) = map.span_of(0, begin, end).expect("road maps back");
        assert_eq!(word, "road");
        assert_eq!(text.find("road"), Some(6), "the byte offset differs from the scalar one");
        assert_eq!((start, stop), (5, 9));
        assert_eq!(text.chars().skip(start).take(stop - start).collect::<String>(), "road");
    }

    #[test]
    fn a_word_of_a_later_chunk_maps_into_the_later_part_of_the_text() {
        // The cursor must advance with the chunks, or a repeated word matches the first one.
        let text = "The road is long. The road is open.";
        let chunks = [chunk("The road is long."), chunk("The road is open.")];
        let map = OriginMap::new(text, &chunks);
        let (begin, end) = prepared_span("The road is open.", "road");
        let (start, stop, word) = map.span_of(1, begin, end).expect("second road maps back");
        assert_eq!(word, "road");
        assert_eq!((start, stop), (22, 26));
        assert_eq!(text.chars().skip(start).take(stop - start).collect::<String>(), "road");
    }

    #[test]
    fn a_span_that_maps_nowhere_is_refused_rather_than_guessed() {
        let map = OriginMap::new("Hello.", &[chunk("Hello.")]);
        assert!(map.span_of(7, 0, 3).is_none(), "no such chunk");
    }
}
