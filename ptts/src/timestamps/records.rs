//! What an aligner emits, and the text it aligns against.
//!
//! Mirrors `_SourceWord`, `_TextUnit`, `WordStart` and `WordEnd` in
//! `pocket_tts_timestamped/timestamps/`. Field for field, so a reader can put
//! the two side by side — including `is_word` being derived rather than stored,
//! which is the one place the two could silently disagree.

/// One word of the input, as the aligner refers to it. The offsets are into the
/// chunk the model was given, which the caller maps back to its own text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceWord {
    pub text: String,
    /// Position among the input's words, across the whole input rather than
    /// this chunk.
    pub word_index: usize,
    pub begin: usize,
    pub end: usize,
}

/// One unit of the prompt as the model sees it: a word, or punctuation.
///
/// `synthetic` marks punctuation the pipeline added rather than the author —
/// the terminal period appended to a prompt that lacked one. It decides whether
/// a trailing silence ends the utterance or is a pause before punctuation that
/// was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextUnit {
    pub text: String,
    pub chunk_begin: usize,
    pub chunk_end: usize,
    pub word: Option<SourceWord>,
    pub synthetic: bool,
}

impl TextUnit {
    /// Derived, as in Python: a unit is a word exactly when it carries one.
    /// Stored separately, the two could disagree.
    pub fn is_word(&self) -> bool {
        self.word.is_some()
    }

    pub fn word(text: impl Into<String>, word_index: usize, begin: usize, end: usize) -> Self {
        let text = text.into();
        Self {
            text: text.clone(),
            chunk_begin: begin,
            chunk_end: end,
            word: Some(SourceWord { text, word_index, begin, end }),
            synthetic: false,
        }
    }

    pub fn punctuation(text: impl Into<String>, begin: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            chunk_begin: begin,
            chunk_end: end,
            word: None,
            synthetic: false,
        }
    }

    pub fn synthetic(mut self) -> Self {
        self.synthetic = true;
        self
    }
}

/// A word began being uttered.
#[derive(Debug, Clone, PartialEq)]
pub struct WordStart {
    pub text: String,
    pub word_index: usize,
    pub start: f64,
}

/// A word finished being uttered. Carries its start, so a consumer watching
/// only ends still has the whole span.
#[derive(Debug, Clone, PartialEq)]
pub struct WordEnd {
    pub text: String,
    pub word_index: usize,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WordEvent {
    Start(WordStart),
    End(WordEnd),
}

/// An aligned word together with its span in the prompt it came from.
///
/// Not a port of a Python record: there the span stays on the `_TextUnit` and
/// is mapped back to the caller's text in a later pass. A caller here is handed
/// the two together, because the span is the only reliable way to know which
/// text a timing belongs to — searching the prompt for the word can match an
/// earlier occurrence of a short word inside a longer one.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    pub word: WordEnd,
    /// Byte offsets into the prepared prompt for this word's chunk.
    pub begin: usize,
    pub end: usize,
}
