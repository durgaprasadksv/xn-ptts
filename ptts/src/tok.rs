//! A [`crate::Tokenizer`] backed by Hugging Face [`tokenizers`].
//!
//! Each checkpoint supplies its own `tokenizer.json`. No tokenizer is bundled or guessed.
//!
//! Available with the `hf` feature.

/// A Hugging Face tokenizer.
pub struct Tok(tokenizers::Tokenizer);

impl Tok {
    /// Opens the checkpoint's explicitly supplied Hugging Face tokenizer JSON.
    pub fn open(path: &std::path::Path) -> xn::Result<Self> {
        tracing::info!(?path, "loading Hugging Face tokenizer");
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| xn::Error::wrap(e).with_path(path))?;
        Ok(Tok(tok))
    }

    /// Loads the contents of a `tokenizer.json`, for callers with no filesystem to read it from
    /// (the wasm demo fetches it over the network).
    pub fn from_bytes(json: &[u8]) -> xn::Result<Self> {
        let tok = tokenizers::Tokenizer::from_bytes(json).map_err(xn::Error::wrap)?;
        Ok(Tok(tok))
    }

}

impl crate::Tokenizer for Tok {
    /// Where each piece of `text` came from, as byte offsets into `text`.
    ///
    /// Word timestamps need to know which word a piece belongs to, and the tokenizer is the only
    /// thing that knows: grouping on `\u{2581}` gets contractions wrong, because "isn't" is one
    /// word but several pieces with punctuation in the middle.
    ///
    /// The offsets are used as `tokenizers` reports them. In the Rust crate they are byte offsets
    /// into the string that was passed in, tracked back through normalisation -- verified here
    /// against a `Prepend` normalizer and a multi-byte character, which is the case that would
    /// expose a mismatch. The Python package reports codepoints instead, so a port of its
    /// conversion step would corrupt every span after the first non-ASCII character; book text is
    /// full of `\u{2019}`, so that would not have stayed hidden for long.
    ///
    /// Still clamped to `text.len()`: a tokenizer whose normalizer changes length can report a
    /// span past the end, and a bad span should narrow a word rather than panic mid-utterance.
    fn piece_layout(&self, text: &str) -> xn::Result<crate::PieceLayout> {
        // `false` matches `encode` above, so the pieces are exactly what the flow LM is fed and
        // the attention window's length is the length of this layout.
        let encoded = self.0.encode(text, false).map_err(xn::Error::wrap)?;
        let spans = encoded
            .get_offsets()
            .iter()
            .map(|&(begin, end)| {
                let begin = begin.min(text.len());
                (begin, end.clamp(begin, text.len()))
            })
            .collect();
        Ok(crate::PieceLayout { pieces: encoded.get_tokens().to_vec(), spans })
    }

    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        let encoded = self.0.encode(text, false).map_err(xn::Error::wrap)?;
        Ok(encoded.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> xn::Result<String> {
        self.0.decode(ids, true).map_err(xn::Error::wrap)
    }
}

#[cfg(test)]
mod tests {
    const MINIMAL: &str = r#"{"version":"1.0","added_tokens":[],
      "model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["ab",-1.0],["c",-2.0]]}}"#;

    #[test]
    fn from_bytes_reads_a_tokenizer_json() {
        use crate::Tokenizer as _;
        let tok = super::Tok::from_bytes(MINIMAL.as_bytes()).unwrap();
        assert_eq!(tok.encode("abc").unwrap(), [1, 2]);
    }

    /// A checkpoint-shaped tokenizer: `Prepend` + `Metaspace`, as the 40M's own
    /// `tokenizer.json` declares. The normalizer rewrites the string, which is the case that
    /// would shift offsets if they were reported against the normalised form.
    const PREPEND: &str = r#"{"version":"1.0","added_tokens":[],
      "normalizer":{"type":"Prepend","prepend":"\u2581"},
      "pre_tokenizer":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"never"},
      "model":{"type":"Unigram","unk_id":0,
        "vocab":[["<unk>",0.0],["\u2581caf",-1.0],["\u00e9",-2.0],["\u2581x",-3.0]]}}"#;

    /// The spans index the original text in BYTES, so a multi-byte character must not shift
    /// the pieces after it. Converting codepoints to bytes here -- as the Python package's
    /// offsets require -- would turn `é`'s (3, 5) into (3, 6) and eat the following space.
    #[test]
    fn piece_layout_spans_are_byte_offsets_into_the_text() {
        use crate::Tokenizer as _;
        let tok = super::Tok::from_bytes(PREPEND.as_bytes()).unwrap();
        let text = "café x";
        let layout = tok.piece_layout(text).unwrap();
        let got: Vec<&str> = layout.spans.iter().map(|&(b, e)| &text[b..e]).collect();
        assert_eq!(got, ["caf", "é", " x"], "spans must slice the text they came from");
        assert_eq!(layout.pieces.len(), layout.spans.len());
        // The last span reaches the end of the text in bytes (7), not in chars (6).
        assert_eq!(layout.spans.last().unwrap().1, text.len());
    }

    /// A span is clamped rather than allowed to panic: a word with a bad span should come out
    /// narrow, not take the whole utterance down mid-sentence.
    #[test]
    fn piece_layout_spans_stay_inside_the_text() {
        use crate::Tokenizer as _;
        let tok = super::Tok::from_bytes(PREPEND.as_bytes()).unwrap();
        for text in ["", "x", "café x", "caf\u{e9}\u{e9}\u{e9} x"] {
            let layout = tok.piece_layout(text).unwrap();
            for &(b, e) in &layout.spans {
                assert!(b <= e && e <= text.len(), "span ({b}, {e}) outside {text:?}");
                assert!(text.is_char_boundary(b) && text.is_char_boundary(e), "span splits a char");
            }
        }
    }
}

