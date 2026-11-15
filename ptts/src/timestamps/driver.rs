//! Driving the aligner over a generation, one chunk at a time.
//!
//! The pieces this joins sit on either side of a thread boundary. The attention comes from
//! [`crate::attn_capture`], a process-global record the flow LM writes as it runs; the text
//! layout comes from the tokenizer, which only the calling thread holds. So the layouts are
//! built up front, where the tokenizer is, and the capture is configured and drained inside
//! the generation loop, where the chunk boundaries are.
//!
//! ## Time comes from the frame rate, not from the audio
//!
//! The reference divides a chunk's audio by its frame count, so the frame duration cannot
//! drift from the rate the model actually ran at, and gates each frame on whether its own
//! audio is voiced. Neither is available here: xn-ptts runs the flow LM ahead of a batching
//! decoder on another thread, so when a chunk's attention is complete its audio does not
//! exist yet, and what the consumer eventually receives has been coalesced and delayed by the
//! post-processing chain.
//!
//! Instead the frame duration is the codec's own `frame_rate`, which is exact by construction
//! -- a frame IS 1/frame_rate of audio -- and every frame is treated as voiced. Dropping the
//! silence gate means nothing suppresses attention during a pause, so a word that the model
//! leads into slowly can start marginally early. Pairing each chunk with its own PCM across
//! the pipeline would restore the gate, at the cost of threading audio back to this point.

use super::{is_voiced, TextUnit, TimedWord, WordAlignment, WordEvent};
use crate::Result;

/// One chunk's text, laid out for the aligner.
pub struct ChunkLayout {
    units: Vec<TextUnit>,
    token_to_unit: Vec<usize>,
    /// How many pieces the prompt holds, which is the width of the attention window.
    pieces: usize,
    words: usize,
}

impl ChunkLayout {
    /// Lay out one chunk's prepared prompt. `text` is the string the model is actually fed,
    /// after [`crate::tts_model::prepare_text_prompt`], because that is what was tokenized
    /// and so what the attention columns correspond to.
    pub fn new(text: &str, tokenizer: &dyn crate::Tokenizer) -> Result<Self> {
        let layout = tokenizer.piece_layout(text)?;
        let (units, token_to_unit) = super::units_from_layout(text, &layout.pieces, &layout.spans)?;
        let words = units.iter().filter(|u| u.is_word()).count();
        Ok(Self { units, token_to_unit, pieces: layout.pieces.len(), words })
    }
}

/// Collects word timings across a generation's chunks.
pub struct Aligner {
    heads: Vec<(usize, usize)>,
    /// Where the text sits in the sequence: after the voice prompt, before any generated
    /// frame, so its position is known before generation starts.
    ///
    /// No adjustment for `insert_bos_before_voice`: xn-ptts models that BOS as the speaker
    /// sink and the prompt length already counts it, so adding it again would put the window
    /// one position late and drop the first audio frame.
    text_start: usize,
    frame_s: f64,
    layouts: Vec<ChunkLayout>,
    words: Vec<TimedWord>,
    /// Carried across chunks so a caller sees one stream: words numbered from the start of
    /// the input, and times measured from its first sample.
    word_offset: usize,
    frames_before: usize,
}

impl Aligner {
    pub fn new(
        heads: Vec<(usize, usize)>,
        text_start: usize,
        frame_rate: f64,
        layouts: Vec<ChunkLayout>,
    ) -> Self {
        Self {
            heads,
            text_start,
            frame_s: 1.0 / frame_rate,
            layouts,
            words: Vec::new(),
            word_offset: 0,
            frames_before: 0,
        }
    }

    /// Point the capture at this chunk's text window. Called on the generating thread, before
    /// the chunk's first step.
    pub fn begin_chunk(&self, index: usize) {
        let Some(layout) = self.layouts.get(index) else {
            crate::attn_capture::stop();
            return;
        };
        crate::attn_capture::configure(crate::attn_capture::Config {
            heads: self.heads.clone(),
            text_start: self.text_start,
            text_end: self.text_start + layout.pieces,
        });
    }

    /// Drain the chunk's captured attention and align it. Called on the generating thread,
    /// once the chunk's last step is done.
    pub fn end_chunk(&mut self, index: usize) {
        let captured = crate::attn_capture::take_frames();
        let Some(layout) = self.layouts.get(index) else { return };
        if captured.is_empty() {
            return;
        }
        let time_offset = self.frames_before as f64 * self.frame_s;
        let unit_count = layout.units.len();
        // A word's span is the only reliable link from a timing back to the text, and the
        // units are about to move into the alignment.
        let spans: std::collections::HashMap<usize, (usize, usize)> = layout
            .units
            .iter()
            .filter_map(|u| u.word.as_ref().map(|w| (w.word_index, (w.begin, w.end))))
            .collect();
        let mut alignment = WordAlignment::new(layout.units.clone());
        let mut ends = Vec::new();
        for (frame, scores) in captured.iter().enumerate() {
            let scores = super::scores_by_unit(scores, &layout.token_to_unit, unit_count);
            // Every frame counts as voiced: see this module's header.
            for event in alignment.process_frame(&scores, true, frame as f64 * self.frame_s) {
                if let WordEvent::End(end) = event {
                    ends.push(end);
                }
            }
        }
        ends.extend(alignment.finish(captured.len() as f64 * self.frame_s));
        self.words.extend(ends.into_iter().map(|mut word| {
            let &(begin, end) = spans.get(&word.word_index).unwrap_or(&(0, 0));
            word.word_index += self.word_offset;
            word.start += time_offset;
            word.end += time_offset;
            TimedWord { word, begin, end }
        }));
        self.word_offset += layout.words;
        self.frames_before += captured.len();
    }

    /// The words aligned so far, in order.
    pub fn words(&self) -> &[TimedWord] {
        &self.words
    }

    pub fn take_words(&mut self) -> Vec<TimedWord> {
        std::mem::take(&mut self.words)
    }
}

/// Kept so the unused-import warning does not fire while the silence gate is unused, and so
/// the gate is one call away if the audio is ever threaded back here.
#[allow(dead_code)]
fn voiced_gate(pcm: &[f32]) -> bool {
    is_voiced(pcm)
}
