//! Turning per-frame attention scores into word boundaries.
//!
//! A port of `WordAlignment` in `pocket_tts_timestamped/timestamps/alignment.py`.
//! One word is open at a time; it closes when a later word starts winning the
//! attention, or when the audio goes quiet and the attention has already moved
//! on. Nothing here touches the model, so it is tested against hand-written
//! score vectors.

use super::records::{SourceWord, TextUnit, WordEnd, WordEvent, WordStart};

/// Below this, the attention on the open word is weak enough that *any* later
/// word outscoring it counts as having moved on — not just the next one. Above
/// it, only the next word may take over, which stops a distant word's noise
/// from closing the current one early.
const NON_NEXT_ATTENTION_THRESHOLD: f32 = 0.001;

/// Audio quieter than this is treated as silence. Energy, not amplitude: the
/// caller compares `sum(x^2)` against `threshold^2 * n`.
const SILENCE_RMS_THRESHOLD: f32 = 1e-3;

/// Is this frame's audio voiced? Silence is what lets a word close at the end
/// of an utterance, where no later word will ever outscore it.
pub fn is_voiced(samples: &[f32]) -> bool {
    is_voiced_at(samples, SILENCE_RMS_THRESHOLD)
}

/// `is_voiced` against a given threshold. Separate so the gate can be tested
/// at a readable level, as the Python tests do by monkeypatching it.
pub fn is_voiced_at(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return false;
    }
    let energy: f32 = samples.iter().map(|s| s * s).sum();
    energy > threshold * threshold * samples.len() as f32
}

pub struct WordAlignment {
    units: Vec<TextUnit>,
    /// Injectable so the gate can be exercised at a readable scale; the Python
    /// tests monkeypatch the same constant.
    non_next_gate: f32,
    /// Indices into `units` of the units that are words, in order.
    word_unit_indices: Vec<usize>,
    next_word_position: usize,
    open_word_position: Option<usize>,
    open_start: Option<f64>,
}

impl WordAlignment {
    pub fn new(units: Vec<TextUnit>) -> Self {
        let word_unit_indices =
            units.iter().enumerate().filter(|(_, u)| u.is_word()).map(|(i, _)| i).collect();
        Self {
            units,
            non_next_gate: NON_NEXT_ATTENTION_THRESHOLD,
            word_unit_indices,
            next_word_position: 0,
            open_word_position: None,
            open_start: None,
        }
    }

    /// For tests: run the gate at a different threshold.
    pub fn with_non_next_gate(mut self, gate: f32) -> Self {
        self.non_next_gate = gate;
        self
    }

    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    fn word(&self, position: usize) -> &SourceWord {
        self.units[self.word_unit_indices[position]]
            .word
            .as_ref()
            .expect("a unit in word_unit_indices always carries a word")
    }

    fn open_next(&mut self, timestamp: f64) -> WordStart {
        let position = self.next_word_position;
        let word = self.word(position).clone();
        self.open_word_position = Some(position);
        self.open_start = Some(timestamp);
        self.next_word_position = position + 1;
        WordStart { text: word.text, word_index: word.word_index, start: timestamp }
    }

    fn close(&mut self, timestamp: f64) -> WordEnd {
        let position = self.open_word_position.expect("close with no open word");
        let start = self.open_start.expect("close with no open start");
        let word = self.word(position).clone();
        self.open_word_position = None;
        self.open_start = None;
        WordEnd { text: word.text, word_index: word.word_index, start, end: timestamp }
    }

    /// Has the attention moved past the open word? The next word taking the
    /// lead is enough on its own; a word further ahead only counts when the
    /// open word's own score has already collapsed.
    fn future_word_dominates(&self, scores: &[f32], current_unit: usize) -> bool {
        let current = scores[current_unit];
        let next_unit = self.word_unit_indices[self.next_word_position];
        if scores[next_unit] > current {
            return true;
        }
        if current >= self.non_next_gate {
            return false;
        }
        self.word_unit_indices[self.next_word_position + 1..]
            .iter()
            .any(|&unit| scores[unit] > current)
    }

    /// One frame of audio. `scores` is one score per text unit, `frame_start`
    /// the frame's offset in seconds from the start of this utterance's audio.
    pub fn process_frame(
        &mut self,
        scores: &[f32],
        voiced: bool,
        frame_start: f64,
    ) -> Vec<WordEvent> {
        assert_eq!(
            scores.len(),
            self.units.len(),
            "expected one score per text unit"
        );
        let mut events = Vec::new();

        if !voiced {
            let Some(open) = self.open_word_position else { return events };
            let current_unit = self.word_unit_indices[open];
            let future = &scores[current_unit + 1..];
            let should_close =
                future.iter().copied().fold(f32::NEG_INFINITY, f32::max) > scores[current_unit];
            let is_final_word = open == self.word_unit_indices.len() - 1;
            // Punctuation the author wrote means this silence is a pause, not
            // the end of the utterance; one we appended means nothing.
            let has_later_punctuation = self.units[current_unit + 1..]
                .iter()
                .any(|u| !u.is_word() && !u.synthetic);
            if should_close || (is_final_word && !has_later_punctuation) {
                events.push(WordEvent::End(self.close(frame_start)));
            }
            return events;
        }

        if self.open_word_position.is_none() {
            if self.next_word_position >= self.word_unit_indices.len() {
                return events;
            }
            events.push(WordEvent::Start(self.open_next(frame_start)));
            return events;
        }

        let current_unit = self.word_unit_indices[self.open_word_position.unwrap()];
        if self.next_word_position >= self.word_unit_indices.len() {
            return events;
        }
        if self.future_word_dominates(scores, current_unit) {
            events.push(WordEvent::End(self.close(frame_start)));
            events.push(WordEvent::Start(self.open_next(frame_start)));
        }
        events
    }

    /// The audio ended. Closes whatever is still open against its end, so the
    /// last word of an utterance always gets a boundary.
    pub fn finish(&mut self, audio_end: f64) -> Vec<WordEnd> {
        let mut events = Vec::new();
        if self.open_word_position.is_some() {
            events.push(self.close(audio_end));
        }
        self.next_word_position = self.word_unit_indices.len();
        events
    }
}

#[cfg(test)]
mod tests {
    //! Ported from `tests/test_timestamps.py`, case for case and number for
    //! number, so a divergence between the two implementations shows up as a
    //! failing test rather than as wrong word boundaries in the field. Names
    //! are the Python names.
    use super::*;

    /// `_units` from the Python tests: "P:x" is punctuation, anything else a
    /// word, with char offsets accumulated across the arguments.
    fn units(values: &[&str]) -> Vec<TextUnit> {
        let mut out = Vec::new();
        let mut word_index = 0;
        let mut position = 0;
        for value in values {
            if let Some(text) = value.strip_prefix("P:") {
                out.push(TextUnit::punctuation(text, position, position + text.len()));
            } else {
                out.push(TextUnit::word(*value, word_index, position, position + value.len()));
                word_index += 1;
            }
            position += value.len();
        }
        out
    }

    /// The gate the Python fixture monkeypatches in.
    const GATE: f32 = 0.1;

    fn start(text: &str, i: usize, at: f64) -> WordEvent {
        WordEvent::Start(WordStart { text: text.into(), word_index: i, start: at })
    }

    fn end(text: &str, i: usize, from: f64, to: f64) -> WordEvent {
        WordEvent::End(WordEnd { text: text.into(), word_index: i, start: from, end: to })
    }

    #[test]
    fn alignment_transitions_to_next_word_and_hard_finishes() {
        let mut a = WordAlignment::new(units(&["one", "two"]));
        let first = a.process_frame(&[0.8, 0.2], true, 0.0);
        let transition = a.process_frame(&[0.2, 0.8], true, 0.08);
        let final_ = a.finish(0.16);

        assert_eq!(first, vec![start("one", 0, 0.0)]);
        assert_eq!(transition, vec![end("one", 0, 0.0, 0.08), start("two", 1, 0.08)]);
        assert_eq!(final_, vec![WordEnd {
            text: "two".into(),
            word_index: 1,
            start: 0.08,
            end: 0.16
        }]);
    }

    #[test]
    fn later_word_evidence_below_gate_advances_one_word() {
        let mut a = WordAlignment::new(units(&["one", "two", "three"])).with_non_next_gate(GATE);
        a.process_frame(&[0.9, 0.05, 0.05], true, 0.0);
        let events = a.process_frame(&[GATE / 2.0, 0.01, 0.9], true, 0.08);
        // "three" is the one with the evidence, but the word that opens is
        // "two": the alignment advances one word at a time, never skips.
        assert_eq!(events, vec![end("one", 0, 0.0, 0.08), start("two", 1, 0.08)]);
    }

    #[test]
    fn non_next_evidence_above_gate_is_blocked() {
        let mut a = WordAlignment::new(units(&["one", "two", "three"])).with_non_next_gate(GATE);
        a.process_frame(&[0.9, 0.05, 0.05], true, 0.0);
        assert_eq!(a.process_frame(&[GATE * 2.0, 0.01, 0.9], true, 0.08), vec![]);
    }

    #[test]
    fn non_next_evidence_uses_no_distance_margin() {
        let mut a =
            WordAlignment::new(units(&["one", "two", "three", "four"])).with_non_next_gate(GATE);
        a.process_frame(&[0.9, 0.05, 0.03, 0.02], true, 0.0);
        let current = GATE / 2.0;
        // The last word beats the open one by one percent, and that is enough.
        // No extra margin is demanded of a more distant word.
        assert!(!a.process_frame(&[current, 0.01, 0.01, current * 1.01], true, 0.08).is_empty());
    }

    #[test]
    fn gate_never_blocks_next_word_evidence() {
        let mut a = WordAlignment::new(units(&["one", "two", "three"]));
        a.process_frame(&[0.9, 0.05, 0.05], true, 0.0);
        // "one" is far above the gate, but its immediate successor outscores
        // it, and the gate never stands in the way of that.
        assert!(!a.process_frame(&[0.5, 0.51, 0.9], true, 0.08).is_empty());
    }

    #[test]
    fn silence_closes_for_future_punctuation_but_never_opens() {
        let mut a = WordAlignment::new(units(&["one", "P:."]));
        a.process_frame(&[0.8, 0.2], true, 0.0);
        let events = a.process_frame(&[0.2, 0.8], false, 0.08);
        assert_eq!(events, vec![end("one", 0, 0.0, 0.08)]);
        // Silence can end a word; it can never begin one.
        assert_eq!(a.process_frame(&[0.1, 0.9], false, 0.16), vec![]);
    }

    #[test]
    fn unpunctuated_final_word_closes_on_first_silence() {
        let mut a = WordAlignment::new(units(&["one"]));
        a.process_frame(&[1.0], true, 0.0);
        assert_eq!(a.process_frame(&[1.0], false, 0.08), vec![end("one", 0, 0.0, 0.08)]);
    }

    #[test]
    fn rms_silence_gate_uses_configured_threshold() {
        // -60 dBFS as the gate, then a hair either side of it.
        let threshold = 10f32.powf(-60.0 / 20.0);
        for (dbfs, expected) in [(-59.0f32, true), (-61.0, false)] {
            let amplitude = 10f32.powf(dbfs / 20.0);
            assert_eq!(
                is_voiced_at(&vec![amplitude; 100], threshold),
                expected,
                "{dbfs} dBFS against a -60 dBFS gate"
            );
        }
    }

    #[test]
    fn empty_audio_is_silent() {
        assert!(!is_voiced(&[]));
    }

    // `silence_detection_supports_non_contiguous_float_inputs` does not port:
    // a Rust slice is contiguous by construction, and there is no dtype to
    // vary. The Python test guards against a torch view being read wrongly.

    #[test]
    fn appended_punctuation_lets_the_final_word_close_on_silence() {
        // The other half of `silence_closes_for_future_punctuation`: when the
        // punctuation ahead was appended by the pipeline rather than written,
        // the silence is the end of the utterance and the word closes.
        let mut a = WordAlignment::new(vec![
            TextUnit::word("one", 0, 0, 3),
            TextUnit::punctuation(".", 3, 4).synthetic(),
        ]);
        a.process_frame(&[0.8, 0.2], true, 0.0);
        assert_eq!(a.process_frame(&[0.9, 0.1], false, 0.08), vec![end("one", 0, 0.0, 0.08)]);
    }

    #[test]
    fn every_word_gets_one_contiguous_span() {
        let mut a = WordAlignment::new(units(&["one", "two", "three", "P:."]));
        let mut all = Vec::new();
        let on = |unit: usize| {
            let mut s = vec![0.0f32; 4];
            s[unit] = 1.0;
            s
        };
        for (frame, unit) in [0usize, 0, 1, 1, 1, 2, 2].iter().enumerate() {
            all.extend(a.process_frame(&on(*unit), true, frame as f64 * 0.08));
        }
        all.extend(a.finish(0.56).into_iter().map(WordEvent::End));

        let spans: Vec<(String, f64, f64)> = all
            .iter()
            .filter_map(|e| match e {
                WordEvent::End(w) => Some((w.text.clone(), w.start, w.end)),
                _ => None,
            })
            .collect();
        assert_eq!(spans.len(), 3);
        for (a, b) in spans.iter().zip(spans.iter().skip(1)) {
            assert_eq!(a.2, b.1, "a word starts where the previous one ended");
        }
        assert_eq!(spans.last().unwrap().2, 0.56);
    }

    #[test]
    fn finish_on_a_closed_alignment_emits_nothing() {
        let mut a = WordAlignment::new(units(&["one"]));
        assert!(a.finish(1.0).is_empty());
    }

    #[test]
    fn silence_before_any_word_emits_nothing() {
        let mut a = WordAlignment::new(units(&["one", "two"]));
        assert!(a.process_frame(&[1.0, 0.0], false, 0.0).is_empty());
    }
}
