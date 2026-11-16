//! The engine everywhere but Apple: [`ptts::synth::Synth`] on the CPU, with `dir` a checkpoint
//! folder as the other frontends read it.
//!
//! The folder holds `tokenizer.json`, the weights (q8_0 GGUF or f32 safetensors, named as in
//! [`WEIGHTS`]), its own `config.json` and its voices
//! in `voices/` or `embeddings/`, or as `default-voice.safetensors`.

use crate::{PTTS_UNIT_CPU, PttsResult};
use ptts::checkpoint::read_config;
use ptts::preprocess::Normalize;
use ptts::synth::{DeviceKind, Quant, SpeechOptions, Synth, SynthBuilder};
use std::ffi::CString;
use std::path::Path;

/// Weight files tried in order, with the format each loads as.
const WEIGHTS: [(&str, Quant); 2] =
    [("model.q8.gguf", Quant::Q80), ("model.safetensors", Quant::F32)];

pub struct Engine {
    synth: Synth,
    voices: Vec<String>,
    languages: Vec<String>,
    voice: Option<String>,
    /// Left `None`, the builder's defaults.
    temperature: Option<f32>,
    seed: Option<u64>,
    /// Applied here rather than by `synth`, which is built with `Normalize::OFF`, so that
    /// `set_language` can change it without a reload. `Synth` applies it first thing anyway.
    normalize: Normalize,
    /// Samples per frame, to count frames in what the stream hands back.
    frame_size: usize,
    /// Whether to report word timings for the next utterance.
    timings: bool,
    /// Whether this checkpoint *can*: word timings read specific attention heads, named by
    /// `timestamp_heads` in its config, and a checkpoint nobody has calibrated has none.
    can_time: bool,
    /// The last utterance's word timings as JSON, for `ptts_word_timings` to borrow.
    last_timings: Option<CString>,
}

impl Engine {
    pub fn open(dir: &Path, unit: u32, lang: &str) -> Result<Self, String> {
        if unit != PTTS_UNIT_CPU {
            return Err(format!("compute unit {unit} is Apple only; use PTTS_UNIT_CPU"));
        }
        let normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        let config = dir.join("config.json");
        let cfg = read_config(&config).map_err(|e| e.to_string())?;
        let (weights, quant) =
            WEIGHTS.iter().map(|&(name, q)| (dir.join(name), q)).find(|(p, _)| p.is_file()).ok_or(
                format!(
                    "no weights in {}; expected one of {}",
                    dir.display(),
                    WEIGHTS.map(|(name, _)| name).join(", ")
                ),
            )?;
        let frame_rate = cfg.mimi.frame_rate;
        let can_time = cfg.timestamp_heads.as_ref().is_some_and(|heads| !heads.is_empty());
        let mut builder = SynthBuilder::new(cfg, weights, Normalize::OFF)
            .tokenizer_file(dir.join("tokenizer.json"))
            .device(DeviceKind::Cpu)
            .quant(quant);
        for (name, path) in ptts::loader::checkpoint_voices(dir) {
            builder = builder.add_voice(name, path);
        }
        let mut synth = builder.build().map_err(|e| e.to_string())?;
        // `Synth` normalizes internally from here on, instead of this engine doing it and
        // handing over the result. That is what lets a word timing's offsets address the
        // caller's own string: see `speak`.
        synth.set_normalization(normalize);
        let frame_size = (synth.sample_rate() as f64 / frame_rate).round() as usize;
        Ok(Self {
            voices: synth.voices(),
            languages: synth.languages().to_vec(),
            voice: synth.default_voice(),
            temperature: None,
            seed: None,
            normalize,
            synth,
            frame_size,
            timings: false,
            can_time,
            last_timings: None,
        })
    }

    pub fn voices(&self) -> &[String] {
        &self.voices
    }

    pub fn languages(&self) -> &[String] {
        &self.languages
    }

    /// Switch to one of [`Self::voices`].
    pub fn set_voice(&mut self, name: &str) -> Result<(), String> {
        self.voice = Some(name.to_string());
        Ok(())
    }

    /// Register one voice file under `name`, so [`Self::set_voice`] will take it.
    ///
    /// Everything in the checkpoint's `voices/` is loaded by `open`, which is the right default
    /// for a handful of voices and the wrong one for a catalogue: a checkpoint serving 189 of
    /// them would hold every embedding resident to speak in one. This loads the one that is
    /// wanted, when it is wanted, at roughly 256KB each.
    pub fn add_voice(&mut self, name: &str, path: &Path) -> Result<(), String> {
        if self.voices.iter().any(|v| v == name) {
            return Ok(()); // Idempotent: re-selecting a voice must not reload it.
        }
        self.synth.add_voice_file(name, path).map_err(|e| e.to_string())?;
        self.voices.push(name.to_string());
        Ok(())
    }

    /// Normalize text as `lang` from the next utterance on.
    pub fn set_language(&mut self, lang: &str) -> Result<(), String> {
        self.normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        // Normalization is read once per request, so the spoken forms change with no reload.
        self.synth.set_normalization(self.normalize);
        Ok(())
    }

    /// Report word timings from the next utterance on.
    ///
    /// Asking for them on a checkpoint with no calibrated heads is *not* an error: it speaks
    /// without them. Refusing here would be worse than useless -- a caller that asks for
    /// timings by default, as the app does, would get an exception instead of audio -- and
    /// failing later inside `speak` would lose the utterance for the same reason.
    pub fn set_timings(&mut self, on: bool) -> Result<(), String> {
        if on && !self.can_time {
            tracing::warn!(
                "this checkpoint has no calibrated timestamp_heads, so it will speak without \
                 word timings"
            );
        }
        self.timings = on;
        Ok(())
    }

    /// The last utterance's word timings as JSON, or `None` if none were asked for.
    pub fn word_timings(&self) -> Option<&CString> {
        self.last_timings.as_ref()
    }

    /// Sample at `temperature`, already checked, from the next utterance on.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = Some(temperature);
    }

    /// Seed the sampling from the next utterance on.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
    }

    pub fn speak(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<PttsResult, String> {
        let start = std::time::Instant::now();
        let opts = SpeechOptions {
            voice: self.voice.clone(),
            temperature: self.temperature,
            seed: self.seed,
            timings: self.timings && self.can_time,
            ..Default::default()
        };
        // The caller's text, not a normalized copy: `Synth` normalizes it itself, and only
        // the string passed here can be what a timing's offsets are measured against. Pass
        // the normalized form instead and `$2` reports the span of "2 dollars" -- offsets
        // that look valid, monotonic and non-empty while pointing at the wrong words.
        self.last_timings = None;
        let mut stream = self.synth.stream_with(text, &opts).map_err(|e| e.to_string())?;
        let (mut samples, mut ttfa) = (0, None);
        // Dropping the stream on an early return stops generation. Borrowed rather than
        // consumed, so the timings can be read off it once the audio is done.
        for pcm in stream.by_ref() {
            let pcm = pcm.map_err(|e| e.to_string())?;
            if pcm.is_empty() {
                continue;
            }
            ttfa.get_or_insert(start.elapsed());
            samples += pcm.len();
            if !sink(&pcm) {
                break;
            }
        }
        if self.timings && self.can_time {
            let words = stream.word_timings_in_text();
            let json = serde_json::to_string(&words).map_err(|e| e.to_string())?;
            self.last_timings = Some(CString::new(json).map_err(|e| e.to_string())?);
        }
        Ok(PttsResult {
            frames: samples.div_ceil(self.frame_size) as u32,
            samples,
            ttfa_ms: ttfa.unwrap_or_default().as_secs_f64() * 1e3,
            total_ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PTTS_UNIT_ANE;

    fn open_error(unit: u32, lang: &str) -> String {
        match Engine::open(Path::new("no-such-checkpoint"), unit, lang) {
            Ok(_) => panic!("opened a checkpoint that does not exist"),
            Err(e) => e,
        }
    }

    // Each is refused before anything is loaded, so none needs weights.
    #[test]
    fn open_refuses_before_loading() {
        assert!(open_error(PTTS_UNIT_ANE, "en").contains("Apple only"));
        assert!(!open_error(PTTS_UNIT_CPU, "klingon").contains("no weights"));
        assert!(open_error(PTTS_UNIT_CPU, "en").contains("config.json"));
    }
}
