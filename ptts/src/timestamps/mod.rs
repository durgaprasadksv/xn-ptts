//! Word-level timestamps from the model's own attention.
//!
//! Two or three attention heads per checkpoint track the text token being
//! spoken (`timestamp_heads` in the checkpoint config, found by searching every
//! layer and head against a reference aligner). Watching where they attend as
//! audio is generated gives word boundaries with no second model and no
//! forced-alignment pass.
//!
//! A port of the Python package's `timestamps/` module. The split is the same:
//! `records` is what comes out, `alignment` is the per-frame state machine, and
//! neither touches the model — the scores arrive from the generation loop.

pub mod alignment;
pub mod records;
pub mod units;

mod driver;

pub use alignment::{is_voiced, WordAlignment};
pub use driver::{Aligner, ChunkLayout};
pub use records::{SourceWord, TextUnit, TimedWord, WordEnd, WordEvent, WordStart};
pub use units::{lexical_word_spans, scores_by_unit, units_from_layout};
