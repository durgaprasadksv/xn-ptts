//! Attention weights for word-level timestamps.
//!
//! Two or three heads per checkpoint track the text token being spoken
//! (`timestamp_heads` in the checkpoint config). Watching where they attend as
//! audio is generated gives word boundaries with no second model.
//!
//! Generation runs one frame at a time, so attention goes through the fused
//! `sdpa_decode`, which returns only its output. So the weights are recomputed
//! here for the configured heads alone — the Python package does the same, for
//! the same reason: its `scaled_dot_product_attention` hides them too.
//!
//! Off unless `configure` has been called, and then it costs one small matmul
//! per frame over two or three heads and the text window.

use std::sync::{Mutex, OnceLock};

use xn::{Backend, Tensor, TensorView, WithDTypeF};

/// What to capture.
///
/// The text occupies a **fixed** range of key positions: the voice prompt comes
/// first, then the text tokens, and each generated frame appends after them. So
/// the window is absolute and the caller computes it, rather than being derived
/// from the key length — the key length grows by one per generated frame, and a
/// window taken relative to it slides off the text immediately.
#[derive(Debug, Clone)]
pub struct Config {
    /// `(layer, query head)` pairs, as the checkpoint config lists them.
    pub heads: Vec<(usize, usize)>,
    /// First text position: the length of the voice prompt.
    pub text_start: usize,
    /// One past the last text position: `text_start + tokens`.
    pub text_end: usize,
}

#[derive(Default)]
struct State {
    config: Option<Config>,
    /// Current layer, set by the transformer loop. The heads of one frame
    /// arrive across several layers.
    layer: usize,
    /// Running sum over the heads captured for the frame in progress, and how
    /// many have arrived. A frame completes when every configured head has.
    sum: Vec<f32>,
    count: usize,
    /// One finished frame per entry: attention over the text tokens, averaged
    /// across the configured heads.
    frames: Vec<Vec<f32>>,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}

/// Start capturing. Clears anything held from a previous utterance.
pub fn configure(config: Config) {
    let mut s = state().lock().unwrap();
    s.config = Some(config);
    s.sum.clear();
    s.count = 0;
    s.frames.clear();
}

pub fn stop() {
    let mut s = state().lock().unwrap();
    *s = State::default();
}

pub fn enabled() -> bool {
    state().lock().unwrap().config.is_some()
}

/// Which layer is in `forward`. A whole-frame pass visits each in turn.
pub fn set_layer(layer: usize) {
    state().lock().unwrap().layer = layer;
}

/// Take the finished frames, leaving the capture configured.
pub fn take_frames() -> Vec<Vec<f32>> {
    std::mem::take(&mut state().lock().unwrap().frames)
}

/// Recompute attention for this layer's configured heads and fold it into the
/// frame in progress.
///
/// `q` is `(b, 1, h, d)` after RoPE; `k` is `(b, kv, h, d)` after the KV cache
/// and the GQA repeat, so a query head index means the same thing on both
/// sides. The softmax is over the text window only — the same normalisation
/// the Python package uses, and the reason its scores are not diluted by the
/// audio positions.
pub fn capture<T: WithDTypeF, B: Backend>(
    q: &Tensor<T, B>,
    k: &TensorView<T, B>,
    scale: f32,
) -> xn::Result<()> {
    let (heads, text_start, text_end) = {
        let s = state().lock().unwrap();
        let Some(cfg) = s.config.as_ref() else { return Ok(()) };
        let heads: Vec<usize> =
            cfg.heads.iter().filter(|(l, _)| *l == s.layer).map(|(_, h)| *h).collect();
        if heads.is_empty() {
            return Ok(());
        }
        (heads, cfg.text_start, cfg.text_end)
    };
    let text_len = text_end.saturating_sub(text_start);

    // Only the single-step decode path. The prefill pass over the text runs
    // through here too, with `t` the whole prompt rather than 1: its scores
    // have a different shape, and it is not an audio frame, so capturing it
    // both corrupts the per-head slicing and shifts every frame that follows.
    // Python captures per generated frame for the same reason.
    if q.dims()[1] != 1 {
        return Ok(());
    }

    let kv = k.dims()[1];
    // While the text is still being read in, the keys reach only as far as the
    // text itself. Those passes are not audio frames, so capturing them would
    // put frames in the sink that no audio corresponds to.
    if text_len == 0 || kv <= text_end {
        return Ok(());
    }
    if std::env::var_os("PTTS_DEBUG_WINDOW").is_some() {
        let s = state().lock().unwrap();
        if s.frames.len() < 4 && s.count == 0 {
            eprintln!(
                "[window] capture {} kv={kv} -> taking {text_start}..{text_end} (fixed)",
                s.frames.len()
            );
        }
    }
    // (b, h, 1, d) against the text slice of (b, h, kv, d).
    let qt = q.transpose(1, 2)?;
    let kt = k.transpose(1, 2)?.narrow(2, text_start..text_end)?.contiguous()?;
    let scores = qt.matmul_t(&kt)?.scale(T::from_f32(scale))?.softmax()?;
    let flat: Vec<f32> =
        scores.flatten_all()?.to_vec()?.iter().map(|v| v.to_f64().unwrap_or(0.0) as f32).collect();

    let mut s = state().lock().unwrap();
    let total = s.config.as_ref().map(|c| c.heads.len()).unwrap_or(0);
    if s.sum.len() != text_len {
        s.sum = vec![0.0; text_len];
    }
    for head in &heads {
        let row = &flat[head * text_len..(head + 1) * text_len];
        for (acc, v) in s.sum.iter_mut().zip(row) {
            *acc += v;
        }
        s.count += 1;
    }
    // Every configured head has reported, so the frame is complete. Averaging
    // rather than summing keeps the scores comparable across checkpoints with
    // different numbers of heads.
    if s.count >= total && total > 0 {
        let n = s.count as f32;
        let frame: Vec<f32> = s.sum.iter().map(|v| v / n).collect();
        s.frames.push(frame);
        s.sum.iter_mut().for_each(|v| *v = 0.0);
        s.count = 0;
    }
    Ok(())
}
