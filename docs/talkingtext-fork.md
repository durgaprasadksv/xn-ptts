# What TalkingText carries on top of xn-ptts

Written 2026-11-15, updated 2026-11-16 when the migration landed. The point of
this document is to make a rebuild of the fork possible without re-deriving any
of it: every modification we carry, why it exists, and what the app would lose
without it.

**Read this before rebuilding `libptts_ffi.so`.** Two things recorded here cost
hours again on 2026-11-16 by not being read: the build needs `ARM64_FEATURES=`
(otherwise SIGILL on the test phone), and `ptts_set_timings`/`ptts_word_timings`
exist only on our branches, so a build from a branch without them bricks the app
at startup.

Everything below was read off the repository, not from memory. To regenerate the
facts:

```bash
git log  --oneline --no-merges dp/main..dp/android-timings   # the timings work
git diff --stat              dp/main...dp/android-timings
git show --stat 801dcbf                                      # attn_capture
git diff --stat dp/android-timings...android-timings-40m     # the 40M fixes
git log  --oneline --no-merges dp/main..origin/pocket-tts-small-models
git log  --oneline --no-merges 5f9522b..dp/pocket-conformance # the current line
```

## The branch landscape

Three repositories, and the naming in `git remote` currently hides it:

```
gradium-ai/xn-ptts              true upstream. NOT a remote here.
gabrieldemarmiesse/xn-ptts      named "origin". The fork that adds pocket-tts
                                model support; Kyutai's own pocket-tts binary
                                depends on it at 7efa269.
durgaprasadksv/xn-ptts          named "dp". Ours.
```

`dp/main` is **0 commits behind and 177 ahead of** `origin/main`, so our main is
current and gabriel's is the one lagging. A bare `git push` from a local `main`
aims at *gabriel's* repo, because local `main` tracks `origin/main`. Renaming the
remotes (`origin` → ours, add `gradium`, keep `gabriel`) removes that footgun.

Four lines of work exist, and **two of them are divergent, not sequential**:

```
dp/main  (cb43656)
  ├─ 5 commits ───────────────── origin/pocket-tts-small-models   (NOT ours)
  │    └─ 7efa269 ── 801dcbf ─── attn-capture                     (ours, 1 commit)
  └─ ...
31e4e9f (2026-10-31, gradium PR #314)
  └─ 6 commits ── 5eb40df ────── dp/android-timings               (ours, 7 commits)
       └─ 3 commits ──────────── android-timings-40m              (ours, 3 commits)
5f9522b  (upstream main, 2026-11-15)
  └─ 12 commits ────────────── dp/pocket-conformance           (ours, SHIPPING)
```

`dp/pocket-conformance` is the convergence the rest of this document anticipated:
all four of our modifications replayed onto current upstream main, plus the
reference timestamps module and the head calibration that gated it.

`7efa269` is **not** a descendant of `31e4e9f`. The timings branch and the
attention-capture branch sit on unrelated bases, which is the single biggest
cost in any migration: converging them is most of the work, not the ports.

`dp/pocket-conformance` is what the app ships as of 2026-11-16; it replaces
`dp/android-timings`, which is kept only as the fallback described in *Migration
notes*. Neither is merged into any main and neither ever will be: Gradium
declined word timings upstream (xn-ptts issue #321 — the model was never trained
to produce them). The carry is permanent.

## A. `dp/android-timings` — word timings (7 commits, 19 files, +1620/-84)

Based on `31e4e9f`, five weeks behind main. Six commits are authored by
`Cursor Agent`, carried from the abandoned PR `christianalfoni/xn-ptts#1`; the
last is ours.

| commit | what it adds |
|---|---|
| `6fae376` | `feat(align): read one frame's text attention when asked` |
| `6df7a1f` | `feat(align): map frames onto the words of the caller's text` |
| `c017100` | `feat(synth): report word timings on the speech stream` |
| `d95a808` | `feat(cli): write word timings as JSON` (`--word-timings`) |
| `bdefee8` | `feat(wasm): stream word timings with the audio` |
| `5d115ee` | `fix(align): cross a light token when a later one is the peak` |
| `5eb40df` | `feat(ffi): word timings through the C interface` (ours) |

Principal files: `ptts/src/align.rs` (new, 749 lines), `ptts/src/transformer.rs`
(+174), `ptts/src/synth.rs` (+218), `ptts/src/flow_lm.rs` (+111),
`ptts/src/tts_model.rs` (+59), plus the FFI (`ptts_set_timings`,
`ptts_word_timings`) and the wasm surface.

**How it decides where a word is**: it reads the **middle flow-LM layer's**
attention over the text tokens per frame and walks it monotonically. That layer
choice is a guess — nothing calibrated it. See *Known gaps* below, because the
reference implementation does this differently and better.

**Two properties the app depends on.** `align.rs` maps offsets back to the
caller's text through `prepared_to_chunk`, which walks forward and **breaks at
the first mismatch**, mapping added bytes to `None`. Any text transformation
that substitutes mid-string therefore destroys the timings from that point on —
which is why the apostrophe normalisation below cannot live in Rust. And the
spans it emits are **contiguous**: each word's `to` equals the next word's
`from`, and the last word's `to` equals the audio duration exactly (measured on
three clips: 0.000 s slack). Anything reasoning about gaps between words, or
about tail slack after the last word, will get zero and be wrong.

## B. `attn-capture` — the attention primitive (1 commit, 4 files, +197)

`801dcbf attn_capture: read the attention weights word timestamps need`, on top
of `7efa269`, i.e. **on gabriel's pocket-support line**. Adds
`ptts/src/attn_capture.rs` (164 lines) plus hooks in `flow_lm.rs`,
`transformer.rs`, `lib.rs`.

This is the xn-ptts half of the timestamp design in
**`dpm63/pocket-tts-timestamped`**, which is a *fork of* kyutai-labs/pocket-tts
that adds word timestamps — **not** Kyutai's own work, as an earlier draft of
this document said. The distinction matters: the design has no upstream home on
either side (Kyutai never shipped it, Gradium declined it in issue #321), so
nothing here is ever going away by being accepted elsewhere.

That fork calls `ptts::attn_capture::configure` / `take_frames` / `stop`
(`rust/src/model.rs:473-507`) and feeds the result to its own
`rust/src/timestamps/` module. This branch is the start of that approach, on the
right base, and is unrelated to A. It is now carried on
`dp/pocket-conformance`.

## C. `android-timings-40m` — running our 40M distill (3 commits, 7 files, +139)

Based on `5eb40df`. All three were needed to get `english-40m` working on the
phone.

**`e0de11c fix(loader): map pocket-tts's bos_before_voice onto the speaker sink`**
— `ptts/src/loader.rs`. pocket-tts names the learnt pre-voice token
`bos_before_voice`; audiocraft calls the same thing the `speaker_wavs` prefix
sink. Without the mapping the checkpoint fails to load with one unused tensor.
Silencing the unused-tensor check instead puts the conditioning one position out
of place and renders **garbled audio rather than failing** — that mistake was
made and heard.

**Resolved 2026-11-16: it is NOT redundant.** This document previously guessed
that gabriel's pocket-support commits might already handle it. They do not — the
40M fails to load on current main without this mapping, so it is carried on
`dp/pocket-conformance` as `fa209f7`. Do not drop it on a future rebase.

**`fccd4a6 feat(ffi): register one voice at a time with ptts_add_voice`** —
`ptts-ffi/{lib.rs,cpu.rs,coreml.rs,include/ptts.h}`. `ptts_new` loads everything
in `voices/`, which is right for a handful and wrong for a catalogue: 189 voices
is ~48 MB resident to speak in one. This loads the chosen voice (~256 KB) on
demand, is idempotent, and refreshes the blob `ptts_voices` returns. The Core ML
engine already resolves by name from its bundle, so there it only validates the
name — the C surface stays the same shape on both engines. Generally useful;
a plausible upstream contribution.

**`252e2c2 fix(text): close a prompt that ends mid-clause`** —
`ptts/src/tts_model.rs`, `ptts/src/plan.rs`. The old rule appended a stop only
after a letter or digit, so a chunk ending `…a mile and a half,` or `…she
wanted:` or `…to the garden,"` kept its weak mark. These models are trained on
sentences ending in terminal punctuation and *"without one the last word is
often mispronounced or repeated"* (Kyutai's own comment). Now a terminal mark is
left alone, weak punctuation (`, ; : - – —`) is **replaced** by a stop, anything
else gains one, and closing quotes/brackets are read past so the stop lands
inside them. Follows `pocket_tts/models/text_chunking.py`
(`append_terminal_punctuation`) while keeping the script-aware stop `full_stop`
picks — the reference only emits ASCII. Two `plan` tests changed, and that is
the fix landing where it matters: a long text cut at a comma now hands the model
a closed phrase.

This is a **conformance fix for every pocket checkpoint**, not something
specific to us, so gabriel's fork is its natural home. It is also the only one
of the three that touches code upstream owns, so it is the one that will
conflict on a rebase.

It applies to **both** checkpoints, phonon included, because
`prepare_text_prompt` is shared. Gating it to pocket models would mean threading
config through `plan.rs` and `align.rs` for a string transformation; the old rule
carried no Phonon-specific justification, unlike the 8-spaces case, so it was
left ungated. Revisit if Ivy regresses.

## D. `dp/pocket-conformance` — the current line (12 commits, tip `bee1689`)

Based on `5f9522b` (upstream main, 52 commits newer than A's base). Pushed to
`dp` only. 147 tests pass with `--features hf`; clippy clean for
`aarch64-linux-android`.

**The four carries replayed.** `fa209f7` (bos_before_voice, still needed — see
C), `2e476b7`+`c235bca` (terminal punctuation, with the rationale for why the
rule is not config-driven), `05cab1c` (`ptts_add_voice`). Audio parity against
the old base was measured at max 1 LSB before any new work landed, so the
migration itself changed nothing audible.

**The reference timestamps module, ported.** `e0bd9ad`+`d8c12ec` bring
`attn_capture` and `timestamps/{mod,records,alignment,units}.rs` from
`dpm63/pocket-tts-timestamped`. Divergence from the source was measured
deliberately: `alignment.rs` is **0 changed lines** of 362, `records.rs` 6
(serde derives), `units.rs` 12 (error type). Keeping it that close is what makes
re-pulling from that fork cheap, and it is why the mapping below lives outside
`timestamps/`.

**`c282108` tokenizer piece offsets.** `Tokenizer::piece_layout`, added as a
*defaulted* trait method so the four existing implementors are untouched.
Rust `tokenizers` reports **byte** offsets; the Python package reports
codepoints. Porting the reference's codepoint arithmetic would corrupt every
span after a non-ASCII character, so the offsets are used as reported.

**`265650d` `origin_map.rs` (new, 346 lines, 5 tests) — ours, no counterpart in
the reference.** `timestamps/` reports a span into the *prepared prompt*; the
app needs scalar offsets into the text it passed, before normalisation. The
chain is prepared byte → chunk byte → normalised word → original word, and the
last step is an edit-distance alignment over content characters, which is what
survives normalisation changing the word count (`$2` → "2 dollars", `3:30` →
"3 30"). Lifted from A's `align.rs`, whose *walk* `timestamps/` replaced but
whose *mapping* half had nothing to replace it. Verified on real model output:
every offset slices back out of the original string, and both halves of a
rewrite point at the `$2` the reader can see.

**`1f34885` the FFI layer.** `ptts_set_timings`/`ptts_word_timings` restored on
the new base, plus the fix that makes their offsets mean anything: the CPU
engine now hands `Synth` the **caller's** text rather than a normalised copy.
Asking for timings on an uncalibrated checkpoint degrades to speaking without
them instead of failing, because the app asks for them by default and a refusal
costs the utterance.

### The calibrated heads, and how they were found

`timestamp_heads` for our 40M (`english-40m-app/config.json`):

```json
"timestamp_heads": [{"layer": 3, "head": 3}, {"layer": 2, "head": 0}]
```

The reference finds these with CrisperWhisper over a 500-clip cohort. That was
not available for our distill, so the search ran on an **aligner-free metric**
instead — per head, `corr(frame, peak_token) x share_of_frames_not_moving_back x
mean_peak_attention`, read from a single deterministic all-heads pass
(`PTTS_SCAN_HEADS=1`, `RAYON_NUM_THREADS=1`; generation is not bit-reproducible
otherwise, which silently confounded an earlier 48-run scan).

**The metric was validated before being trusted.** Run over all 96 heads of
`english_2026-10` — a checkpoint the reference has already calibrated — with
`timestamp_heads` stripped from the config so nothing could leak, it ranked
their published primary `L3H8` **first of 96**, and their secondary `L2H0`
sixth. Two more of its top four (`L3H14`, `L5H2`) are heads the reference
publishes for other checkpoints in the family.

**What the metric cannot do is order the shortlist**, and this is the part worth
remembering. It scores monotonic text-tracking and never looks at a clock, so it
is blind to a head that tracks the text perfectly while firing early. On the 40M
it ranked `L3H7` second; against an ASR reference `L3H7` was *worst*, because it
needs a 200 ms lag correction. The shortlist therefore needs a clock to resolve:
whisper.cpp `small.en` with DTW word timestamps, 48 sentences, paired per word
(`L3H3` and `L2H0` statistically tied, `L3H7` eliminated at 95%). Both surviving
heads are shipped, matching the reference's own practice — 93% of its 27
checkpoints use two or more, and on `english_2026-10` the published pair beat
each of its own members in our harness.

Do **not** quote the absolute MAE from that run as our accuracy: `small.en`'s own
DTW error dominates it. It was built to rank three heads and it did.

### Costs measured on the phone (moto g play 2024, debug build)

- **RTF 1.54x** aggregate (387 s audio in 251 s wall, 43 blocks; median 1.53,
  warm 1.43–1.72). The first block of a session is 0.68–0.90x — a warm-up cost
  every listener feels, and the worst number here.
- **Voices are loaded eagerly and held.** `SynthBuilder::add_voice` registers at
  load time, so all 189 are resident: measured **+44.95 MB** (242.07 → 287.02 MB
  peak RSS, x86 release CLI, with and without `voices/`). `ptts_add_voice` (C)
  exists to avoid this, but a pack with a `voices/` directory still loads it all.
- **Native heap on device read 833 MB, and that figure is NOT trustworthy.** It
  was sampled after a hot restart, with two `[phonon] worker ready` events in the
  log, so it plausibly double-counts an orphaned engine. A fresh-start
  measurement with exactly one worker is still owed. If a hot restart really does
  orphan an engine, that is ~290 MB leaked per reload during development.

## What is NOT ours

`origin/pocket-tts-small-models` — five commits on top of main that make
xn-ptts able to run pocket checkpoints at all, and which our 40M needs by name:

```
7efa269  feat: flow_lm.lut_rank, a low-rank text table
1d7b6b4  feat: flow_lm.z_dim, a projection between the backbone and the heads
ee55842  feat: flow LM attention with its own width and grouped-query attention
99e2f90  feat: Mimi inner_dim, the width of the encoder's latents
ac24e4c  feat: flow heads other than LSD, picked by flow_lm.flow_type
```

Our `english-40m` config sets `z_dim: 1024`, `inner_dim: 32`, `flow_type: lsd`.
Adopt these rather than reimplementing pocket support.

## What the app binds to

Changing any of these breaks the app, not just the library.

- **`libptts_ffi.so`**, vendored at `android/app/src/main/jniLibs/arm64-v8a/`.
  Build it with:

  ```bash
  ARM64_FEATURES= ./android/build.sh      # the generic build
  ```

  **`ARM64_FEATURES=` is not optional.** `build.sh` defaults to
  `+dotprod,+fp16` with a comment that those are "on about every phone since
  2019"; that is false for budget SoCs. The moto g play 2024 (`bengal`, SM6225)
  reports only `fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid` — no
  `asimddp`, no `fphp`/`asimdhp` — so a featured build dies with **SIGILL in
  `ptts_new`** the moment it reaches a q8_0 kernel. Confirm with
  `adb shell cat /proc/cpuinfo` before trusting a new device.

  **Provenance, since nothing recorded it before and that cost hours twice:**

  | field | value |
  |---|---|
  | repo | `durgaprasadksv/xn-ptts` (remote `dp`) |
  | branch / commit | `pocket-conformance` @ `bee1689` |
  | flags | `ARM64_FEATURES=` |
  | NDK | r28 (`28.2.13676358`) |
  | rustc | 1.98.0 (`~/.rustup`, **not** Homebrew's — see below) |

  Record these on every rebuild. The model is pinned by commit
  (`phonon-31m-en@8c64b55e`) and the binary deserves the same.
- **Symbols the app looks up** (`lib/tts/phonon_ffi.dart`): `ptts_new`,
  `ptts_speak`, `ptts_set_timings`, `ptts_word_timings`, `ptts_voices`,
  `ptts_set_voice`, `ptts_add_voice`, `ptts_last_error`, `ptts_free`. Two of
  these exist only on our branches: `ptts_word_timings`/`ptts_set_timings` (A)
  and `ptts_add_voice` (C).
- **The timings JSON shape** `[{word, start, end, from, to}]`, parsed by
  `lib/tts/phonon_timings.dart`, where `start`/`end` are scalar offsets into the
  text passed to `ptts_speak` and `from`/`to` are seconds.
- **`pocketlocal-en`'s checkpoint layout**: `config.json`, `tokenizer.json`,
  `model.q8.gguf`, `default-voice.safetensors`, and `voices/<voice_id>.safetensors`
  for 189 catalogue voices. Note "on demand" is aspirational: a pack with a
  `voices/` directory loads all of them at `ptts_new` (section D measures the
  cost at ~45 MB).

**A build trap worth knowing before touching `ptts-ffi`.** `cpu.rs` is behind
`cfg(not(target_vendor = "apple"))`, so a host `cargo check`/`clippy`/`test` on
a Mac compiles `coreml.rs` instead and **never sees it**. A missing dependency
in `cpu.rs` passed a completely green host check this way. Always:

```bash
cargo clippy -p ptts -p ptts-ffi --target aarch64-linux-android
```

Homebrew's `rustc` on this machine is also broken (it links an `llvm` keg whose
`libz3` was relinked out from under it); use the `~/.rustup` toolchain, which has
the `aarch64-linux-android` target installed.

## Known gaps against the reference implementation

Found by reading **`dpm63/pocket-tts-timestamped`**'s Rust
(`~/tt/pocket-tts-timestamped/rust/`) — a fork of kyutai-labs/pocket-tts, not
Kyutai's own, which depends on xn-ptts at `7efa269` and adds its own wrapper.
The model code is therefore identical; everything below lives in that wrapper.

**Timestamps — DONE, 2026-11-16.** This was the big one and it gated everything
else. The reference uses two or three attention heads per checkpoint, named in
the config as `timestamp_heads` and found by searching every layer and head
against a reference aligner (41 ms MAE on its cohort). We read the middle layer,
uncalibrated.

Our 40M now declares `[{layer:3,head:3},{layer:2,head:0}]` and the module is
ported — see section D for the search, its validation, and the one thing the
search cannot do. The teacher's heads were never going to transfer (teacher 16
heads, student 8, so `L3H8` is not even a valid index — a mistake worth not
repeating, since it reads as a plausible config right up to the bounds check).

**The cost of this approach, which the middle-layer guess did not have:** named
heads are a property of *weights*, so a checkpoint nobody has calibrated gets no
timings at all. `phonon-31m-en` (Ivy) declares none and so has lost word
timings, where A's layer-averaging produced them for free on any checkpoint. She
is 6x8, the same shape as the 40M, so the same scan calibrates her.

**`end_on_pause`** (`rust/src/audio.rs:86`) has no xn-ptts equivalent: 20 ms
frames, cut after the last frame within **35 dB of the clip's own peak**, 20 ms
fade, then exactly 80 ms of silence. Peak-relative, so it adapts per voice —
which matters, because measured noise floors differ by 20 dB between voices
(Claire −63 dB, Abigail −83 dB) and an absolute threshold behaves differently on
each. The app currently hand-rolls this in Dart against an absolute −45 dBFS
with a 250 ms tail, so our block tails may be ~170 ms longer than intended.

**`pocket_text.json` is read by their wrapper and ignored by xn-ptts.** Status
for our 40M's file:

| rule | value | state |
|---|---|---|
| `append_terminal_punctuation` | `true` | done (commit `252e2c2`) |
| `capitalize_first_letter` | `true` | already in xn-ptts |
| `remove_semicolons` | `false` | nothing to do |
| `pad_with_spaces_for_short_inputs` | `false` | xn-ptts deliberately never pads |
| `default_temperature` | `0.3` | matches xn-ptts's default |
| `replace_characters` | `’→'`, `‘→'` | **nothing to do** (measured) |

`replace_characters` was carried as an open task on the strength of a tokenizer
measurement that bypassed the pipeline. Measured properly: xn-ptts's own
normaliser already folds fifteen single-quote variants to ASCII `'`
(`preprocess.rs:1081`) and the app passes `en`, so by the time text reaches the
tokenizer the curly quotes are gone. The feared +27–38% token cost does not
exist on this path. Retired.

The byte-vs-codepoint reasoning below it is still worth keeping, because it is
why the mapping in `origin_map` walks bytes and reports scalars.

**`snap_starts`** (`voiceengine/app/timings.py:78`) is unported: it moves each
word's start to the first frame above `max(-40 dB, word_peak - 18 dB)` within
that word's span, so the highlight does not lead the voice. It touches no
offsets and the worker has both the PCM and the timings, so it ports cleanly.

**A known xn-ptts bug, flagged by Kyutai themselves** in `audio.rs`:
*"TODO: upstream to xn-ptts: `ptts::audio::decode_file` keeps only the first
channel and does not trim the MP3/AAC encoder delay."* Harmless for our `.wav`
references; it would bite if voice references ever arrive compressed.

**Sampler parameters are identical** and need no work — verified field by field:
temperature 0.3, `cfg_coef` none (≡ 1.0), seed 4242424242424242, EOS threshold
−4.0, decode steps 1, `noise_clamp` none, `MAX_TOKENS_PER_CHUNK` 50,
`teacher_force_previous_chunk` off in both. `split_into_best_sentences` already
mirrors the Python.

## Migration notes

1. ~~Keep `dp/android-timings` shipping until a rebuilt path renders one
   correct utterance.~~ **Done 2026-11-16**: `dp/pocket-conformance` renders and
   highlights correctly on the phone, EPUB and PDF. `dp/android-timings` is the
   fallback if a regression appears; the previous `.so` and pack config should be
   kept until the new path has run for a while.
2. ~~Adopt gabriel's five commits and check whether they resolve
   `bos_before_voice`.~~ **Resolved**: they do not. See C.
3. **Offer `fccd4a6`/`05cab1c` and `252e2c2`/`2e476b7` upstream** (gabriel's
   fork is the natural home for both). Still open, and still the right trade: a
   thin fork rebases cheaply; carrying conformance fixes forever does not. The
   timings work is NOT in this category — issue #321 settled that.
4. ~~The head search gates the timings rewrite.~~ **Done** — see D. The method
   is reusable: the same scan calibrates Ivy, or any future checkpoint, without
   needing an aligner to shortlist.
5. **Reasons to rebase now that did not exist before**: `0470f02`
   (`perf(flow): compute the flow net's time conditioning once` — our 40M is
   `lsd`, two time conditions, so this is on the hot path) and
   `18aa467`/`5da4d76` (opt-in q8_0 Mimi decoder on the CPU — Mimi is f32 for us
   and ~41 MB of a 410 MB native heap). Both land on the two open problems: RTF
   1.44 and memory. Try cherry-picking them before committing to the full
   150–200 line surgery through upstream's `Flow` refactor.
