# Bulgarian ASR v2 — corpus recovery & data-quality plan

**Status:** drafted 2026-08-03, during the v1 baseline run (epoch ~19/30). Companion to
[BG_ASR_PLAN.md](BG_ASR_PLAN.md) (v1 recipe/inventory) and the `3edd7d6` commit
("quarantine stale parliament sessions"). Scope: recover the quarantined parliament
audio and close the audio/transcript **desync** problem across both the ASR corpus and
the Accentor labels — *without* disturbing the running v1 baseline or the read-only
Kaldi aligner.

Do **not** start any of this until the v1 baseline finishes 30 epochs and is decoded —
v1 is the reference number every v2 change is measured against, on the **identical
frozen test set**.

## Background — two flavors of stale audio

A `refresh_bg_parliament` pass (mtimes 2026-07-02 … 07-30) replaced 110/212 parliament
FLACs with **shorter** audio but did not re-derive the Kaldi segments. There are two
distinct failure modes, and only the first is currently detected:

1. **End-trim** — a segment's end runs past the shorter file's EOF. `extract-segments`
   drops/truncates it → mostly *lost yield*. `detect_stale_sessions.py` catches this
   (max segment-end > file duration + tol) → 110 sessions quarantined
   (`data/lhotse/bg/stale_sessions.txt`, `data/bg_parliament/manifest_stale.jsonl`,
   256.9 h).
2. **Start-trim / content-swap** — every timestamp is shifted but stays **in-bounds**.
   The segment points at the *wrong* audio: it sounds desynced but is **silently kept**.
   `detect_stale_sessions.py` (duration check only) **does not catch this**, so such
   sessions may still be in the v1 649 h training corpus.

## Finding (2026-08-03) — the desync is real, and vetting already absorbed it for the Accentor

Native vetting verdicts (`data/kaldi/vetting_results.jsonl`, 402 parliament tokens)
cross-referenced against the stale set:

| Session origin | Tokens | Marked bad/unsure | Reject rate |
|---|---|---|---|
| **Stale** | 215 | 161 | **75 %** |
| Consistent | 187 | 82 | 44 % |

Stale-session tokens were rejected ~1.7× more often — the fingerprint of audio/transcript
desync, concentrated in the mismatched sessions. **This protected the Accentor rather than
harming it:** `build_accentor_finetune.py` drops every `bad`/`unsure` token before
training, so those 161 desynced tokens never entered the model; the 54 stale tokens kept
`ok` were each human-confirmed (word+stress correct ⇒ synced at that token). Desync is an
*audible* error, which manual vetting catches near-perfectly.

**Residual Accentor risk:** TIER-3 homograph classes keep the aligner's variant for tokens
never *sampled* for vetting — an unsampled desynced stale token could slip through (diluted
5:1 by chitanka). Small; closed by workstream **B** below.

**The live exposure is the ASR corpus**, which has *no* per-token human gate and where the
in-bounds desync (mode 2) is undetected. Workstreams **C**/**D** address it.

## Workstreams

### A. Re-segment the 110 quarantined sessions  →  +~257 h (train only)

`resegment_stale_sessions.sh` (drafted 2026-08-03). Same pipeline as
`segment_asr_sessions.sh` stages 1–2 (`tri2b_comb` biased-LM segmentation → `align_fmllr`
`tri4b` → `clean_and_segment_data`), seeded from `manifest_stale.jsonl` into a fresh
`parl_restale` workspace. Re-derives segments against the **current** audio, so the
mismatch is repaired. CPU-only; aligner models read-only; v1 dirs untouched. Output
`exp/kaldi_pilot/data/parl_restale_cleaned` (segments/text/wav.scp, lhotse-importable).

Integration (train-only — the frozen dev/test contains **none** of these sessions, so
v1↔v2 stay comparable):
- add **`--reseg-dir`** to `prep_lhotse_manifests.py` (TO BUILD): for a recovered
  recording, use the new `parl_restale_cleaned` segments and **ignore the old stale ones**
  (they share the audio file, differ only in segments). Still-broken sessions stay excluded
  via the blocklist.
- `compute_fbank_bg.py --splits train` (dev/test fbank frozen), then retrain v2 from
  scratch, decode the **same** frozen test set.

### B. Accentor closure — re-harvest + `--diff-old`

After A, re-harvest the homograph stress labels from `parl_restale_cleaned` and diff the
per-homograph **majority** stress against the old (possibly-stale) labels. Add a
`--diff-old <old_labels>` report to the harvest step. Any homograph whose majority *flips*
is a concrete harm candidate; if nothing flips beyond already-vetted classes, the shipped
Accentor is **confirmed** clean (not just probably). Cheap once A exists.

### C. Corpus self-clean — decode the train set with the v1 baseline

Standard forced-decoding cleanup, enabled by the finished v1 model: decode the **training**
cuts and flag utterances with pathological WER. A desynced (mode-2) utterance scores near
100 % WER because the audio says nothing like the transcript — so this surfaces in-bounds
desync **corpus-wide**, across all sessions, not just the 110. Drop/quarantine the
high-WER tail before building the v2 train set. (Guard: also catches genuinely hard audio;
use a conservative threshold + spot-check, and log what's dropped — no silent truncation.)

### D. Upgrade `detect_stale_sessions.py` — catch in-bounds desync

Current detector = end-overrun only. Add an **alignment-score** signal (per-session mean
acoustic likelihood / fraction of well-aligned frames from a quick `tri4b` re-align, or
reuse C's per-utterance WER aggregated per session). Flag sessions whose alignment score is
an outlier even when segments are in-bounds. This makes the quarantine robust to start-trim
and content-swap, not just end-trim.

## Sequencing

v1 baseline finishes + decoded (reference WER)  →  **C** (self-clean, needs the v1 model)
∥ **A** (re-seg, CPU, independent)  →  **D** (fold in C's scores)  →  **B** (Accentor
closure, needs A)  →  build v2 train set (recovered + self-cleaned, frozen dev/test) →
retrain v2 → decode same frozen test → compare v1↔v2 in `docs/BG_ASR_BENCHMARK.md`.

## Guardrails (unchanged from v1)

- Read-only: `tri2b_comb`/`tri4b`/`tdnn_a`/`data/lang` and all v1 `*_cleaned` dirs.
- All new work in fresh dirs (`parl_restale*`, `exp/*_parl_restale`).
- **Frozen dev/test never change** — every v2 delta lands in train only, so WER stays
  comparable to v1 on the identical holdout.
- No silent caps — C/D must log every dropped utterance/session.

## Verification

- A: `parl_restale_cleaned/segments` non-empty; report `recovered N/110` sessions + hours.
- B: `--diff-old` shows 0 unexpected homograph-majority flips (or names them).
- C: high-WER drop list reviewed; corpus hours before/after logged.
- D: re-run flags the known 110 **plus** any in-bounds desync C found.
- v2: TEST WER (same frozen split) vs v1 — quantifies what +257 h recovered − noise buys.

## Deliverables

`resegment_stale_sessions.sh` · `prep_lhotse_manifests.py --reseg-dir` · re-harvest
`--diff-old` report · v1-baseline train-set WER filter + drop list · upgraded
`detect_stale_sessions.py` · v2 Zipformer + v1↔v2 comparison row.
