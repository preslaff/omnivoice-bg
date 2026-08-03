# BG ASR — roadmap / next actions (sequenced)

**v1 shipped:** 5.29% test WER on the frozen parliament holdout
([BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md)). This file is the ordered index of what comes
next; each phase links to its detailed plan.

## Order of next actions

### 0. Whisper external comparison — *in progress*
`eval_whisper_bg.py` — faster-whisper large-v3 on the frozen **test** cuts, scored through a
shared normalizer (parity is mandatory — see [BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md)
"Future work"). Gives the legible "custom model vs off-the-shelf Whisper" row. No training.

### 1. General-purpose (out-of-domain) audio eval
The 5.29% is **in-domain parliament only** — distribution shift is unmeasured. Evaluate the
v1 model on **non-parliament** Bulgarian audio to quantify robustness beyond the domain.
- **Caveat:** the AB400 audiobook is already **in the training set**, so it cannot serve as a
  clean OOD test (leakage). A true general-purpose eval needs **held-out non-parliament**
  audio with zero overlap with training.
- **Candidates:** Common Voice BG, FLEURS BG (Google), or a fresh *chitanka* audiobook slice
  confirmed absent from training. ("chitanka" is a literary corpus — verify no audio overlap.)
- **Deliverable:** an OOD WER row alongside the in-domain 5.29% → shows how far generalization
  holds off-domain (expect meaningfully higher; that gap is the point).

### 2. V2 training with the re-segmented data
Execute [BG_ASR_V2_PLAN.md](BG_ASR_V2_PLAN.md): recover the 110 quarantined sessions
(+~257 h, **train only**) via `resegment_stale_sessions.sh` → `prep_lhotse_manifests.py
--reseg-dir` (to build) → self-clean in-bounds desync (decode the train set with the v1
model, drop pathological-WER utts) → retrain → decode the **same frozen test** for a clean
v1↔v2 delta. Frozen dev/test never change, so the comparison stays honest.

### 3. Streaming model (`--causal 1`)
Re-train the Zipformer as a **streaming** (chunk-wise, causal) model for real-time / on-device
use. Reuses the same fbank + BPE + (v2) corpus; costs ~1–2 % abs WER vs offline but enables
live captioning and a sherpa-onnx **streaming** export. Do this **after v2** so it trains on
the larger, cleaner corpus.

## Rationale for this order

Evaluation before retraining: **1** tells us whether the model is already good enough
off-domain (and gives Whisper context), *before* investing GPU-days. **2** is the one phase
that actually lowers real-world WER (more data + cleaner data). **3** changes the deployment
mode (offline → streaming) and should ride on the best/largest corpus, hence last.

## Cross-references
[BG_ASR_PLAN.md](BG_ASR_PLAN.md) (v1 recipe/inventory) ·
[BG_ASR_V2_PLAN.md](BG_ASR_V2_PLAN.md) (recovery + data-quality detail) ·
[BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md) (results + future work).
