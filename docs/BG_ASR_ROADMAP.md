# BG ASR — roadmap / next actions (sequenced)

**v1 shipped:** 5.29% test WER on the frozen parliament holdout
([BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md)). This file is the ordered index of what comes
next; each phase links to its detailed plan.

## Direction — GENERAL-PURPOSE (decided 2026-08-04)

Target is **general-purpose Bulgarian ASR**, not a parliament-only tool. The 3-domain benchmark
showed v1 is a **specialist** (5.29 in-domain, ~20–22% OOD) vs Whisper the **generalist** (flat
~11–15%). Closing that gap needs **data diversity**, not more of the same register — 397 h of
audiobook is already in training yet OOD stayed ~20–22%. So the primary next phase is a
**diversified corpus** folding FLEURS + Common Voice into TRAINING (phase 2). Phases 0–1 (Whisper
comparison + OOD eval) are **DONE** — results in the benchmark.

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
- **NOTE on the wiki tool:** `fetch_eval_wild.py` (→ `data/eval_wild/wiki.txt`) pulls unseen
  BG Wikipedia **text** — built for out-of-register *Accentor* eval (`eval_accentor_ood.py`),
  it has **no audio**, so it can't be an ASR test set directly. Do NOT synthesize audio from
  it via TTS and ASR that back — circular (tests TTS artifacts, doubly so with our own TTS).
- **Best fit — FLEURS BG** (Google): Bulgarian **Wikipedia sentences read aloud** with audio +
  transcripts — the audio analog of `fetch_eval_wild.py`, genuinely OOD vs. parliament (read
  register, different speakers, clean recording). **Common Voice BG** (Mozilla) as a second
  OOD row (more speakers/noise). Score both with the SAME normalizer (`bg_verbalize` +
  case/punct/stress) as the Whisper harness.
- **Deliverable:** an OOD WER row (FLEURS ± Common Voice) alongside the in-domain 5.29% →
  shows how far generalization holds off-domain (expect meaningfully higher; that gap is the point).

### 2. V2 = diversified general-purpose corpus  ← PRIMARY
Build a broader training set and retrain:
- **Diversity (the scope lever):** add **FLEURS-train** + **Common Voice train/validated** (bg)
  to training — new speakers, mics, registers audiobooks/parliament lack. Reuse
  `prep_fleurs_bg.py` / `prep_commonvoice_bg.py` on the *train* splits.
- **Parliament recovery (deepen in-domain):** [BG_ASR_V2_PLAN.md](BG_ASR_V2_PLAN.md) — recover
  the 110 quarantined sessions (+~257 h) via `resegment_stale_sessions.sh` →
  `prep_lhotse_manifests.py --reseg-dir` → self-clean in-bounds desync → retrain.
- **Splitting discipline:** train on the FLEURS/CV **train** splits; keep their **test** splits
  held out (no utterance leakage) + the frozen parliament dev/test. **Caveat:** once FLEURS/CV
  are in training they are no longer *OOD* — the v1 numbers (FLEURS 20.65 / CV 22.27) are the
  **pre-diversification baseline to beat**, and a truly-OOD measure for the general model needs a
  **new unseen domain** (self-recorded set — see phase 1). **Target:** pull the ~20% OOD toward
  Whisper's ~15% while keeping parliament strong.

### 2b. Model-size ladder — Zipformer-L gated on GP-v2 evidence (decided 2026-08-04)

Change ONE variable at a time so effects stay attributable:
- **GP-v2 = Zipformer-M (65M) on the ~1,750 h corpus** — isolates the DATA effect vs v1
  (same architecture). Known VRAM behavior (M @ max-duration 350 ≈ 94.7% of 16 GB); ~1.5–2
  days under `train_resumable.sh`.
- **GP-v3 = Zipformer-L (~150M), only if GP-v2 shows capacity starvation** (train loss floors
  early while dev still improves with data; or the mean-WER target is missed with data-side
  levers exhausted). L needs `--max-duration ~200–250` on 16 GB (~2× compute/batch → ~3–4
  days). Isolates the CAPACITY effect.
- Deployment note: M stays the edge/streaming artifact (~66 MB int8 sherpa-onnx); L (if built)
  is the server-side best model. Published M→L gains at ~1–2k h are ~5–10 % relative — worth
  having only after the data levers (balance/mux, broad LM, self-training round 2) are spent.

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
