# BG ASR — roadmap / next actions (sequenced)

**GP-v2 shipped (2026-08-08):** **4.12%** test WER on the frozen parliament holdout, from
1,744.8 h across eight registers — v1's 5.29% at 649 h is the baseline it replaced
([BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md)). This file is the ordered index of what comes
next; each phase links to its detailed plan.

**Status of the phases below:** 0, 1, 2 and 2b are resolved — see "Where GP-v2 landed" after
the phase list. Phases 2c onward are still open.

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
- Deployment note: M stays the edge/streaming artifact (**73.6 MB** int8 sherpa-onnx, measured
  2026-08-08 — the old "~66 MB" was params × 1 byte, not a real export); L (if built)
  is the server-side best model. Published M→L gains at ~1–2k h are ~5–10 % relative — worth
  having only after the data levers (balance/mux, broad LM, self-training round 2) are spent.

### 2c. Whisper-BG fine-tune — second architecture (FUTURE WORK, after GP-v2)

LoRA/PEFT fine-tune of **whisper-large-v3** on the GP corpus (full FT of 1.5B is infeasible on
16 GB; LoRA fits, ~2–4 GPU-days, 1–2 epochs). Train on **raw punctuated** texts (stored in
pseudolabels.jsonl + punctuated stenograms) to preserve Whisper's punctuation ability. Three
payoffs, in value order:
1. **Teacher upgrade** — relabel the weak axes (BNR/YT ~700 h) with the BG-FT Whisper =
   self-training round 2 with a stronger teacher → better GP-v3 data.
2. **Complementary deployment** — Zipformer wins covered registers/streaming/edge; Whisper-FT
   likely wins the unseen tail; **agreement-mode jobs** (both decode; agree → auto, disagree →
   human review) as a high-stakes product pattern (the labeling arbitration, productized).
3. **Benchmark sparring partner** — an independent strong model keeps the eval honest.
Sequencing: strictly after GP-v2 (same GPU; GP-v2 answers the diversity question first).

### 3. Streaming model (`--causal 1`)
Re-train the Zipformer as a **streaming** (chunk-wise, causal) model for real-time / on-device
use. Reuses the same fbank + BPE + (v2) corpus; costs ~1–2 % abs WER vs offline but enables
live captioning and a sherpa-onnx **streaming** export. Do this **after v2** so it trains on
the larger, cleaner corpus.

## Where GP-v2 landed (2026-08-08) — what's resolved, what's now blocking

- **Phase 2 delivered, and the data bet was right.** 649 h → 1,744.8 h with *nothing else
  changed* moved the frozen test from 5.29 % to **4.12 %** (−22 % rel), dev 5.29 → 3.94,
  and dev loss 0.1315 → 0.1180. Substitutions carried the gain (3038 → 2256, −26 %).
- **Phase 2b: Zipformer-L is NOT triggered.** The stated gate was "train loss floors early
  while dev still improves". Neither floored — at epoch 30 train was still descending
  (0.1839 → 0.1827) alongside dev (0.1192 → 0.1180), with Eden's LR already near zero. That
  is *under-trained relative to the data*, not capacity starvation. Cheaper levers first: a
  longer schedule (~40 epochs), then mux rebalancing, then L.
- **The external LM is retired.** It was v1's best system and is now a loss on every set,
  even rebuilt on the full GP corpus text — the benchmark's LM section has the isolating
  experiment (OOV halved, WER worse). Ship LM-free `modified_beam_search`; this also
  simplifies the ONNX/streaming export, which no longer needs a lexicon or graph.
- **NEW BLOCKER — the generalization claim has no clean measurement.** Folding FLEURS/CV
  *train* into training (the phase-2 lever) consumed the only OOD sets we had: their test
  splits are verified leak-free but are now *in-corpus*, and the Whisper comparison no
  longer holds training exposure constant. The fresh self-recorded set (phase 1's leftover)
  is therefore promoted from "nice to have" to **the next thing to build** — without it,
  4.12 % on parliament is the only number GP-v2 can honestly claim.

## Rationale for this order

Evaluation before retraining: **1** tells us whether the model is already good enough
off-domain (and gives Whisper context), *before* investing GPU-days. **2** is the one phase
that actually lowers real-world WER (more data + cleaner data). **3** changes the deployment
mode (offline → streaming) and should ride on the best/largest corpus, hence last.

## Cross-references
[BG_ASR_PLAN.md](BG_ASR_PLAN.md) (v1 recipe/inventory) ·
[BG_ASR_V2_PLAN.md](BG_ASR_V2_PLAN.md) (recovery + data-quality detail) ·
[BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md) (results + future work).
