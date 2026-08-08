# Bulgarian ASR — benchmark (v1 · GP-v2)

**Headline: 4.12% WER** on the frozen parliament test set (unseen sessions) — GP-v2, a
65.5M-param Zipformer transducer trained from scratch on **1,744.8 h** of Bulgarian speech
across eight registers. Supersedes v1's 5.29% (649 h, same architecture, same frozen split),
a **22% relative** gain from data diversity alone. Companion to
[BG_ASR_PLAN.md](BG_ASR_PLAN.md) (recipe/inventory),
[BG_ASR_DATA_PIPELINE.md](BG_ASR_DATA_PIPELINE.md) (corpus construction) and
[BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md) (what's next). Reference paper: `2203.14876`
(Finnish Parliament ASR).

# GP-v2 (2026-08-08) — the general-purpose model

## Model & data

| | |
|---|---|
| Architecture | Zipformer transducer (pruned), non-causal, **65,549,011** params — *identical to v1* |
| Features / tokenizer | 80-dim fbank @ 16 kHz · BPE-500 (same model as v1) |
| Training | fp16, ScaledAdam + Eden, 30 epochs, `--max-duration 350`, **MUSAN CutMix on**; final dev loss **0.1180** (v1: 0.1315) |
| Decode model | epoch 30, average of last 5 (`--avg 5 --use-averaged-model`) |
| Train | 872,975 cuts / **1,744.8 h** — plsten 573.3 · AB400 397.5 · ytdr 292.6 · ytgi 214.0 · ytbb 135.1 · BNR 102.9 · sivov 12.8 · FLEURS-train 9.5 · CV-train 7.0 |
| Dev / Test | unchanged frozen parliament holdout (6.2 h / 16.4 h) |

Only the **data** changed vs v1 — architecture, tokenizer, schedule and split are identical,
so the delta is attributable (the roadmap's one-variable-at-a-time discipline).

## Results

| Decoding method | LM | dev WER % | **test WER %** | FLEURS | CV |
|---|---|---|---|---|---|
| greedy_search | — | 4.00 | — | 10.01 | 9.01 |
| **modified_beam_search (beam 4)** | — | **3.94** | **4.12** | **9.84** | **8.69** |
| fast_beam_search_nbest_LG @ 0.1 | GP word 3-gram | 3.99 | — | 11.13 | 11.20 |
| fast_beam_search_nbest_LG @ 0.2 | GP word 3-gram | 3.99 | — | 12.80 | 11.58 |
| fast_beam_search_nbest_LG @ 0.2 | *v1* parliament 3-gram | 4.12 | — | — | — |

Test error breakdown (best system, 123,605 words): **5087 errors = 1314 ins / 1517 del /
2256 sub** (v1: 6543 = 1487/2018/3038). Substitutions fell 26% — the signature of a
diversity intervention: fewer wrong-word confusions, not merely better silence handling.
dev 3.94 → test 4.12 is a tight gap, so the LM sweep did not lean on dev.

### The external LM stopped helping — and the rebuild proved why

On v1 the word 3-gram was the best system (5.46 → 5.29). On GP-v2 it **loses everywhere**,
even after being rebuilt on the full GP corpus text (14.38 M words / 296,230 types, vs v1's
7.91 M / 164,262 — `bg_word_3gram_gp.arpa`, `data/lang_bpe_500_gp/LG.pt`):

| | v1 LM | GP LM |
|---|---|---|
| dev OOV | 0.26 % | 0.15 % |
| FLEURS OOV | 4.98 % | 3.21 % |
| CV OOV | 7.90 % | **4.06 %** |

The rebuild **halved CV's OOV and CV's WER still got worse** (8.69 → 11.20), with
substitutions rising 2006 → 2376. That isolates the mechanism: an n-gram contributes
coverage *and* a phrasing prior, and only coverage improved. The prior is dominated by
parliament/audiobook/YouTube speech, so on read-encyclopedic or crowd-read input the LM
actively rewrites correct words into corpus-likelier ones. A bigger n-gram cannot fix this;
only register-matched text could, and for "arbitrary Bulgarian" no such text exists.
The 1.7 B-param-equivalent lesson in one line: **the transducer's own prior now exceeds a
3-gram's, so the shipped system is LM-free.** `LG.pt` and the ARPA are kept on disk, unused.

### ONNX / int8 export parity (2026-08-08)

Exported for CPU serving (`zipformer/export-onnx.py --epoch 30 --avg 5 --use-averaged-model 1`,
int8 dynamic quantization) and scored with `eval_onnx_bg.py` — sherpa-onnx on CPU, torch-free,
same `bg_text_norm` normalizer and `jiwer` scoring as every other row here:

| System | dev WER % | Δ | RTF (4 threads) |
|---|---|---|---|
| torch fp32, greedy (icefall `decode.py`) | 4.00 | reference | — |
| ONNX fp32, greedy | 4.03 | +0.03 | 0.008 |
| **ONNX int8, greedy** | **4.06** | **+0.03 vs ONNX fp32** | 0.007 |

**Quantization costs 0.03 points** — and the ONNX↔torch gap is smaller than the difference
between the two scoring paths (47,717 reference words here vs 47,605 in icefall's scorer), so
it is not resolvable as a regression. sherpa-onnx's fbank defaults (`low_freq=20`,
`high_freq=-400`, `dither=0`, `snip_edges=False`) are identical to the lhotse config used for
training, which is why the feature pipeline transfers without adjustment.

Caveat: measured on a Ryzen 9 9900X3D with `avx512_vnni`. `export-onnx.py` does not pass
`reduce_range=True`, so **int8 accuracy and speed must be re-measured on any non-VNNI
deployment CPU** before trusting these numbers there.

## Caveats — read before quoting the FLEURS/CV numbers

- **FLEURS and CV are no longer out-of-domain.** GP-v2 trained on their *train* splits, so
  9.84 / 8.69 measure held-out-split, **in-corpus** performance — domain adaptation, not
  generalization. The v1 → GP-v2 movement on these sets (20.35 → 9.84, 22.27 → 8.69)
  conflates the two and must not be quoted as a generalization gain.
- **Not leakage, verified:** zero cut-id, zero recording-id and **zero sentence-text**
  overlap between the train and test splits of both corpora (0/658 FLEURS, 0/3201 CV).
- **Unverifiable residual:** speaker overlap in FLEURS. Its splits are sentence-level
  (inherited from FLoRes) and it publishes no speaker ids, so the same voices may appear in
  both splits. CV's official splits are speaker-disjoint by construction.
- **The Whisper head-to-head no longer holds training exposure constant** — Whisper is
  zero-shot on FLEURS while we adapted to it. GP-v2's only unambiguous number is the
  parliament one (4.12 %). A fresh self-recorded set in nobody's training data is now the
  blocking dependency for any generalization claim (roadmap phase 1).

## Reproducibility (GP-v2)

- Corpus: `merge_train_corpus.py` → `bg_cuts_train_gp_shuf.jsonl.gz` (global seeded shuffle).
- Train: `asr/train_resumable.sh` with `EXP=zipformer/exp_gp ENABLE_MUSAN=true
  TRAIN_MANIFEST=bg_cuts_train_gp_shuf.jsonl.gz`.
- LM (built, measured, not shipped): `prep_lm_text.py --cuts <gp manifest> --exclude-cuts
  <dev test fleurs cv>` → `TAG=_gp PREP=0 LEAK_CHECK=ids build_kenlm_bg.sh` →
  `TAG=_gp compile_lg_bg.sh`. `LEAK_CHECK=ids` asserts zero cut/recording-id overlap with
  every eval manifest — the phrase-level assertion is v1-only and *false-positives* on GP
  text, because its "train" reference is the three Kaldi dirs and cannot see recovered
  sessions or the non-parliament axes.
- Decode: `zipformer/decode.py --epoch 30 --avg 5 --use-averaged-model 1 --exp-dir
  zipformer/exp_gp --decoding-method modified_beam_search --beam-size 4 --eval-set {dev,test}`.

---

# v1 (2026-08-03) — the parliament specialist

Retained as the historical record and the baseline GP-v2 is measured against.

**v1 headline: 5.29% WER**, 649 h, parliament + audiobook only.

## Model & data

| | |
|---|---|
| Architecture | Zipformer transducer (pruned), non-causal, **65,549,011** params |
| Features / tokenizer | 80-dim fbank @ 16 kHz (no speed perturb) · BPE-500 |
| Training | fp16, ScaledAdam + Eden, 30 epochs, `--max-duration 350`; final dev loss 0.1315 |
| Decode model | epoch 30, average of last 5 (`--avg 5 --use-averaged-model`) |
| Train | 419,915 cuts / **649.2 h** (parliament + AB400 audiobook) |
| Dev | 4,546 cuts / **6.2 h** (whole-session holdout, tuning) |
| Test | 11,472 cuts / **16.4 h** / 123,605 ref words (whole-session holdout, **decoded once**) |

Frozen whole-session split (`data/lhotse/bg/split/`) — zero train↔dev↔test session overlap.

## Results

| Decoding method | LM | dev WER % | **test WER %** |
|---|---|---|---|
| greedy_search | — | 5.29 | 5.52 |
| modified_beam_search (beam 4) | — | 5.19 | 5.46 |
| **fast_beam_search_nbest_LG** (ngram-lm-scale 0.2) | word 3-gram (KenLM) | **5.14** | **5.29** |

Test error breakdown (best system, 123,605 words): **6543 errors = 1487 ins / 2018 del /
3038 sub**.

### External comparison — Whisper large-v3 (zero-shot), 2026-08-04

Off-the-shelf Whisper large-v3 on the **same** frozen parliament test set, scored through the
**same** `bg_verbalize` normalizer (`eval_whisper_bg.py --model models/faster-whisper-large-v3`):

The specialist↔generalist crossover (same sets, same `bg_verbalize` normalizer):

| Test set | BG Zipformer (ours) | Whisper large-v3 | winner |
|---|---|---|---|
| Parliament (our domain) | **5.29** | 15.37 | ours, ~2.9× |
| FLEURS (read Wikipedia, Whisper's turf) | 20.35 | **11.97** | Whisper, ~1.7× |
| Common Voice (crowdsourced read, proxy neutral) | 22.27 | **15.42** | Whisper, ~1.4× |

**Each model wins its home turf, and the *shape* is the story.** Ours is a **specialist** — superb
in-domain (5.29) but a large swing off-domain (~20–22% on both read-speech OOD sets, ~4×). Whisper
is a **generalist** — flat ~11–15% across all three (~1.3× max/min), never great, never bad. On the
(proxy) neutral Common Voice, Whisper wins 15.42 vs 22.27.

**Neutrality caveat — read carefully:** Common Voice is public and almost certainly in Whisper's
training. Tellingly, Whisper's CV (15.42) ≈ its parliament (15.37) — i.e. ~15% is just Whisper's
general-Bulgarian baseline (possibly aided by having seen CV), so this still tilts toward Whisper.
A **self-recorded fresh set** remains the only clean tiebreaker (see [BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md)).
Parliament-Whisper breakdown: 10,354 sub / 3,654 del / 5,035 ins / 123,874 words. Minor scorer
caveat: ours via icefall, Whisper via `bg_verbalize`; both refs number-spelled, so paths differ << the gaps.

### Out-of-domain — FLEURS BG (read Wikipedia), 2026-08-04

Zero-adaptation eval on FLEURS `bg_bg` **test** (658 unique recordings / 1.9 h) — Bulgarian
Wikipedia sentences read aloud: a different register + speaker pool, clean studio audio. Same
epoch-30 / avg-5 model, same `bg_verbalize` normalizer. (Corrected 2026-08-04 after a cut-id-
collision fix, commit b96eb3a; biased vs clean differ ~0.3 pt — a weighting bias, not garbage.)

| Set | Method | WER % |
|---|---|---|
| FLEURS OOD | greedy_search | **20.35** |
| FLEURS OOD | modified_beam_search | 20.44 |
| FLEURS — Whisper large-v3 (ref) | greedy | 11.97 |

**In-domain 5.29% → OOD ~20.4% (≈4× relative).** This is the *domain-specialization cost*, not
a defect. FLEURS is clean audio, so the gap is purely register / vocabulary / speaker shift
(spontaneous parliamentary speech + its lexicon vs. read encyclopedic Wikipedia), not noise:
- The model **generalizes** — ~20.4% on a wholly unseen domain/speakers means it learned real
  Bulgarian ASR, not memorized parliament (a memorizer would be far worse). Deletions + substitutions
  dominate = out-of-vocabulary encyclopedic terms and names — the expected OOD failure mode.
- **Beam ≥ greedy here** (20.44 vs 20.35), the reverse of in-domain: on OOD input the model's
  internal prior is domain-mismatched, so a wider search doesn't help. The parliament-trained
  KenLM/LG is likewise domain-biased and would not be expected to help FLEURS, so **greedy is
  the clean acoustic-generalization number**.

### Reading the numbers

- **Best system 5.29% test WER** — competitive with published in-domain parliament ASR,
  and notable for being trained *from scratch* on a self-built, force-aligned corpus.
- **The transducer is a strong LM in itself.** Greedy → beam gains almost nothing
  (5.52→5.46), because the 649 h model already predicts Bulgarian well. The external
  n-gram (LG) adds a real but modest 0.17 % abs on test (5.46→5.29, ~3 % rel).
- **Healthy generalization.** dev↔test gap is small and consistent across methods
  (LG +0.15, beam +0.27, greedy +0.23). A tight gap means the tuning did not lean on dev;
  test reflects true performance on unseen sessions.
- The LG scale (0.2) was chosen on **dev** (sweep 0.01→0.3, clean single minimum); test was
  decoded exactly once with the pre-registered three-method ladder.

### Speed (batched offline decode, RTX 5070 Ti, max-duration 600)

The 16.4 h test set decoded in roughly: greedy ~20 s, modified_beam_search ~120 s,
fast_beam_search_nbest_LG ~80 s wall — i.e. **hundreds to thousands× faster than real time
in batch**. NOTE: these are batched-throughput figures (many utterances in parallel); a
single-stream latency RTF (batch=1) would be higher and is a separate measurement. The
model is offline/non-causal — excellent for batch/near-live transcription, not word-level
streaming (a `--causal 1` variant would be a separate run).

## Position & caveats

- **Corpus-size slope, not converged SOTA.** In-domain parliament ASR saturates ≈ 1800 h;
  at 649 h we are on the steep part. The ~257 h recoverable via
  [BG_ASR_V2_PLAN.md](BG_ASR_V2_PLAN.md) workstream A should move this down.
  *(Settled by GP-v2: 1,744.8 h → 4.12 % test, so the slope was real.)*
- **Reference-transcript ceiling.** References are stenographic + Kaldi-aligned; some
  residual WER is reference noise, not model error.
- **Domain matters (measured).** In-domain parliament = 5.29%; out-of-domain read Wikipedia
  (FLEURS) = ~20.4% (§ Out-of-domain above). Conversational / dialectal / noisy audio is still
  unmeasured and would likely differ again.

## Reproducibility

- LM text (dev/test-excluded): `prep_lm_text.py` → `build_kenlm_bg.sh` (leakage assertion).
- LG graph: `compile_lg_bg.sh` → `data/lang_bpe_500/LG.pt`.
- Decode: `zipformer/decode.py --epoch 30 --avg 5 --use-averaged-model 1 --eval-set {dev,test}
  --decoding-method <m> [--ngram-lm-scale 0.2]` (the `--eval-set` switch is a BG addition).
- Env: torch 2.8.0+cu128, k2 1.24.4 (sm_120), lhotse 1.33.0.

## Next

- **v2 — DONE (2026-08-08).** Sessions recovered, corpus diversified to 1,744.8 h, retrained,
  same frozen test decoded once: **5.29 % → 4.12 %**. See the GP-v2 section at the top.
- Remaining, in order: fresh self-recorded eval set (the only clean generalization measure) ·
  sherpa-onnx int8 export of GP-v2 · self-training round 2 (GP-v2 as teacher for the
  weak-labeled BNR/YT axes) · Zipformer-L only if the cheap levers are spent
  ([BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md)).

## Future work / benchmark TODO

These add benchmark credibility, not product WER — the transducer is already the best system.

### Whisper external comparison — RECOMMENDED (cheap, high-legibility)

The single most legible external reference: "Zipformer 5.29% vs Whisper-large-v3 X%" on the
**same frozen test set**. Zero-shot (never trained on this data), so it answers "does the
custom 649 h model beat the best off-the-shelf multilingual model?" — almost certainly yes,
a favorable result to report.

- **Harness:** `eval_whisper_bg.py` (faster-whisper large-v3 on the frozen test cuts →
  shared normalizer → `jiwer`). Short pre-segmented cuts sidestep Whisper's long-form
  hallucination. Install: `pip install faster-whisper jiwer`.
- **Normalization parity (handled):** both Whisper output and references go through one
  normalizer — lowercase, strip punctuation, strip U+0301 — so Whisper isn't penalized for
  casing/punctuation it produced correctly. A parity self-check confirms the normalizer
  leaves already-clean references unchanged.
- **Numbers — handled by the project verbalizer.** `num2words` has no Bulgarian support, so
  the harness expands Whisper's digits with **`bg_verbalize.normalize`** (the gender-aware
  verbalizer built for the TTS/Accentor front-end: два/две, currency/%/date/time/ordinal
  handlers, guarantees no digits remain), applied to hyp AND ref for parity. Residual: bare
  1/2 default to masculine ("2 реплики"→"два реплики" where the ref has "две") — a small subset,
  vs. the guaranteed error of a raw digit. The run prints a digit-token sanity count (~0).
  Optionally add a BG HF fine-tune as a second external row.

### Kaldi TDNN-F chain baseline (2nd system) — DEFERRED

Skipped for v1: the k2 build succeeded and the transducer is the best system, so a hybrid
HMM/TDNN-F won't beat 5.29% (expect it to land ~6–8%). The original "ship a number first /
k2-build contingency" rationale is spent. Build it only if publishing a formal benchmark or
if lattices/confidences are needed downstream.

- **Value if built:** an independent *second paradigm* (WFST + n-gram + phone lexicon + HMM)
  → publication credibility; a cross-check that validates the split / normalization /
  references (both systems landing in a sensible range ⇒ no leakage/normalization bug); a
  reference floor; word-level lattices + confidence scores + forced alignment.
- **Cost/risk:** LOW install risk — reuses `tri4b`, `train_cleaned` alignments, `data/lang`,
  the word KenLM, `~/kaldi` chainbin. Recipe scoped in [BG_ASR_PLAN.md](BG_ASR_PLAN.md)
  step 4 (new tree from `tri4b` → `chain-make-den-fst` → LF-MMI → `mkgraph` → decode).
  Multi-day effort; aligner dirs stay read-only.

### Other

- **CER alongside WER** — Bulgarian is inflectional (near-miss endings count as full word
  subs); CER is a useful companion metric, cheap from the existing `recogs-*` files.
- **sherpa-onnx int8 export** — BUILT 2026-08-08, measured: encoder **72.8 MB** + decoder
  0.5 MB + joiner 0.3 MB = **73.6 MB** int8 (fp32: 263.1 / 2.1 / 1.0 = 266.2 MB). The earlier
  "~66 MB" was arithmetic (65.5 M params × 1 byte), not a measurement — `export-onnx.py`
  quantizes `MatMul` only, so convolutions, biases and norm scales stay fp32.
  Op census: the encoder's 370 fp32 `MatMul` became 290 `MatMulInteger` + 80 `MatMul`; the 80
  survivors are exactly the attention activation×activation products (5 per layer × 16 layers),
  which dynamic quantization cannot touch — i.e. **100 % of eligible ops were quantized**, and
  the encoder contains no `Gemm` for the quantizer to silently skip.
- **`--causal 1` streaming variant** — separate run if word-level live streaming is needed.
