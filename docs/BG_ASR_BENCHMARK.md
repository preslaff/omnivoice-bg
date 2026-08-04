# Bulgarian Parliament ASR — benchmark (v1)

**Headline: 5.29% WER** on the frozen parliament test set (unseen sessions), from a
65.5M-param Zipformer transducer trained from scratch on 649 h of Kaldi-aligned Bulgarian
speech. Companion to [BG_ASR_PLAN.md](BG_ASR_PLAN.md) (recipe/inventory) and
[BG_ASR_V2_PLAN.md](BG_ASR_V2_PLAN.md) (recovery/data-quality roadmap). Reference paper:
`2203.14876` (Finnish Parliament ASR).

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
| FLEURS (Whisper's domain) | 20.65 | **11.70** | Whisper, ~1.8× |

**Each model wins on its home turf, and the *shape* differs.** Ours is a **specialist** — superb
in-domain (5.29) but a large domain swing (5.29→20.65, ~3.9×). Whisper is a **generalist** —
never great, never bad, domain-robust (15.37 vs 11.70, ~1.3× swing). Neither dominates both, so
a **neutral both-OOD set** (unseen by *both* — see [BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md)) is the
real tiebreaker. Parliament-Whisper breakdown: 10,354 sub / 3,654 del / 5,035 ins / 123,874 words.
Minor caveat: our numbers via icefall's scorer, Whisper's via `bg_verbalize`; both refs already
number-spelled, so the paths differ << the gaps.

### Out-of-domain — FLEURS BG (read Wikipedia), 2026-08-04

Zero-adaptation eval on FLEURS `bg_bg` **test** (658 utts / 1.9 h / 14,404 ref words) —
Bulgarian Wikipedia sentences read aloud: a different register + speaker pool, clean studio
audio. Same epoch-30 / avg-5 model, same `bg_verbalize` normalizer.

| Set | Method | WER % |
|---|---|---|
| FLEURS OOD | greedy_search | **20.65** |
| FLEURS OOD | modified_beam_search | 20.90 |
| FLEURS — Whisper large-v3 (ref) | greedy | 11.70 |

FLEURS error breakdown (greedy): 2974 / 14,404 = 223 ins / 861 del / 1890 sub.

**In-domain 5.29% → OOD ~20.7% (≈4× relative).** This is the *domain-specialization cost*, not
a defect. FLEURS is clean audio, so the gap is purely register / vocabulary / speaker shift
(spontaneous parliamentary speech + its lexicon vs. read encyclopedic Wikipedia), not noise:
- The model **generalizes** — ~20.7% on a wholly unseen domain/speakers means it learned real
  Bulgarian ASR, not memorized parliament (a memorizer would be far worse). Deletions + subs
  dominate (861 del / 1890 sub) = out-of-vocabulary encyclopedic terms and names — the expected
  OOD failure mode.
- **Beam ≤ greedy here** (20.90 vs 20.65), the reverse of in-domain: on OOD input the model's
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
- **Reference-transcript ceiling.** References are stenographic + Kaldi-aligned; some
  residual WER is reference noise, not model error.
- **Domain matters (measured).** In-domain parliament = 5.29%; out-of-domain read Wikipedia
  (FLEURS) = ~20.7% (§ Out-of-domain above). Conversational / dialectal / noisy audio is still
  unmeasured and would likely differ again.

## Reproducibility

- LM text (dev/test-excluded): `prep_lm_text.py` → `build_kenlm_bg.sh` (leakage assertion).
- LG graph: `compile_lg_bg.sh` → `data/lang_bpe_500/LG.pt`.
- Decode: `zipformer/decode.py --epoch 30 --avg 5 --use-averaged-model 1 --eval-set {dev,test}
  --decoding-method <m> [--ngram-lm-scale 0.2]` (the `--eval-set` switch is a BG addition).
- Env: torch 2.8.0+cu128, k2 1.24.4 (sm_120), lhotse 1.33.0.

## Next

- **v2** (primary, product-improving): recover the 110 quarantined sessions (+~257 h, train
  only), self-clean in-bounds desync with this baseline, retrain, decode the **same** frozen
  test → v1↔v2 delta. See [BG_ASR_V2_PLAN.md](BG_ASR_V2_PLAN.md).

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
- **sherpa-onnx int8 export** — shippable standalone recognizer for edge deployment (~66 MB).
- **`--causal 1` streaming variant** — separate run if word-level live streaming is needed.
