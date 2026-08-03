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
- **In-domain only.** This measures unseen *parliament* speech, not conversational/dialectal
  or noisy audio (distribution shift not evaluated here).

## Reproducibility

- LM text (dev/test-excluded): `prep_lm_text.py` → `build_kenlm_bg.sh` (leakage assertion).
- LG graph: `compile_lg_bg.sh` → `data/lang_bpe_500/LG.pt`.
- Decode: `zipformer/decode.py --epoch 30 --avg 5 --use-averaged-model 1 --eval-set {dev,test}
  --decoding-method <m> [--ngram-lm-scale 0.2]` (the `--eval-set` switch is a BG addition).
- Env: torch 2.8.0+cu128, k2 1.24.4 (sm_120), lhotse 1.33.0.

## Next

- **v2**: recover the 110 quarantined sessions (+~257 h, train only), self-clean in-bounds
  desync with this baseline, retrain, decode the **same** frozen test → v1↔v2 delta.
- Optional: Kaldi TDNN-F chain baseline (second system), Whisper comparison row, sherpa-onnx
  int8 export for edge deployment, CER alongside WER.
