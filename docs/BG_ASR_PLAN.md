# Bulgarian ASR byproduct — plan (icefall/Zipformer)

**Status:** approved 2026-07-30. Byproduct of the OmniVoice-BG parliament pipeline,
opened after the Accentor stress-marking track shipped (ft7d). Companion docs:
[BG_FINAL_EFFORT_PLAN.md](BG_FINAL_EFFORT_PLAN.md) (§ "Recognized byproduct" lines
64-101 — original scope), [BG_PARLIAMENT_INGESTION.md](BG_PARLIAMENT_INGESTION.md)
(corpus stats, Finnish-paper reuse map), [ACCENTOR_REFRESH_PLAN.md](ACCENTOR_REFRESH_PLAN.md).
Reference paper: `2203.14876v1.pdf` (Finnish Parliament ASR) at repo root.

## Context

Force-aligning the parliament + audiobook corpus for stress-label harvesting
produced, for free, two of the three parts of a recognizer — a large
aligned+cleaned speech corpus and a pronunciation lexicon. Only an n-gram LM +
decoding graph are missing. The Accentor track is done, so this byproduct is now
in scope. **Goal:** a shippable BG ASR model AND a research artifact / WER
benchmark (mirroring `2203.14876` and BG-PARLAMA, Geneva 2019). **Data:** everything
(~998 h incl. 30 newly-crawled sessions). **Toolkit:** icefall/Zipformer.

## Inventory (verified 2026-07-30)

| Asset | Location | Size |
|---|---|---|
| Aligned+cleaned train set (parliament+AB400) | `exp/kaldi_pilot/data/train_cleaned{,_hires}` + ali `tri4b_ali_cleaned` | **740 h / 480,947 utts** (AB400 397.5 h + parl 342.6 h) |
| Incremental parliament (2026-07-28) | `exp/kaldi_pilot/data/parl_new_{cleaned,hires}` + ali `tri4b_ali_new_seg` | **228 h / 166,551 utts** (72 sessions) |
| Newly crawled — NOT yet segmented | 30 sessions in `data/bg_parliament/manifest.jsonl` (`plsten_11143/10953/10948…`) | **~51 h raw / ~30 h effective** |
| **Total after segmenting the new 30** | — | **≈ 998 h effective** |
| Source audio | `data/bg_parliament/audio/*.flac` (24 kHz mono, 93 GB) · `data/bg400/clips/*.wav` (24 kHz mono, 74 GB) | 221 sessions + 195,747 clips |
| Lexicon + lang | `data/bg_lexicon/lexicon_full.txt` (2.5 M) → `exp/kaldi_pilot/data/lang/L.fst` (34-phone set) | reused only for baseline/HLG |
| LM text | `data/bg_parliament/transcripts/*.txt` (cased) + `data/chitanka/` (11 GB) | — |
| Kaldi build | `~/kaldi` (cuda-13-support, sm_120): chainbin + `nnet3-latgen-faster` | — |

NOT installed: k2 / lhotse / icefall (venv = torch 2.8+cu128, torchaudio). Existing
Kaldi models are **aligners only** and are the untouchable stress-label factory.
In-domain ASR saturates ≈ 1800 h — we're on the steep part.

## Recommendation

**Deliverable = Zipformer in icefall/k2** (pruned-transducer + auxiliary CTC head,
BPE ~500 from transcripts+chitanka; the E2E model needs no phone lexicon). Best WER
at this scale, k2-fsa coherence with OmniVoice, publishable, sherpa-exportable.

**Kaldi TDNN-F chain baseline FIRST** (fast WER number): reuses `tri4b` +
`train_cleaned` alignments + `data/lang` + `~/kaldi` chainbin/decode — zero new
install, zero sm_120 risk, days. De-risks split+LM, ships the first number, gives
the paper a second system. New `exp/` subtree; aligner untouched. Whisper kept as an
external comparison row; wav2vec2/SSL rejected.

## Plan

**0. Persist plan + record Accentor progress** — this doc + the ft7d "marking done /
ceiling acoustic" end-to-end result (`ACCENTOR_REFRESH_PLAN.md`, paper §6.5, memory).

**1. Segment the 30 newly-crawled sessions** (~51 h → ~30 h). Run the fixed
`harvest_new_sessions.sh` (stage-0 diff auto-selects them). ASR needs only its
**stage 1–2 output** — `parl_new2_cleaned` (`segments`/`text`/`wav.scp`); stages 3–4
stress-harvest optional (grows the Accentor corpus too). Fresh/date-disjoint ⇒ good
held-out TEST candidates.

**2. Freeze the held-out split** (shared by both systems). By **whole session**
(key from `segments` rec-id) — no leakage. TEST = ~10–20 whole parliament sessions
(~5–10 h; the new 30 are natural) + a few whole audiobooks, **reported separately**.
DEV = disjoint ~3–5 h. Freeze id lists at `data/lhotse/bg/split/{train,dev,test}_recordings.txt`.

**3. Language model (KenLM).** 3/4-gram word LM from parliament `transcripts/`
(up-weighted) + a chitanka slice. **Strip U+0301 + normalize identically to the ASR
text; exclude TEST transcripts.** → `G.fst` (baseline) + shallow-fusion/rescoring
(icefall). Lexicon reused.

**4. Kaldi TDNN-F chain baseline** (new `exp/kaldi_pilot/exp/chain_bg/`). New tree
from `tri4b` on `train_cleaned` (NOT `tdnn_a`) → `chain-make-den-fst` →
`steps/nnet3/chain/train.py` (LF-MMI) → `mkgraph` w/ §3 LM → `nnet3-latgen-faster`
decode → `compute-wer`. Aligner dirs read-only.

**5. Install k2 stack** (start early — the risk gate). k2 → lhotse → icefall →
kenlm → optional sherpa-onnx. **Main risk: no prebuilt k2 wheel for sm_120 / torch
2.8 → source build** (`TORCH_CUDA_ARCH_LIST="12.0"`, CUDA 12.8 nvcc). CTC smoke test
before full prep. cuda-13 Kaldi build is irrelevant to icefall (links venv torch).

**6. Corpus → lhotse manifests.** `lhotse kaldi import` on `train_cleaned` +
`parl_new_cleaned` + `parl_new2_cleaned` → recordings (24 kHz FLAC/WAV, **resample
16 kHz**) + supervisions from `text`. `trim_to_supervisions` → CutSet → 80-dim
fbank. **Strip stress marks from supervision text** (must not leak into ASR
targets); remove punctuation. **Circularity guard:** human stenograms only, never
Accentor-marked/synthetic. BPE ~500 in `data/lang_bpe_500/`.

**7. Train the Zipformer** (new `egs/bg_parliament/ASR/`). Zipformer-M (~65 M) +
`--use-ctc 1`; 80-dim fbank; `--max-duration 150`; `--use-fp16 1`; ScaledAdam,
~30 epochs, checkpoint-average last ~8, dev-WER selection. ~2–4 h/epoch → **~3–5
days** on the 16 GB card (Zipformer-S / CTC-only fallback if tight).

**8. Decode + benchmark.** WER + CER on TEST, **parliament and audiobook
separately**, LM ladder greedy → beam → +n-gram fusion → +neural-LM rescore.
Kaldi-chain + Whisper comparison rows. Finnish-paper table shape; position vs
BG-PARLAMA (249 h); state ~1800 h saturation. Optional 400 h-subset ablation
(corpus-size/WER slope). → **`docs/BG_ASR_BENCHMARK.md`**. Optional sherpa-onnx
export = shippable recognizer.

## Sequencing & effort
Step 0 → segment new 30 → freeze split → LM → **Kaldi baseline (~2–4 days, first
number)** ∥ **k2 source build early** → manifests + BPE → **Zipformer (~3–5 days)**
→ benchmark. icefall deliverable ≈ **1–2 weeks** wall incl. the k2-build risk. All
big jobs user-run.

## Risks
- **k2 source build on sm_120 + torch 2.8/cu128 (highest)** — start early; ship the
  Kaldi baseline first as contingency.
- ~998 h < 1800 h saturation — report WER as a slope point, not converged SOTA.
- **Do-not-disturb** — ASR writes to NEW dirs only (`chain_bg/`, `parl_new2_*`,
  `egs/bg_parliament/`, `data/lhotse/bg/`, `data/lang_bpe_500/`); `tri4b`/`tdnn_a`/
  `data/lang` read-only.
- **Text normalization** — one normalization (stress-stripped, no-punct) across BPE,
  LM, and Kaldi text.
- 16 GB VRAM — fp16/bf16, max-duration tuning, Zipformer-S fallback.
- LM/BPE leakage — exclude TEST transcripts.

## Verification
- New 30 segmented: `parl_new2_cleaned/{segments,text}` non-empty (~30 h), `text` not `<unk>`.
- Install: k2 CTC smoke test passes (arch 12.0 compiled in).
- Manifests: CutSet ≈ 670 k cuts (~998 h); spot-check 3 cuts align.
- Split: zero session-id overlap train↔test.
- Baseline: Kaldi-chain WER (floor to beat).
- Model: dev WER decreases; final TEST WER + CER (parliament & audiobook separately),
  LM-rescored vs greedy → `docs/BG_ASR_BENCHMARK.md`.

## Deliverables
This doc · reproducible Kaldi→lhotse prep script · `parl_new2_cleaned` (+30 h) ·
trained Zipformer + decode/WER · Kaldi-chain baseline WER · `docs/BG_ASR_BENCHMARK.md`
· optional sherpa-onnx standalone recognizer.
