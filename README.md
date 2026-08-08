# OmniVoice-BG

**A Bulgarian speech stack: recognition and synthesis.** Started as a Bulgarian TTS layer on
[OmniVoice](https://github.com/k2-fsa/OmniVoice) to voice the **TrainGen** training videos, and
grew a full ASR half — a self-built 1,744 h corpus and a Zipformer transducer trained from
scratch on it.

> Fork of OmniVoice (Apache-2.0). The upstream model/readme is preserved in
> [`README.OmniVoice.md`](README.OmniVoice.md). This README documents the **Bulgarian layers**
> we added; the TTS operator's manual is **[`OMNIVOICE_BG.md`](OMNIVOICE_BG.md)**.

## Two tracks

| | What it does | Headline | Start here |
|---|---|---|---|
| **ASR** | Bulgarian speech → text. Own corpus (crawl → align/pseudo-label → filter), icefall/Zipformer training, LM + benchmark tooling | **4.12 % WER** on a frozen parliament holdout, from 1,744.8 h | [`docs/BG_ASR_BENCHMARK.md`](docs/BG_ASR_BENCHMARK.md) |
| **TTS** | Bulgarian text → speech. Native accent, correct lexical stress, one consistent voice, served over the LAN | Deterministic cloned narrator for TrainGen | [`OMNIVOICE_BG.md`](OMNIVOICE_BG.md) |

They share the Bulgarian text machinery — the stress lexicon, the number verbalizer
(`bg_verbalize.py`), the normalizer — which is why they live in one repo.

---

# ASR — Bulgarian speech recognition

A Zipformer transducer (65.5 M params, non-causal, 80-dim fbank @ 16 kHz, BPE-500) trained
**from scratch** on a corpus we built ourselves. Trained via [icefall](https://github.com/k2-fsa/icefall)
(a repo on `PYTHONPATH`, not a pip package — see [`asr/README.md`](asr/README.md)).

## Results

| Model | Corpus | Parliament dev | **Parliament test** |
|---|---|---|---|
| v1 (2026-08-03) | 649 h (parliament + audiobook) | 5.14 | 5.29 |
| **GP-v2 (2026-08-08)** | **1,744.8 h across 8 registers** | **3.94** | **4.12** |

Same architecture, tokenizer, schedule and frozen split in both — only the data changed, so
the 22 % relative gain is attributable to the corpus. Shipped decoder is
`modified_beam_search` (beam 4), **LM-free**: an external word 3-gram was v1's best system but
now loses on every set, even rebuilt on the GP corpus text
([why](docs/BG_ASR_BENCHMARK.md#the-external-lm-stopped-helping--and-the-rebuild-proved-why)).

**Read the caveats before quoting anything but the parliament number.** FLEURS/CV test WERs
(9.84 / 8.69) are *in-corpus* — we train on those corpora's train splits — so they measure
domain adaptation, not generalization, and the zero-shot Whisper comparison no longer holds
training exposure constant. A fresh self-recorded eval set is the next thing to build.

## Corpus

872,975 cuts / 1,744.8 h. Gold-aligned parliament + audiobook, the rest weak-labeled
(Whisper pseudo-labels, filtered, two-teacher agreement on the YouTube axes):

| Register | Source | Hours |
|---|---|---|
| Parliament (spontaneous) | plsten — own crawl + Kaldi alignment | 573.3 |
| Read literary | AB400 audiobook | 397.5 |
| Conversational | DnevenRed, Govori Internet (YouTube) | 506.7 |
| Broadcast news | Bloomberg BG (YouTube) | 135.1 |
| Archival radio | BNR archives | 102.9 |
| Read modern | FLEURS + Common Voice (train splits) | 16.5 |

Live view: [`docs/BG_ASR_Corpus.ipynb`](docs/BG_ASR_Corpus.ipynb) (re-runnable — reads the
manifests and decode logs off disk). Method: [`docs/BG_ASR_DATA_PIPELINE.md`](docs/BG_ASR_DATA_PIPELINE.md).

## Pipeline

```bash
# 1. Harvest + label (per axis; all print per-unit progress and resume)
./venv/bin/python prep_bg_parliament.py          # parliament sessions + stenograms
./venv/bin/python bnr_crawl.py && ./venv/bin/python bnr_pseudolabel.py
./venv/bin/python yt_harvest.py --channel <id> --limit N
./venv/bin/python prep_fleurs_bg.py --split train
./venv/bin/python prep_commonvoice_bg.py --split train

# 2. Split (whole-session, leakage-safe) → cuts → features → BPE
./venv/bin/python make_asr_split.py
./venv/bin/python prep_lhotse_manifests.py
./venv/bin/python compute_fbank_bg.py
./venv/bin/python train_bpe_bg.py

# 3. Merge the axes into one training manifest (globally shuffled by default;
#    --holdout enforces the leakage guard on recording-ids AND cut-ids)
./venv/bin/python merge_train_corpus.py \
  --train data/fbank/bg_cuts_train.jsonl.gz  --train data/fbank/bg_cuts_bnr.jsonl.gz \
  --train data/fbank/bg_cuts_ytdr.jsonl.gz   --train data/fbank/bg_cuts_ytgi.jsonl.gz \
  --train data/fbank/bg_cuts_ytbb.jsonl.gz   --train data/fbank/bg_cuts_fleurs_train.jsonl.gz \
  --train data/fbank/bg_cuts_cv_train.jsonl.gz \
  --holdout data/fbank/bg_cuts_dev.jsonl.gz  --holdout data/fbank/bg_cuts_test.jsonl.gz \
  --holdout data/fbank/bg_cuts_fleurs.jsonl.gz --holdout data/fbank/bg_cuts_cv.jsonl.gz \
  --out data/fbank/bg_cuts_train_gp_shuf.jsonl.gz

# 4. Train (power-cut resilient; resumes from the last epoch-N.pt)
EXP=zipformer/exp_gp ENABLE_MUSAN=true \
  TRAIN_MANIFEST=bg_cuts_train_gp_shuf.jsonl.gz bash asr/train_resumable.sh

# 5. Decode (test is decoded ONCE, after picking the system on dev)
cd ~/icefall/egs/bg_parliament/ASR && python zipformer/decode.py \
  --epoch 30 --avg 5 --use-averaged-model 1 --exp-dir zipformer/exp_gp \
  --bpe-model data/lang_bpe_500/bpe.model --decoding-method modified_beam_search \
  --beam-size 4 --eval-set dev
```

Optional n-gram LM (built and measured, not shipped): `prep_lm_text.py` →
`build_kenlm_bg.sh` → `compile_lg_bg.sh`. `TAG` selects a corpus variant so builds never
clobber each other; `LEAK_CHECK=ids` asserts zero cut/recording-id overlap with every eval
manifest.

---

# TTS — Bulgarian text-to-speech

**Sounds native, stresses words correctly, keeps one consistent voice** — served over the LAN.
Stock OmniVoice speaks Bulgarian, but for narration three things were wrong. Each gets an
independent lever:

| Problem out of the box | Fix | Where |
|---|---|---|
| A different voice/accent every clip | **Deterministic cloning** — fixed reference + greedy/seeded decoding | `omnivoice/cli/serve.py` |
| Wrong word **stress** (БУтона → буТОна) | **Stress front-end** — mark the stressed vowel before TTS, from the *rechko* dictionary | `bg_accent.py`, `build_bg_lexicon.py`, `bg_stress_add.py` |
| Non-native **accent** | **Fine-tuned model** on a Bulgarian audiobook | `models/omnivoice-bg/` (`prep_bg_hf.py`, `config/bg/`, `BG_FINETUNE_PLAN.md`) |

These are orthogonal — cloning fixes *identity*, the lexicon fixes *stress*, the
fine-tune fixes *nativeness* — and all three run together at inference.

```
  TrainGen (no GPU)                         GPU machine (this repo)
  ─────────────────                         ───────────────────────
  RemoteOmniVoiceBackend  ── POST /synth ─▶ omnivoice/cli/serve.py
   • plain Bulgarian text                    ├─ BgAccenter: text → stress-marked text
                          ◀── WAV 24 kHz ──  ├─ fixed-voice cloning + deterministic decode
                                             └─ fine-tuned BG model (models/omnivoice-bg)
```

## Quickstart

On the GPU machine (Python env in `venv/`):

```bash
# 1. Build the stress lexicon once (from the rechko dictionary dump)
./venv/bin/python build_bg_lexicon.py --dump ../rechko/db.sql --out data/bg_lexicon/bg_stress.sqlite

# 2. Launch the server (fine-tuned model + stress accenting + cloned narrator)
OMNIVOICE_BG_REF_AUDIO=data/bg_ref/narrator.wav \
OMNIVOICE_BG_REF_TEXT="$(cat data/bg_ref/narrator.txt)" \
bash serve_bg.sh

# 3. Synthesize
curl -s -X POST http://localhost:8002/synth -H 'Content-Type: application/json' \
  -d '{"text":"Натиснете бутона, за да отворите менюто."}' --output out.wav
```

Fix a mispronounced word any time (stressed vowel in CAPS; hot-reloads, no restart):

```bash
./venv/bin/python bg_stress_add.py думАта портАла
```

---

## Documentation

**ASR**

| Doc | What's in it |
|---|---|
| [docs/BG_ASR_BENCHMARK.md](docs/BG_ASR_BENCHMARK.md) | Results (v1 + GP-v2), the LM experiment, caveats, reproducibility |
| [docs/BG_ASR_ROADMAP.md](docs/BG_ASR_ROADMAP.md) | Sequenced next actions; what GP-v2 resolved and what now blocks |
| [docs/BG_ASR_DATA_PIPELINE.md](docs/BG_ASR_DATA_PIPELINE.md) | Preprocessing per axis, quality gates, register balance, mux lever |
| [docs/BG_ASR_Corpus.ipynb](docs/BG_ASR_Corpus.ipynb) | Live corpus composition + results (re-runnable) |
| [docs/BG_ASR_PLAN.md](docs/BG_ASR_PLAN.md) · [BG_ASR_V2_PLAN.md](docs/BG_ASR_V2_PLAN.md) · [BG_ASR_BNR_PLAN.md](docs/BG_ASR_BNR_PLAN.md) | v1 recipe/inventory · corpus recovery + LM stack · BNR axis |
| [docs/BG_PARLIAMENT_INGESTION.md](docs/BG_PARLIAMENT_INGESTION.md) | The parliament crawl and alignment in detail |
| [asr/README.md](asr/README.md) | icefall scaffolding + the recipe's own commands |

**TTS**

| Doc | What's in it |
|---|---|
| [OMNIVOICE_BG.md](OMNIVOICE_BG.md) | Full operator's manual: architecture, components, workflows, config, troubleshooting |
| [BG_FINETUNE_PLAN.md](BG_FINETUNE_PLAN.md) | Fine-tuning runbook (data → tokenize → train → evaluate → deploy) |
| [docs/BG_VOICE_CLONING.md](docs/BG_VOICE_CLONING.md) · [docs/BG_LESSONS.md](docs/BG_LESSONS.md) | Cloning behavior · accumulated lessons |
| [CLAUDE.md](CLAUDE.md) | Guidance for AI coding assistants working in this repo |
| [README.OmniVoice.md](README.OmniVoice.md) | The original upstream OmniVoice README |

## Repo layout (added on top of OmniVoice)

```
ASR
  asr/                       icefall scaffolding, datamodule, resumable train launcher
  prep_bg_parliament.py      parliament crawl (sessions + stenograms)
  bnr_crawl.py               BNR archive crawl
  bnr_pseudolabel.py         Whisper pseudo-labeling + filtering (long-file safe)
  yt_harvest.py              YouTube axis harvest (per-channel, capped)
  prep_fleurs_bg.py          FLEURS bg → cuts + fbank
  prep_commonvoice_bg.py     Common Voice bg → cuts + fbank
  make_asr_split.py          whole-session train/dev/test split (leakage-safe)
  prep_lhotse_manifests.py   → data/lhotse/bg/cuts_*.jsonl.gz
  compute_fbank_bg.py        → data/fbank/bg_cuts_*.jsonl.gz
  train_bpe_bg.py            → data/lang_bpe_500/
  merge_train_corpus.py      merge axes → one shuffled training manifest
  prep_lm_text.py            LM text from Kaldi dirs or lhotse cuts (leakage guards)
  build_kenlm_bg.sh          word n-gram (KenLM), TAG-versioned
  compile_lg_bg.sh           ARPA → LG.pt decoding graph
  eval_whisper_bg.py         Whisper baseline through the shared normalizer
  audit_pseudolabels.py      label-quality audit / calibration
  repair_flac_headers.py     fix broken FLAC containers before lhotse prep
  musan_prep.sh              MUSAN noise manifests for CutMix augmentation

TTS
  omnivoice/cli/serve.py     LAN TTS server (cloning, determinism, server-side accenting)
  serve_bg.sh                production launcher
  bg_accent.py               stress-marking front-end (lexicon + overrides, hot-reload)
  build_bg_lexicon.py        rechko dump → stress lexicon (sqlite)
  bg_stress_add.py           add/fix a word's stress (CAPS-vowel notation)
  prep_bg_hf.py              ingest the HF Bulgarian dataset → training JSONL
  eval_bg_ab.py              A/B listening harness across models
  config/bg/                 fine-tune + data configs, stress overrides

Shared
  bg_verbalize.py            number/date/currency verbalizer (gender-aware)
  bg_text_norm.py            text normalization shared by ASR scoring and TTS
```

Large artifacts — models (`models/`), data and corpora (`data/`), checkpoints (`exp/`) — are
gitignored. Rebuild them with the pipelines above.

## Status

Internal project. Not packaged for PyPI; run from source on a CUDA GPU (developed on a 16 GB
card). ASR training goes through icefall on `PYTHONPATH`; TTS runs from this repo directly.

## Attribution & license

Derivative of **OmniVoice** by k2-fsa, under **Apache-2.0** (see [`LICENSE`](LICENSE)). The
Bulgarian layers are our addition under the same license.

The ASR side builds on **icefall**, **k2** and **lhotse** (k2-fsa, Apache-2.0) and follows the
Finnish Parliament ASR paper (`2203.14876`) as its reference recipe; the Kaldi pilot uses
**Kaldi** (Apache-2.0). Stress data derives from the **rechko / chitanka_rechko** dictionary —
check its terms before redistributing a built `bg_stress.sqlite`. Corpus sources (parliament
stenograms, BNR archives, YouTube channels, FLEURS, Common Voice) carry their own terms; the
built corpora are not redistributed here.
