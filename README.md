# OmniVoice-BG

**Bulgarian text-to-speech that sounds native, stresses words correctly, and keeps
one consistent voice** — built on [OmniVoice](https://github.com/k2-fsa/OmniVoice)
and served over the LAN. It exists to voice the **TrainGen** training videos, but
works as a standalone Bulgarian TTS service.

> Fork of OmniVoice (Apache-2.0). The upstream model/readme is preserved in
> [`README.OmniVoice.md`](README.OmniVoice.md). This README documents the
> **Bulgarian layer** we added; the full operator's manual is
> **[`OMNIVOICE_BG.md`](OMNIVOICE_BG.md)**.

## What it adds

Stock OmniVoice speaks Bulgarian, but for narration three things were wrong. We fix
each with an independent lever:

| Problem out of the box | Fix | Where |
|---|---|---|
| A different voice/accent every clip | **Deterministic cloning** — fixed reference + greedy/seeded decoding | `omnivoice/cli/serve.py` |
| Wrong word **stress** (БУтона → буТОна) | **Stress front-end** — mark the stressed vowel before TTS, from the *rechko* dictionary | `bg_accent.py`, `build_bg_lexicon.py`, `bg_stress_add.py` |
| Non-native **accent** | **Fine-tuned model** on a Bulgarian audiobook | `models/omnivoice-bg/` (`prep_bg_hf.py`, `config/bg/`, `BG_FINETUNE_PLAN.md`) |

These are orthogonal — cloning fixes *identity*, the lexicon fixes *stress*, the
fine-tune fixes *nativeness* — and all three run together at inference.

## Architecture

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

## Documentation

| Doc | What's in it |
|---|---|
| **[OMNIVOICE_BG.md](OMNIVOICE_BG.md)** | Full operator's manual: architecture, every component, all workflows, config, troubleshooting, design rationale |
| [BG_FINETUNE_PLAN.md](BG_FINETUNE_PLAN.md) | The fine-tuning runbook (data → tokenize → train → evaluate → deploy) |
| [CLAUDE.md](CLAUDE.md) | Guidance for AI coding assistants working in this repo |
| [README.OmniVoice.md](README.OmniVoice.md) | The original upstream OmniVoice README |

## Repo layout (added on top of OmniVoice)

```
omnivoice/cli/serve.py     LAN TTS server (cloning, determinism, server-side accenting)
serve_bg.sh                production launcher
bg_accent.py               stress-marking front-end (lexicon + overrides, hot-reload)
build_bg_lexicon.py        rechko dump → stress lexicon (sqlite)
bg_stress_add.py           add/fix a word's stress (CAPS-vowel notation)
prep_bg_hf.py              ingest the HF Bulgarian dataset → training JSONL
eval_bg_ab.py              A/B listening harness across models
config/bg/                 fine-tune + data configs, stress overrides
```

Large artifacts — the fine-tuned model (`models/`), data (`data/`), and checkpoints
(`exp/`) — are gitignored; rebuild them with the scripts above and `BG_FINETUNE_PLAN.md`.

## Status

Internal project for Bulgarian narration in TrainGen. Not packaged for PyPI; run
from source on a CUDA GPU (developed on a 16 GB card).

## Attribution & license

Derivative of **OmniVoice** by k2-fsa, under **Apache-2.0** (see [`LICENSE`](LICENSE)).
The Bulgarian layer is our addition under the same license. Stress data derives from
the **rechko / chitanka_rechko** dictionary — check its terms before redistributing a
built `bg_stress.sqlite`.
