# OmniVoice-BG

A Bulgarian text-to-speech system built on [OmniVoice](https://github.com/k2-fsa/OmniVoice)
(Apache-2.0). It produces **natural, correctly-stressed** Bulgarian narration in a
**consistent, deterministic** voice, served over the LAN. Built to voice the
**TrainGen** training videos, but usable as a standalone Bulgarian TTS service.

> This is an internal runbook for the cloned OmniVoice repo plus our additions.
> Upstream OmniVoice docs live in `README.OmniVoice.md` / `docs/`; this file
> documents the **Bulgarian layer** we added and how to operate the whole thing.

---

## Contents
- [Why it exists](#why-it-exists)
- [The three pillars](#the-three-pillars)
- [System architecture](#system-architecture)
- [Quickstart: run the production server](#quickstart-run-the-production-server)
- [Using it from TrainGen](#using-it-from-traingen)
- [Everyday task: fix a mispronounced word](#everyday-task-fix-a-mispronounced-word)
- [Component reference (file by file)](#component-reference-file-by-file)
- [Rebuilding artifacts](#rebuilding-artifacts)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Design decisions (why it's built this way)](#design-decisions-why-its-built-this-way)
- [Attribution & license](#attribution--license)

---

## Why it exists

Stock OmniVoice can *speak* Bulgarian (it's in its 600+ languages), but for
narration three things were wrong out of the box:

1. **Every clip used a different voice/accent** (voice design is non-deterministic).
2. **Wrong word stress** — e.g. *БУтона* instead of *буТОна*, *натИснете* instead of
   *натисНЕте*. Bulgarian stress is lexical, mobile, and unmarked in text, so the
   model can't reliably infer it.
3. **A non-native "foreign" accent** on the synthesized Bulgarian.

OmniVoice-BG fixes all three.

## The three pillars

| Problem | Fix | Where |
|---|---|---|
| Voice drifts every clip | **Deterministic cloning** — fixed reference voice + greedy/seeded decoding | `omnivoice/cli/serve.py` |
| Wrong lexical **stress** | **Stress front-end** — mark the stressed vowel (U+0301) before TTS, from the *rechko* dictionary | `bg_accent.py`, `build_bg_lexicon.py`, `bg_stress_add.py`, `config/bg/bg_stress_overrides.json` |
| Non-native **accent** | **Fine-tuned model** on a Bulgarian audiobook | `models/omnivoice-bg/` (+ `prep_bg_hf.py`, `config/bg/`, `BG_FINETUNE_PLAN.md`) |

These are **orthogonal** — each fixes a different failure, all three are needed.

## System architecture

```
  TrainGen machine (no GPU)                 GPU machine (OmniVoice-BG)
  ─────────────────────────                 ──────────────────────────
  traingen/tts.py                           omnivoice/cli/serve.py  (FastAPI, :8002)
   RemoteOmniVoiceBackend                      │
   • plain Bulgarian text  ── POST /synth ──▶  ├─ bg_accent.BgAccenter
   • per-voice reference                       │    plain text → stress-marked text
     (assets/voices/<voice>.wav+.txt)          │    (rechko lexicon + overrides)
                                               ├─ fixed-voice cloning (reference)
                              WAV (24 kHz) ◀──  ├─ deterministic decode (seed, greedy)
                                               └─ OmniVoice model: models/omnivoice-bg
                                                    (fine-tuned BG) + Higgs audio tokenizer
```

The whole Bulgarian intelligence (model + lexicon) lives on the **GPU machine**;
TrainGen just sends plain text and gets back audio.

---

## Quickstart: run the production server

On the GPU machine (LAN IP e.g. `192.168.78.101`), with any old server stopped
(`pkill -f omnivoice.cli.serve`), use the launch script:

```bash
cd ~/WORK/OmniVoice
bash serve_bg.sh
```

`serve_bg.sh` wraps the full command with sensible defaults (model
`models/omnivoice-bg`, the lexicon, overrides, deterministic seed, the
`expandable_segments` env). Override any setting via env var, e.g.
`OMNIVOICE_BG_PORT=9000 bash serve_bg.sh`, or set a server-wide fixed voice with
`OMNIVOICE_BG_REF_AUDIO=narrator.wav OMNIVOICE_BG_REF_TEXT="<transcript>" bash serve_bg.sh`.

<details><summary>Equivalent raw command</summary>

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./venv/bin/python -m omnivoice.cli.serve \
    --model    models/omnivoice-bg \
    --ref-audio  <narrator.wav> --ref-text "<exact transcript of narrator.wav>" \
    --lexicon  data/bg_lexicon/bg_stress.sqlite \
    --overrides config/bg/bg_stress_overrides.json \
    --host 0.0.0.0 --port 8002 --seed 42
```
</details>

Check it: `curl http://192.168.78.101:8002/health` →
`{"status":"ok","sampling_rate":24000,"fixed_voice":true,"bg_accent":true}`

`POST /synth` with `{"text": "...", ...}` returns a 24 kHz mono WAV. Text is
accented server-side, spoken in the cloned voice, deterministically.

> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is optional for inference
> (it was load-bearing for *training* on the 16 GB card); harmless to keep.

---

## Using it from TrainGen

In TrainGen's `.env` (on the TrainGen machine):

```bash
TRAINGEN_TTS=omnivoice_remote
TRAINGEN_OMNIVOICE_URL=http://192.168.78.101:8002   # GPU machine IP + port
TRAINGEN_VOICE_DIR=assets/voices                     # per-voice reference clips
TRAINGEN_OMNIVOICE_AUTOVOICE=0                        # 1 = use the model's default voice (no cloning)
```

**Voice modes** from the client:
- `AUTOVOICE=1` → sends only text → the model's **default voice** (the fine-tuned narrator). Simplest; deterministic via the server seed.
- `AUTOVOICE=0` + per-voice files present → **cloning** (tightest consistency).
- `AUTOVOICE=0` + no files → **instruct** voice design (non-deterministic voice).

Per-voice references (optional, for client-chosen voices): drop
`assets/voices/<voice>.wav` + `<voice>.txt` (exact transcript). If present,
`RemoteOmniVoiceBackend` sends them for cloning; otherwise it falls back to
`instruct` voice design. See `assets/voices/README.md`.

> Note: if the server is launched with a fixed `--ref-audio`, that voice is used
> for *every* request regardless of what the client sends. Pick one approach:
> server-fixed voice **or** client per-voice references.

---

## Everyday task: fix a mispronounced word

When you hear a word stressed wrong, add it with the **stressed vowel in CAPS**
(Cyrillic). No restart needed — the server hot-reloads overrides.

```bash
cd ~/WORK/OmniVoice
./venv/bin/python bg_stress_add.py думАта портАла полЕто
```

- Writes to `config/bg/bg_stress_overrides.json` (you never type the invisible
  combining accent).
- **Overrides beat everything** (lexicon, projection, imperative rule), so this
  fixes any polysyllabic word, including homographs you want pinned.
- Monosyllables are skipped (one syllable → no stress ambiguity).
- Use **Cyrillic** capitals (`А`, not Latin `A`).

---

## Component reference (file by file)

### OmniVoice-BG repo (`~/WORK/OmniVoice`)

| Path | Purpose |
|---|---|
| `omnivoice/cli/serve.py` | LAN HTTP TTS server. Fixed-voice cloning, deterministic decoding (`--seed`, `position/class_temperature=0`), server-side BG accenting (`--lexicon`/`--overrides`), per-request reference caching, OOM handling. |
| `bg_accent.py` | `BgAccenter` — turns plain Cyrillic text into stress-marked text. Lookup order: overrides → rechko lexicon → leave unmarked. Thread-safe, hot-reloads overrides, skips monosyllables. |
| `build_bg_lexicon.py` | Parses the **rechko** SQL dump → `bg_stress.sqlite`. Phase 1: ~115k lemmas (backtick→U+0301). Phase 2: ~2.5M inflected forms projected from lemma stress by vowel ordinal, with a penult-vowel rule for `-ете/-йте` imperatives. |
| `bg_stress_add.py` | CLI to add/fix words in the overrides file using CAPS-vowel notation. |
| `config/bg/bg_stress_overrides.json` | Manual stress overrides (highest priority). Hot-reloaded by the server. |
| `prep_bg_hf.py` | Ingests the HF dataset `raditotev/bg-audiobooks-tts` → WAV clips + `train.jsonl`/`dev.jsonl`. |
| `config/bg/train_config_bg.json` | Fine-tune config tuned for one 16 GB GPU (SDPA, `batch_tokens=1024`, `grad_accum=8`, init from `k2-fsa/OmniVoice`). |
| `config/bg/data_config_bg.json` | Points training at the tokenized WebDataset shards. |
| `eval_bg_ab.py` | A/B listening harness — synthesizes the same sentences through multiple models with shared reference + optional `--accent`, for by-ear comparison. |
| `BG_FINETUNE_PLAN.md` | The end-to-end fine-tuning runbook (data → tokenize → train → evaluate → deploy). |
| `models/omnivoice-bg/` | The deployed fine-tuned model (slimmed checkpoint-2750, ~2.3 GB). *gitignored.* Published privately to HF: [`preslaff/omnivoice-bg`](https://huggingface.co/preslaff/omnivoice-bg) (weights + model card + TensorBoard metrics). |
| `data/bg_lexicon/bg_stress.sqlite` | The stress lexicon (~2M forms, ~380 MB). *gitignored.* |
| `exp/bg_keep/` | Preserved training checkpoints (250, 1250, 2750). *gitignored.* |

### TrainGen repo (`~/WORK/TrainGen`)

| Path | Purpose |
|---|---|
| `traingen/tts.py` → `RemoteOmniVoiceBackend` | LAN client: POSTs plain text to the server, writes the returned WAV. Resolves per-voice references from `TRAINGEN_VOICE_DIR`. Selected via `TRAINGEN_TTS=omnivoice_remote`. |
| `assets/voices/` | Per-voice reference clips (`<voice>.wav` + `<voice>.txt`) and README. |
| `.env.example` | Documents `TRAINGEN_TTS`, `TRAINGEN_OMNIVOICE_URL`, `TRAINGEN_VOICE_DIR`. |

### External input

| Path | Purpose |
|---|---|
| `~/WORK/rechko/db.sql` | The rechko (chitanka_rechko) dictionary MySQL dump — source of authoritative Bulgarian lemma stress. Not part of the repo. |

---

## Rebuilding artifacts

**Stress lexicon** (after a new rechko dump, or changing the build logic):
```bash
./venv/bin/python build_bg_lexicon.py --dump ../rechko/db.sql --out data/bg_lexicon/bg_stress.sqlite
```

**Fine-tuned model** — see `BG_FINETUNE_PLAN.md`. In brief:
```bash
./venv/bin/python prep_bg_hf.py --out data/bg            # data → JSONL
# tokenize train+dev to WebDataset (plan §4)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./venv/bin/accelerate launch \
    --gpu_ids 0 --num_processes 1 -m omnivoice.cli.train \
    --train_config config/bg/train_config_bg.json \
    --data_config config/bg/data_config_bg.json --output_dir exp/omnivoice_bg
# pick best checkpoint by ear (eval_bg_ab.py), slim it into models/omnivoice-bg/
```

**A/B compare models** (e.g. stock vs fine-tuned, with production accenting):
```bash
./venv/bin/python eval_bg_ab.py \
    --models k2-fsa/OmniVoice=stock models/omnivoice-bg=bg \
    --ref-from-dataset --accent --out eval_ab
```

---

## Configuration reference

**Server flags** (`omnivoice/cli/serve.py`):
`--model`, `--ref-audio`/`--ref-text`, `--lexicon`, `--overrides`, `--seed`,
`--num-step`, `--guidance-scale`, `--position-temperature`, `--class-temperature`,
`--host`, `--port`, `--device`.

**`/synth` body** (all optional except `text`): `text`, `instruct`, `language`,
`speed`, `duration`, `ref_audio_b64`, `ref_text`, `num_step`, `guidance_scale`,
`position_temperature`, `class_temperature`, `seed`, `accent` (set `false` to skip
marking for one request).

**TrainGen env**: `TRAINGEN_TTS`, `TRAINGEN_OMNIVOICE_URL`, `TRAINGEN_VOICE_DIR`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/synth` 500, `SQLite objects created in a thread...` | Fixed — accenter uses `check_same_thread=False` + lock. If seen, update `bg_accent.py`. |
| A word is stressed wrong | `bg_stress_add.py думАта` (no restart). If still wrong, it may be a non-stress accent issue (model), not fixable by marking. |
| Clitic like `и`/`а` sounds emphasized | Should be skipped (monosyllable guard). Confirm `bg_accent.py` has the `<2 vowels` check. |
| `портАл` skipped as "monosyllable" | You typed a **Latin** vowel. Use Cyrillic `А`/`О`/`Е`. |
| Training OOM on 16 GB | Lower `batch_tokens` (1024→512), raise `gradient_accumulation_steps`; keep the `expandable_segments` env. See `BG_FINETUNE_PLAN.md`. |
| Strange/foreign accent | Use the fine-tuned model (`--model models/omnivoice-bg`), not stock; try a cleaner same-speaker reference. |
| Numbers/dates read wrong | Stress marking doesn't normalize numbers — needs a separate text-normalization pass (not yet built). |

---

## Design decisions (why it's built this way)

- **Three orthogonal levers.** Cloning fixes *voice identity*, the lexicon fixes
  *lexical stress*, the fine-tune fixes *accent/nativeness*. None substitutes for
  another — we verified each by ear.
- **Stress must be in the text.** Bulgarian stress is lexical/mobile and unmarked
  in orthography; the model can't infer it. OmniVoice *does* respect explicit
  U+0301 marks, so we supply them — the same role a pronunciation dictionary plays
  in any TTS. This is the SOTA-aligned approach (cf. BAS "Accentor").
- **rechko as the lexicon.** A ~115k-lemma authoritative Bulgarian stress
  dictionary, locally available — far better than training a G2P model. Trusted
  as-is (native-verified). Inflected forms are projected; imperatives use a
  penult-vowel rule; the rest is overridable.
- **Parse the dump, don't import.** No MySQL runtime dependency; one reproducible
  script → a portable sqlite the server reads.
- **Fine-tune was *not* redundant.** Initially assumed cloning made it
  unnecessary; A/B listening showed the fine-tuned model sounds materially more
  native. Stress still needed the lexicon regardless.
- **Server-side accenting.** The 380 MB lexicon stays on the GPU box; TrainGen
  stays a thin client sending plain text.

---

## Attribution & license

OmniVoice-BG is a derivative of **OmniVoice** by k2-fsa, licensed under
**Apache-2.0** (see `LICENSE`). The Bulgarian layer (stress engine, server,
data/eval tooling, fine-tune configs) is our addition under the same license.
Stress data is derived from the **rechko / chitanka_rechko** dictionary — verify
its license terms before any redistribution of `bg_stress.sqlite`.
```
