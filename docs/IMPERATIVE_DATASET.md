# Imperative Stress Dataset — provenance & locations

Data built to teach OmniVoice-BG **imperative (final-stress) rendering** —
купи́/обади́/наслади́/вдигни́/прави́/ходи́ — the verb class the narration corpus
starves and the Accentor + rechko dict mark wrong. Two sources: a **synthetic**
BgTTS-teacher set (the bulk) and **human recordings** (for words BgTTS can't
say). Built by `teacher_distill.py`; see also [BG_FINAL_EFFORT_PLAN.md] and the
`teacher_distill` commit history.

> **Gitignore:** everything under `data/` and `exp/` is gitignored (large
> artifacts). The paths below are **on the training machine only**, not in git.
> The **source** (configs + pipeline) *is* in git and regenerates the synthetic
> audio; the **judged labels** and **human recordings** cannot be trivially
> reproduced — see [Backup](#backup).

## Source (in git — regenerates everything synthetic)

| path | what |
|---|---|
| `config/bg/imperative_verbs.txt` | **164 verbs**, native-vetted stress marks, 8 stress-pattern sections |
| `config/bg/imperative_frames.txt` | **22** sentence frames with an `{imp}` slot |
| `teacher_distill.py` | the pipeline: `expand → gen → judge → review → build` |
| `examples/record_pravi_hodi.txt` | the human-recording script (30 sentences) |

## Synthetic set — BgTTS teacher (`exp/imperatives/`, ~585 MB)

| path | rows | what |
|---|---|---|
| `exp/imperatives/bank.jsonl` | 3,608 | verb×frame sentences (`plain` + `marked` + `verb`/`verb_marked`/`homograph`) |
| `exp/imperatives/wavs/` | 7,196 | BgTTS renders — 4 reference voices × 2 seeds; ~1.8 s each |
| `exp/imperatives/gen.jsonl` | 7,196 | render log (verb, ref voice, wav path) |
| `exp/imperatives/judged.jsonl` | 7,196 | **labeled set** — per clip: `verdict` (pass/fail/uncertain) + `aligned` (stress the Kaldi judge heard) |
| `data/bg/imperatives/train.jsonl` | 6,648 | kept for training (marked text; **прави/ходи excluded** — BgTTS renders them stem-stress) |
| `data/bg/imperatives/tokens/` | 33 shards | Higgs tokens (`--brighten 6`), what ft7/ft8 trained on |

**Reference voices** (`examples/refs/`, symlinks to clean sources): 2 male
(`reklama_clean`, `viktor_nikolaev_20s`) + 2 female (`Viktoria/11s`,
`viktoria_bg`). Cycled per clip so the model learns stress, not one voice.

## Human set — native recordings (`data/bg/imperatives_human/`)

The words no synthetic teacher can produce (BgTTS + OmniVoice + Kaldi all
render/expect stem-stress for прави/ходи — A/B-confirmed).

| path | rows | what |
|---|---|---|
| `examples/pravi.wav`, `examples/hodi.wav` | — | raw continuous recordings (44.1 kHz, ~37 s each) |
| `data/bg/imperatives_human/wavs/` | 30 | segmented clips (15 прави́ + 15 ходи́, 24 kHz, silence-split, level-normalized) |
| `data/bg/imperatives_human/train.jsonl` | 30 | marked rows (verb-only, sparse-style) |
| `data/bg/imperatives_human/tokens/` | 1 shard | Higgs tokens (**no brighten** — natural speech isn't dark) |

## What trained on what

| model | data config | imperative sources |
|---|---|---|
| ft7 (`exp/omnivoice_bg_ft7`) | `data_config_bg_ft7.json` | BgTTS 6,667 clips (incl. polluted прави/ходи), repeat 8 |
| ft8 (`exp/omnivoice_bg_ft8`) | `data_config_bg_ft8.json` | BgTTS 6,648 (прави/ходи excluded) repeat 8 + human 30 repeat 20 |
| ft9 (`exp/omnivoice_bg_ft9`) *(planned)* | `data_config_bg_ft9.json` | BgTTS 6,648 repeat 8 + human 30 repeat 12 + **volunteer many-voice set** repeat ~6 (tune to volume) |

## Known limitation (why this dataset didn't fully solve imperatives)

**Imperative stress is learned voice-specifically, not as a voice-independent
text→stress rule.** A/B evidence: each verb renders correctly only with the
voice it was trained in — обади (BgTTS voices) works with the reklama clone but
not the user's voice; прави/ходи (human, one voice) work with the user's voice
but not reklama. From few clips per (verb, voice) the model memorizes rather
than abstracts. Consequence: no checkpoint renders imperatives robustly across
arbitrary cloned voices; the text mark is a weak lever vs. the reference's
timbre/prosody. Practical fallbacks: **BgTTS for imperative-heavy ad copy**
(its own prior; serve.py engine routing), and/or serve-time best-of-N + stress
judge for "sometimes-right" verbs. See the ft8-validation discussion.

## Volunteer many-voice set — Track B (the ceiling-breaker)

The root cause above is **too few speakers per imperative**. The fix is speaker
*diversity*: collect the same marked imperatives from many real volunteers so the
model can abstract a voice-independent text→stress rule (real timbres, unlike
pitch-shifted pseudo-speakers). Tooling:

| tool | role |
|---|---|
| `omnivoice/cli/collect.py` (`omnivoice-collect`) | FastAPI mic-record web app. Serves marked prompts from `imperative_verbs.txt`×`imperative_frames.txt`, **ending-stress class first**, per-speaker frame variety. Stores browser recordings + `submissions.jsonl` under `data/bg/imperatives_volunteer/`. Duration gate on submit; resumes per speaker across reloads. **No auth** — LAN/trusted group or front with a proxy. |
| `convert_volunteer.py` | Decodes accepted clips → 24 kHz mono, full QC (RMS floor + clipping + duration), peak-normalises, writes marked `train.jsonl` (imperatives_human schema) + prints the GPU tokenize command (**no `--brighten`** — natural speech). |
| `config/bg/data_config_bg_ft9.json` | ft9 mix = base bg + BgTTS synth + human + **volunteer set** (repeat tuned to volume). Reuses `train_config_bg_ft7.json`. |

**Validate ft9 across MULTIPLE held-out reference voices** with
`sample_imperatives.py` — voice-invariance (прави́/ходи́/купи́ ending-stress for
*unseen* clones) is the pass bar, not one training voice. Full roadmap:
`.claude/plans/woolly-swimming-popcorn.md`.

## Regenerate

```bash
./venv/bin/python teacher_distill.py expand          # verbs×frames -> bank.jsonl
./venv/bin/python teacher_distill.py gen  --bgtts-dir models/bgtts-38m \
    --ref-audio examples/refs/ --seeds 2             # -> wavs/ (GPU)
./venv/bin/python teacher_distill.py judge --judge kaldi --model tri2b   # -> judged.jsonl
./venv/bin/python teacher_distill.py build --oversample 8 --exclude-verbs "прави ходи"
# human: re-segment examples/{pravi,hodi}.wav (see teacher_distill / the ft8 build)
```

## Backup

Two bundles are not reproducible from git and should be preserved:
- **`exp/imperatives/`** — the full synthetic set **+ `judged.jsonl`** (the
  Kaldi labels; re-judging needs the Kaldi models and is non-deterministic).
- **`data/bg/imperatives_human/`** + **`examples/{pravi,hodi}.wav`** — the human
  recordings (irreplaceable without re-recording).

Everything else (tokens, train manifests) rebuilds from these + the git source.
