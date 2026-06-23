# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmniVoice is a massively multilingual (600+ languages) zero-shot text-to-speech model built on a **diffusion language model-style architecture**. It supports voice cloning, voice design (attribute-controlled voices), and auto voice. The package is published to PyPI as `omnivoice`; weights live at `k2-fsa/OmniVoice` on HuggingFace.

> **This repo is OmniVoice-BG** — a fork specialized for Bulgarian narration (for the TrainGen project). On top of upstream OmniVoice we added a Bulgarian stress-correction front-end, a LAN TTS server, a fine-tune pipeline, and a TrainGen client. **Read [OMNIVOICE_BG.md](OMNIVOICE_BG.md) first** for the full picture; [BG_FINETUNE_PLAN.md](BG_FINETUNE_PLAN.md) is the training deep-dive. Key added files: `omnivoice/cli/serve.py` (server), `bg_accent.py` / `build_bg_lexicon.py` / `bg_stress_add.py` (stress engine from the *rechko* dict), `prep_bg_hf.py`, `eval_bg_ab.py`, `config/bg/`. Large artifacts (`models/`, `data/`, `exp/`) are gitignored.

## Environment & Install

There are no tests, linters, or formatters configured in this repo — do not assume a `pytest`/`ruff` setup exists.

```bash
# Editable dev install (pip)
pip install -e .
# or with uv (installs CUDA torch on Linux/Windows via the pinned pytorch-cuda index)
uv sync
uv sync --extra eval   # adds WER/speaker-sim/UTMOS eval deps (jiwer, s3prl, funasr, ...)
```

PyTorch must match the target backend (CUDA / Apple `mps` / Intel `xpu`). `flash_attn` is unavailable on XPU and MPS — the model auto-falls back to SDPA.

## Common Commands

```bash
# Web demo
omnivoice-demo --ip 0.0.0.0 --port 8001

# Single inference (voice cloning / voice design / auto voice via flags)
omnivoice-infer --model k2-fsa/OmniVoice --text "..." --ref_audio ref.wav --output out.wav

# Batch inference across GPUs, driven by a JSONL test list
omnivoice-infer-batch --model k2-fsa/OmniVoice --test_list test.jsonl --res_dir results/

# Training / finetuning — always launched via accelerate, NOT python directly
accelerate launch --gpu_ids 0,1 --num_processes 2 -m omnivoice.cli.train \
    --train_config train.json --data_config data.json --output_dir exp/

# End-to-end example pipelines (data check → tokenize → train/eval)
bash examples/run_emilia.sh       # pretrain from scratch on Emilia (stages 0-2)
bash examples/run_finetune.sh     # finetune from a pretrained checkpoint
bash examples/run_eval.sh         # WER + speaker similarity + UTMOS
```

The example shell scripts use `stage`/`stop_stage` variables at the top to run individual pipeline stages.

If HuggingFace downloads fail, set `export HF_ENDPOINT="https://hf-mirror.com"`.

## Architecture

The whole model lives in one file: `omnivoice/models/omnivoice.py` (~1600 lines). `OmniVoice` is a HuggingFace `PreTrainedModel`; `from omnivoice import OmniVoice` is the public entry point.

**Core design** — a text LLM backbone (`Qwen/Qwen3-0.6B` by default) is repurposed to predict audio codec tokens via a mask-prediction / diffusion objective:

- **Audio tokens**: reference and target audio are encoded by the Higgs Audio V2 tokenizer (`HiggsAudioV2TokenizerModel`) into `num_audio_codebook` (8) parallel codebooks at 24 kHz.
- **Embeddings** (`_prepare_embed_inputs`): `input_ids` has shape `(batch, layers, seq)`. Layer 0 is text (embedded via the LLM's token embeddings); the 8 audio layers are shifted by `codebook_layer_offsets` into a single `audio_embeddings` table and **summed** across codebooks. An `audio_mask` selects text vs audio embeddings per position.
- **Output** (`audio_heads`): a single linear projects hidden states to `num_audio_codebook * audio_vocab_size`, reshaped to per-codebook logits.
- **Training loss** (`forward`): per-codebook cross-entropy (ignore_index=-100), averaged per layer then combined with `audio_codebook_weights` (`[8,8,6,6,4,4,2,2]` — earlier/coarser codebooks weighted higher).
- **Inference** (`generate` → `_generate_iterative`): iterative denoising over `num_step` (32 default, 16 fast) steps with classifier-free guidance (`guidance_scale`), confidence-based token unmasking (`_predict_tokens_with_scoring`), and Gumbel sampling. Long text is split (`_generate_chunked`, `chunk_text_punctuation`) and cross-faded.

**Three generation modes**, all through `model.generate()`:
- Voice cloning: pass `ref_audio` (+ optional `ref_text`; Whisper auto-transcribes when omitted, requires `load_asr=True`).
- Voice design: pass `instruct` (comma-separated attributes; validation/normalization in `utils/voice_design.py`).
- Auto voice: pass neither.

**Special text/control syntax** is parsed in `utils/text.py` and `_tokenize_with_nonverbal_tags`: non-verbal tags like `[laughter]`, Chinese pinyin pronunciation overrides (`ZHE2`), and English CMU-dict overrides (`[B EY1 S]`).

### Module map

- `omnivoice/cli/` — entry points: `infer.py`, `infer_batch.py`, `demo.py` (Gradio), `train.py` (accelerate). Mapped to console scripts in `pyproject.toml`.
- `omnivoice/training/` — `config.py` (`TrainingConfig` dataclass loaded from JSON), `builder.py` (builds model+tokenizer and dataloaders; adds special tokens like `<|text_start|>`, resizes embeddings), `trainer.py` (`OmniTrainer`, Accelerate + optional DeepSpeed), `checkpoint.py`.
- `omnivoice/data/` — `dataset.py` (WebDataset/JSONL readers + multiplexing), `processor.py`, `collator.py`, `batching.py` (token-budget batching; length-grouped batching used only when `attn_implementation != "flex_attention"`).
- `omnivoice/scripts/` — offline data prep: `jsonl_to_webdataset.py`, `extract_audio_tokens.py`, `denoise_audio.py`.
- `omnivoice/eval/` — WER (`wer/`), speaker similarity (`speaker_similarity/`), MOS/UTMOS (`mos/`). Needs the `eval` extra.
- `omnivoice/utils/` — `duration.py` (`RuleDurationEstimator`), `audio.py` (load/trim/silence/crossfade), `lang_map.py` (`LANG_IDS`/`LANG_NAMES`), `text.py`, `voice_design.py`.

### Data flow for training

JSONL manifests (`id`, `audio_path`, `text`, optional `language_id`) → audio tokenized into **WebDataset shards** (`scripts/extract_audio_tokens.py`, `jsonl_to_webdataset.py`) → readers in `data/dataset.py` → token-budget batching → DataLoader. Configs are split into a **train config** and a **data config** (see `examples/config/`).

### Attention implementations

`attn_implementation` (training config) is `flex_attention` by default, which enables packed-sequence training via `create_block_mask` / `document_ids`. Set it to `sdpa` for broader hardware compatibility (Intel XPU, environments without flex_attention) — see `examples/config/train_config_finetune_sdpa.json`. SDPA mode falls back to length-grouped batching (`max_sample_tokens`, `min_sample_tokens`, `max_batch_size`).

## Conventions

- New model behavior generally belongs in `omnivoice/models/omnivoice.py`; keep inference-only attributes (`text_tokenizer`, `audio_tokenizer`, `duration_estimator`, `_asr_pipe`) `None` in `__init__` — they're populated by `from_pretrained` only when `train=False`.
- All source files carry the Apache-2.0 header (Xiaomi Corp.).
- Generation defaults live in `OmniVoiceGenerationConfig`; training defaults in `TrainingConfig`. Both load from dicts/JSON and silently filter unknown keys.
