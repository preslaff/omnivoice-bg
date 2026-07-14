#!/usr/bin/env bash
# TTS re-fine-tune on ft5b-marked text (Phase B, guide 6f).
# Re-marking already done (2026-07-13, all 4 dirs). Foreground on purpose —
# run inside tmux for an overnight session:  tmux new -s tts; bash train_tts_ft5b.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./venv/bin/accelerate launch \
    --gpu_ids 0 --num_processes 1 -m omnivoice.cli.train \
    --train_config config/bg/train_config_bg_combined.json \
    --data_config config/bg/data_config_bg_combined.json \
    --output_dir exp/omnivoice_bg_ft5b
