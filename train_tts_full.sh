#!/usr/bin/env bash
# TTS fine-tune on the FULL ~413 h corpus (bg + bg400full), ft5b marks.
# Prereq: tokenize_bg400_full.sh completed (all 3 stages).
# 10K steps ≈ 2.3 h at the ft4/ft5b pace. Run inside tmux:
#   tmux new -s tts; bash train_tts_full.sh
set -euo pipefail
cd "$(dirname "$0")"

for f in data/bg400full/tokens/train/data_marked.lst data/bg400full/tokens/dev/data_marked.lst; do
  [ -f "$f" ] || { echo "ERROR: $f missing — run tokenize_bg400_full.sh first"; exit 1; }
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./venv/bin/accelerate launch \
    --gpu_ids 0 --num_processes 1 -m omnivoice.cli.train \
    --train_config config/bg/train_config_bg_full.json \
    --data_config config/bg/data_config_bg_full.json \
    --output_dir exp/omnivoice_bg_full
