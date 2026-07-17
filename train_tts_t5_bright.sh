#!/usr/bin/env bash
# T5 "bright" refresh — the spectral-darkening fix (dose-dependent finding,
# 2026-07-17): the 413 h/10K-step scale-up collapsed the output's sibilance
# band (4-8 kHz: stock 10% -> ft6b 1%), while bg_keep-2750 (63 h) kept it
# bright (8.4%). This repeats the PROVEN ft4 recipe — bg_keep/checkpoint-2750
# + 3000 marked steps on the ~113 h combined set — but with the CURRENT ft6b
# mark dialect (F9), giving bright consonants + native accent + ft6b marks.
#
# Prereq re-mark (old bg400 subset tokens were NOT covered by the ft6b
# re-mark of bg400full/bg):
#   ./venv/bin/python mark_tts_txts.py --ckpt exp/accentor_ft6b/last.pt \
#       --overrides config/bg/bg_stress_overrides.json \
#       --dirs data/bg400/tokens/train data/bg400/tokens/dev
#
# Then:  tmux new -s tts; bash train_tts_t5_bright.sh     (~40 min)
set -euo pipefail
cd "$(dirname "$0")"

for d in data/bg400/tokens/train data/bg/tokens/train; do
  [ -f "$d/data_marked.lst" ] || { echo "ERROR: $d/data_marked.lst missing"; exit 1; }
  if [ "$d/txts_marked" -ot exp/accentor_ft6b/last.pt ]; then
    echo "ERROR: $d/txts_marked older than the ft6b checkpoint — run the"
    echo "       re-mark in this script's header first (F9)."; exit 1
  fi
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./venv/bin/accelerate launch \
    --gpu_ids 0 --num_processes 1 -m omnivoice.cli.train \
    --train_config config/bg/train_config_bg_combined.json \
    --data_config config/bg/data_config_bg_combined.json \
    --output_dir exp/omnivoice_bg_t5_bright
