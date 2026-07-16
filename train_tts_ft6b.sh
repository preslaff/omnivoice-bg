#!/usr/bin/env bash
# F9 marking-consistency refresh for Accentor ft6b (guide stage 6f):
# the served Accentor changed (ft5b -> ft6b/last.pt + suppression list +
# еди́н/се́бе pins), so the TTS must re-learn the new mark dialect.
# Audio tokens are REUSED — only txts_marked/ changes (mark_tts_txts.py),
# then a 3000-step refresh from the best 413 h checkpoint.
#
# Prereq (run first, ~10 min):
#   ./venv/bin/python mark_tts_txts.py --ckpt exp/accentor_ft6b/last.pt \
#       --overrides config/bg/bg_stress_overrides.json \
#       --dirs data/bg400full/tokens/train data/bg400full/tokens/dev \
#              data/bg/tokens/train data/bg/tokens/dev
#
# Then:  tmux new -s tts; bash train_tts_ft6b.sh        (~40 min)
set -euo pipefail
cd "$(dirname "$0")"

for f in data/bg400full/tokens/train/data_marked.lst data/bg400full/tokens/dev/data_marked.lst; do
  [ -f "$f" ] || { echo "ERROR: $f missing — run tokenize_bg400_full.sh stage 2 first"; exit 1; }
done
# Guard: the re-mark must be NEWER than the ft6b checkpoint, else the refresh
# would train on stale ft5b marks (F9 violation).
if [ data/bg400full/tokens/train/txts_marked -ot exp/accentor_ft6b/last.pt ]; then
  echo "ERROR: txts_marked older than exp/accentor_ft6b/last.pt — run the"
  echo "       mark_tts_txts.py re-mark (see header) before this refresh."
  exit 1
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./venv/bin/accelerate launch \
    --gpu_ids 0 --num_processes 1 -m omnivoice.cli.train \
    --train_config config/bg/train_config_bg_ft6b.json \
    --data_config config/bg/data_config_bg_full.json \
    --output_dir exp/omnivoice_bg_ft6b
