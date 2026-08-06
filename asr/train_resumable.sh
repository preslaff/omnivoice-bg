#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Power-cut-resilient launcher for the BG Zipformer run. icefall saves epoch-N.pt at
# the end of every epoch; this wrapper detects the highest completed epoch and resumes
# from the next one, and relaunches automatically if train.py dies (crash OR — after you
# rerun this script post-reboot — a power cut). Worst case you re-do the in-progress
# epoch (~1 h), never the whole run.
#
# Usage (under tmux):
#   bash ~/WORK/OmniVoice/asr/train_resumable.sh
# After a reboot, just run the same line again — it picks up from the last epoch-N.pt.
# For zero-touch recovery, add to crontab:  @reboot tmux new -d -s train 'bash .../train_resumable.sh'
set -uo pipefail

ICEFALL=${ICEFALL:-$HOME/icefall}
REC="$ICEFALL/egs/bg_parliament/ASR"
EXP="${EXP:-zipformer/exp}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
MAX_DUR="${MAX_DUR:-350}"
PY="${PY:-$HOME/WORK/OmniVoice/venv/bin/python}"
ENABLE_MUSAN="${ENABLE_MUSAN:-false}"        # GP-v2: true (needs data/fbank/musan_cuts.jsonl.gz)
TRAIN_MANIFEST="${TRAIN_MANIFEST:-bg_cuts_train.jsonl.gz}"   # GP-v2: bg_cuts_train_gp.jsonl.gz

cd "$REC"
export PYTHONPATH="$ICEFALL:${PYTHONPATH:-}"

while true; do
  last=$(ls "$EXP"/epoch-*.pt 2>/dev/null \
         | sed -E 's/.*epoch-([0-9]+)\.pt/\1/' | sort -n | tail -1)
  start=$(( ${last:-0} + 1 ))
  if [ "$start" -gt "$NUM_EPOCHS" ]; then
    echo "[resumable] all $NUM_EPOCHS epochs done (last=$last) — nothing to do."
    break
  fi
  echo "[resumable] launching train.py --start-epoch $start (last completed: ${last:-none})"
  "$PY" zipformer/train.py \
    --world-size 1 --num-epochs "$NUM_EPOCHS" --start-epoch "$start" \
    --use-fp16 1 --causal 0 \
    --exp-dir "$EXP" \
    --bpe-model data/lang_bpe_500/bpe.model \
    --max-duration "$MAX_DUR" --num-workers 2 --save-every-n 2000 \
    --enable-musan "$ENABLE_MUSAN" --train-manifest "$TRAIN_MANIFEST"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "[resumable] train.py finished cleanly."
    break
  fi
  echo "[resumable] train.py exited with code $code — resuming in 10s (Ctrl-C to stop)."
  sleep 10
done
