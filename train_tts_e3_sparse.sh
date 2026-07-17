#!/usr/bin/env bash
# E3 sparse-dialect refresh — the over-marking fix, done F9-correctly.
# E1 finding: dense 77% marking makes the TTS over-articulate (mark-triggered
# с/ч/ш/щ darkening); plain sounds better. E3 marks only homographs + hard
# words + pins (config/bg/sparse_keep.json). Serving sparse marks to the
# dense-trained T5 would violate F9, so we TRAIN on the sparse dialect: same
# bright recipe as T5 (bg_keep-2750 + 3000 steps on the ~113 h combined set)
# but with SPARSELY-marked txts.
#
# Prereq — sparse re-mark (writes txts_sparse/ + data_sparse.lst, leaving the
# dense txts_marked/ intact; ~5 min):
#   ./venv/bin/python mark_tts_txts.py --ckpt exp/accentor_ft6b/last.pt \
#       --overrides config/bg/bg_stress_overrides.json \
#       --sparse-keep config/bg/sparse_keep.json \
#       --dirs data/bg400/tokens/train data/bg400/tokens/dev \
#              data/bg/tokens/train data/bg/tokens/dev
#
# Then:  tmux new -s tts; bash train_tts_e3_sparse.sh      (~40 min)
set -euo pipefail
cd "$(dirname "$0")"

for d in data/bg400/tokens/train data/bg/tokens/train; do
  [ -f "$d/data_sparse.lst" ] || {
    echo "ERROR: $d/data_sparse.lst missing — run the sparse re-mark first"
    echo "       (see this script's header)"; exit 1; }
  if [ "$d/txts_sparse" -ot exp/accentor_ft6b/last.pt ]; then
    echo "ERROR: $d/txts_sparse older than the ft6b ckpt — re-mark (F9)."; exit 1
  fi
done

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./venv/bin/accelerate launch \
    --gpu_ids 0 --num_processes 1 -m omnivoice.cli.train \
    --train_config config/bg/train_config_bg_combined.json \
    --data_config config/bg/data_config_bg_sparse.json \
    --output_dir exp/omnivoice_bg_e3_sparse
