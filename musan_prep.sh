#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Prepare MUSAN noise/music/babble augmentation for GP-v2 (scarcity fix #1: acoustic
# conditions — the corpus is almost entirely clean/studio audio; CutMix-ing MUSAN at
# train time converts it into multi-condition data at zero data-collection cost).
#
# The BG datamodule already supports it: bg_asr_datamodule.py --enable-musan loads
# data/fbank/musan_cuts.jsonl.gz and applies CutMix(p=0.5, snr=(10,20)) in the feature
# domain — precomputed feats keep working, training speed unaffected.
#
# Stages (resumable — each skips if its output exists):
#   1. download MUSAN (OpenSLR-17, ~11 GB — NOTE: shares the ~1 MB/s link; run AFTER
#      the podcast crawl finishes, or overnight)
#   2. lhotse manifests for music/noise/speech
#   3. 80-dim fbank -> data/fbank/musan_cuts.jsonl.gz (via icefall's compute_fbank_musan)
#
# Usage:  bash musan_prep.sh
# Then GP-v2 trains with:  train.py ... --enable-musan true
set -euo pipefail

OMNI=$(cd "$(dirname "$0")" && pwd)
ICEFALL=${ICEFALL:-$HOME/icefall}
REC="$ICEFALL/egs/bg_parliament/ASR"
PY="$OMNI/venv/bin/python"
DL="$OMNI/data/musan_dl"

echo "=== stage 1: download MUSAN (~11 GB, resumable) ==="
if [ -d "$DL/musan/noise" ] && [ -d "$DL/musan/music" ] && [ -d "$DL/musan/speech" ]; then
  echo "  [skip] $DL/musan already extracted"
else
  mkdir -p "$DL"
  "$OMNI/venv/bin/lhotse" download musan "$DL"
fi

echo "=== stage 2: lhotse manifests ==="
mkdir -p "$REC/data/manifests"
if [ -f "$REC/data/manifests/musan_recordings_noise.jsonl.gz" ]; then
  echo "  [skip] manifests exist"
else
  "$OMNI/venv/bin/lhotse" prepare musan "$DL/musan" "$REC/data/manifests"
fi

echo "=== stage 3: fbank -> data/fbank/musan_cuts.jsonl.gz ==="
if [ -f "$OMNI/data/fbank/musan_cuts.jsonl.gz" ]; then
  echo "  [skip] musan_cuts.jsonl.gz exists"
else
  cd "$REC"
  [ -e local ] || ln -s "$ICEFALL/egs/librispeech/ASR/local" local
  export PYTHONPATH="$ICEFALL:${PYTHONPATH:-}"
  "$PY" local/compute_fbank_musan.py
fi

echo
"$PY" - <<'PYEOF'
from lhotse import CutSet
cs = CutSet.from_file("data/fbank/musan_cuts.jsonl.gz")
print(f"MUSAN ready: {len(cs):,} cuts, {sum(c.duration for c in cs)/3600:.1f} h "
      f"-> enable at train time with --enable-musan true")
PYEOF
