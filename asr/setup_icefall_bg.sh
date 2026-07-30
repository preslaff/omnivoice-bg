#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Scaffold an icefall zipformer recipe for the Bulgarian ASR at
# $ICEFALL/egs/bg_parliament/ASR, wired to this repo's fbank + BPE outputs.
#
# It copies the (dataset-agnostic) librispeech zipformer scripts, drops in our
# BG datamodule (asr/bg_asr_datamodule.py -> zipformer/asr_datamodule.py), and
# symlinks our data/fbank + data/lang_bpe_500 into the recipe. Nothing here
# trains or downloads — pure file scaffolding, safe to re-run (idempotent).
#
# Usage:  ICEFALL=~/icefall bash asr/setup_icefall_bg.sh
set -euo pipefail

ICEFALL=${ICEFALL:-$HOME/icefall}
OMNI=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ICEFALL/egs/librispeech/ASR/zipformer"
REC="$ICEFALL/egs/bg_parliament/ASR"

[ -d "$SRC" ] || { echo "ERROR: librispeech zipformer recipe not found at $SRC"; exit 1; }
[ -f "$OMNI/asr/bg_asr_datamodule.py" ] || { echo "ERROR: BG datamodule missing"; exit 1; }
for f in bg_cuts_train.jsonl.gz bg_cuts_dev.jsonl.gz bg_cuts_test.jsonl.gz; do
  [ -e "$OMNI/data/fbank/$f" ] || echo "  NOTE: $OMNI/data/fbank/$f not present yet (fbank still running?)"
done
[ -e "$OMNI/data/lang_bpe_500/bpe.model" ] || echo "  NOTE: data/lang_bpe_500/bpe.model missing (run train_bpe_bg.py)"

mkdir -p "$REC/data"
rm -rf "$REC/zipformer"
cp -r "$SRC" "$REC/zipformer"
cp "$OMNI/asr/bg_asr_datamodule.py" "$REC/zipformer/asr_datamodule.py"
[ -e "$ICEFALL/egs/librispeech/ASR/shared" ] && \
  ln -sfn "$ICEFALL/egs/librispeech/ASR/shared" "$REC/shared"
ln -sfn "$OMNI/data/fbank"          "$REC/data/fbank"
ln -sfn "$OMNI/data/lang_bpe_500"   "$REC/data/lang_bpe_500"

echo "recipe scaffolded -> $REC"
echo "  zipformer/asr_datamodule.py  = BG (bg_cuts_{train,dev,test}, MUSAN off)"
echo "  data/fbank -> $OMNI/data/fbank"
echo "  data/lang_bpe_500 -> $OMNI/data/lang_bpe_500"
echo
echo "Launch training from $REC with PYTHONPATH=$ICEFALL (see asr/README below)."
