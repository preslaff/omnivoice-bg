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
# -L dereferences symlinks: librispeech/zipformer has relative symlinks
# (asr_datamodule.py, beam_search.py, encoder_interface.py -> ../other_recipe/...)
# that would dangle when copied outside egs/librispeech. Copy real files instead.
cp -rL "$SRC" "$REC/zipformer"
cp "$OMNI/asr/bg_asr_datamodule.py" "$REC/zipformer/asr_datamodule.py"

# Re-apply the BG --eval-set {dev,test} switch to decode.py. Librispeech's decode.py
# hardwires test-clean/test-other; BG has a single dev/test holdout and must keep test
# pristine during tuning. The cp -rL above overwrites any prior patch, so we re-apply it
# here. Idempotent; fails loudly if an anchor drifts (upstream decode.py changed).
python3 - "$REC/zipformer/decode.py" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
if "--eval-set" in s:
    print("  decode.py already has --eval-set — nothing to patch"); raise SystemExit(0)
edits = [
(
'''    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",''',
'''    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--eval-set",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="BG holdout to decode: 'dev' (tune freely) or 'test' (frozen "
        "benchmark - decode once). Keeps the test set pristine during tuning.",
    )

    parser.add_argument(
        "--decoding-method",'''
),
(
'''    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]''',
'''    # BG: single-set holdout. --eval-set dev tunes freely; --eval-set test is the
    # frozen benchmark (decode once). The librispeech *_other sets are empty for BG.
    if params.eval_set == "dev":
        eval_cuts = librispeech.dev_clean_cuts()
    else:
        eval_cuts = librispeech.test_clean_cuts()

    test_sets = [params.eval_set]
    test_dl = [librispeech.test_dataloaders(eval_cuts)]'''
),
]
for anchor, new in edits:
    if anchor not in s:
        sys.exit("ERROR: decode.py patch anchor not found (upstream changed?):\n"
                 + anchor.splitlines()[0])
    s = s.replace(anchor, new, 1)
p.write_text(s, encoding="utf-8")
print("  patched decode.py with --eval-set {dev,test}")
PYEOF
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
