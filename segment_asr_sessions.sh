#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# ASR-only segmentation of NEW crawl sessions into a SEPARATE workspace.
#
# Same segment+clean pipeline as harvest_new_sessions.sh stages 1-2 (tri2b_comb
# biased-LM segmentation -> align_fmllr with tri4b -> clean_and_segment), but:
#   * writes to a distinct ${SET} workspace (default parl_new2) so it NEVER
#     touches the existing parl_new_* dirs (the 2026-07-28 72-session incr set the
#     ASR corpus already uses) or the harvest_new_sessions.sh label pipeline;
#   * STOPS after producing data/${SET}_cleaned (segments + text + wav.scp) —
#     ASR needs no TDNN align / stress-label harvest (that's the Accentor track).
#
# The aligner models (tri2b_comb / tri4b) are READ-ONLY inputs — untouched.
# Output data/${SET}_cleaned is directly importable by lhotse for the ASR corpus.
#
# Usage:  KALDI_ROOT=~/kaldi bash segment_asr_sessions.sh
#         KALDI_ROOT=~/kaldi SET=parl_new2 bash segment_asr_sessions.sh
set -euo pipefail

nj=${nj:-$(nproc)}
seg_nj=${seg_nj:-8}
SET=${SET:-parl_new2}
KALDI_ROOT=${KALDI_ROOT:?set KALDI_ROOT to your kaldi checkout}
repo=$(cd "$(dirname "$0")" && pwd)
work=$repo/exp/kaldi_pilot
kdata=$repo/data/kaldi

SEED=exp/tri2b_comb          # segmentation biased-LM decode seed (read-only)
GMM=exp/tri4b                # cleanup + fMLLR (read-only)
manifest=data/bg_parliament/manifest.jsonl
manifest_new=data/bg_parliament/manifest_${SET}.jsonl

cd "$work"
[ -e steps ] || ln -s "$KALDI_ROOT/egs/wsj/s5/steps" steps
[ -e utils ] || ln -s "$KALDI_ROOT/egs/wsj/s5/utils" utils
. ./path.sh; . ./cmd.sh

for d in "$SEED" "$GMM" data/lang; do
  [ -d "$d" ] || { echo "ERROR: $d missing — run milestone A first"; exit 1; }
done
[ "$SET" = parl_new ] && { echo "ERROR: SET=parl_new would collide with the incr dirs; pick another (e.g. parl_new2)"; exit 1; }

# --- stage 0: which sessions are new? (in manifest, not in any labels set) ---
echo "=== stage 0: diff crawl manifest vs already-harvested sessions ==="
(cd "$repo" && ./venv/bin/python - "$manifest" "$manifest_new" \
     $(ls data/kaldi/stress_labels_milestone_a.jsonl data/kaldi/stress_labels_incr_*.jsonl 2>/dev/null) <<'PY'
import json, re, sys, pathlib
manifest, manifest_new, *label_files = sys.argv[1:]
done = set()
for lf in label_files:
    p = pathlib.Path(lf)
    if not p.exists(): continue
    for line in open(p, encoding='utf-8'):
        m = re.match(r'(plsten_\d+)', json.loads(line).get('utt', ''))
        if m: done.add(m.group(1))
new = [l for l in open(manifest, encoding='utf-8') if json.loads(l)['id'] not in done]
print(f'harvested sessions: {len(done)};  new: {len(new)}', file=sys.stderr)
if not new: sys.exit(9)
pathlib.Path(manifest_new).write_text(''.join(new), encoding='utf-8')
print(f'wrote {manifest_new} ({len(new)} sessions)', file=sys.stderr)
PY
) || { rc=$?; [ $rc -eq 9 ] && { echo "No new sessions — nothing to segment."; exit 0; }; exit $rc; }

# --- stage 1: prep the long-utterance dir for the NEW sessions only ---
echo "=== stage 1: prep data/${SET} from ${manifest_new} ==="
(cd "$repo" && ./venv/bin/python prep_kaldi_data.py --out data/kaldi \
    --long-set "${SET}=${manifest_new}")
rm -rf "data/${SET}" && cp -r "$kdata/${SET}" "data/${SET}"
utils/fix_data_dir.sh "data/${SET}" || true
steps/make_mfcc.sh --nj $seg_nj --cmd "$train_cmd" "data/${SET}" "exp/make_mfcc/${SET}" mfcc
steps/compute_cmvn_stats.sh "data/${SET}" "exp/make_mfcc/${SET}" mfcc
utils/fix_data_dir.sh "data/${SET}"

# --- stage 2: segment (tri2b_comb seed) + clean (tri4b fMLLR) ---
echo "=== stage 2: segment (seed $SEED) + clean (fMLLR $GMM) ==="
if [ -f "data/${SET}_seg/feats.scp" ] && [ -s "data/${SET}_seg/segments" ]; then
  echo "  [skip] data/${SET}_seg already built ($(wc -l <data/${SET}_seg/segments) segments) — reusing"
else
  steps/cleanup/segment_long_utterances.sh --nj $seg_nj --cmd "$train_cmd" \
      "$SEED" data/lang "data/${SET}" "data/${SET}_seg" "exp/segment_${SET}"
  utils/fix_data_dir.sh "data/${SET}_seg"
  awk '{n++; d+=$4-$3} END {printf "new: %d segments, %.1f h retained\n", n, d/3600}' \
      "data/${SET}_seg/segments"
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" "data/${SET}_seg" "exp/make_mfcc/${SET}_seg" mfcc
  steps/compute_cmvn_stats.sh "data/${SET}_seg" "exp/make_mfcc/${SET}_seg" mfcc
  utils/fix_data_dir.sh "data/${SET}_seg"
fi
# fMLLR-align the NEW segmented data with tri4b (trans.* keyed to the new speakers)
steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    "data/${SET}_seg" data/lang "$GMM" "exp/tri4b_ali_${SET}"
steps/cleanup/clean_and_segment_data.sh --nj $nj --cmd "$train_cmd" \
    "data/${SET}_seg" data/lang "exp/tri4b_ali_${SET}" "exp/tri4b_cleanup_${SET}" "data/${SET}_cleaned"
utils/fix_data_dir.sh "data/${SET}_cleaned"

echo
awk '{n++; d+=$4-$3} END {printf "ASR-ready: %d segments, %.1f h in data/%s_cleaned\n", n, d/3600, "'"$SET"'"}' \
    "data/${SET}_cleaned/segments"
echo "-> exp/kaldi_pilot/data/${SET}_cleaned  (segments/text/wav.scp — lhotse-importable)"
echo "   existing parl_new_* dirs untouched; aligner (tri2b_comb/tri4b) read-only."
