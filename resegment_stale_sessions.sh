#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# RECOVER the 110 quarantined parliament sessions (~257 h) for the ASR v2 corpus.
#
# Background (see the 3edd7d6 "quarantine stale parliament sessions" commit):
# a refresh_bg_parliament pass replaced 110/212 parliament FLACs with SHORTER audio
# but did NOT re-derive the Kaldi segments, so the old segments run past the real end
# of the current audio. detect_stale_sessions.py blocklisted them
# (data/lhotse/bg/stale_sessions.txt) and staged their manifest rows
# (data/bg_parliament/manifest_stale.jsonl). The audio + transcripts are intact — the
# ONLY thing broken is the segment/audio alignment, so re-segmenting the CURRENT audio
# against the same transcript recovers the data.
#
# This is byte-for-byte the segment_asr_sessions.sh stage 1-2 pipeline
# (tri2b_comb biased-LM segmentation -> align_fmllr tri4b -> clean_and_segment), but:
#   * seeded from manifest_stale.jsonl (the 110 stale sessions) instead of a crawl diff;
#   * writes to a DISTINCT ${SET}=parl_restale workspace so it NEVER touches the existing
#     parl_new*/train_cleaned dirs the v1 corpus uses;
#   * STOPS at data/parl_restale_cleaned (segments + text + wav.scp), lhotse-importable.
#
# The aligner models (tri2b_comb / tri4b) are READ-ONLY inputs — untouched. This is a
# CPU-only Kaldi run (no GPU) — but run it AFTER the baseline finishes so it does not
# starve the training dataloader for cores.
#
# Usage:  KALDI_ROOT=~/kaldi bash resegment_stale_sessions.sh
# Resume: KALDI_ROOT=~/kaldi stage=N bash resegment_stale_sessions.sh   (0 prep, 1 seg, 2 clean)
set -euo pipefail

nj=${nj:-$(nproc)}
seg_nj=${seg_nj:-8}                 # memory-bound: one whole-session biased-LM graph per job
stage=${stage:-0}
SET=${SET:-parl_restale}
KALDI_ROOT=${KALDI_ROOT:?set KALDI_ROOT to your kaldi checkout}
repo=$(cd "$(dirname "$0")" && pwd)
work=$repo/exp/kaldi_pilot
kdata=$repo/data/kaldi

SEED=exp/tri2b_comb                 # segmentation biased-LM decode seed (READ-ONLY)
GMM=exp/tri4b                       # cleanup + fMLLR aligner            (READ-ONLY)
manifest_stale=data/bg_parliament/manifest_stale.jsonl

# --- guards: don't clobber v1 dirs, and fail loudly if inputs are missing ---
case "$SET" in
  parl_new|parl_new2|train_cleaned|parl) echo "ERROR: SET=$SET collides with a v1 corpus dir"; exit 1;;
esac
[ -s "$repo/$manifest_stale" ] || { echo "ERROR: $manifest_stale missing/empty (run detect_stale_sessions.py)"; exit 1; }

cd "$work"
[ -e steps ] || ln -s "$KALDI_ROOT/egs/wsj/s5/steps" steps
[ -e utils ] || ln -s "$KALDI_ROOT/egs/wsj/s5/utils" utils
. ./path.sh; . ./cmd.sh
for d in "$SEED" "$GMM" data/lang; do
  [ -d "$d" ] || { echo "ERROR: $d missing — run milestone A first"; exit 1; }
done

n_stale=$(wc -l < "$repo/$manifest_stale")
echo "re-segmenting $n_stale stale sessions -> data/${SET}_cleaned  (seed $SEED, clean $GMM)"

# --- stage 0: long-utterance dir for the stale sessions (points at CURRENT audio) ---
if [ "$stage" -le 0 ]; then
  echo "=== stage 0: prep data/${SET} from ${manifest_stale} ==="
  (cd "$repo" && ./venv/bin/python prep_kaldi_data.py --out data/kaldi \
      --long-set "${SET}=${manifest_stale}")
  rm -rf "data/${SET}" && cp -r "$kdata/${SET}" "data/${SET}"
  utils/fix_data_dir.sh "data/${SET}" || true
  steps/make_mfcc.sh --nj $seg_nj --cmd "$train_cmd" "data/${SET}" "exp/make_mfcc/${SET}" mfcc
  steps/compute_cmvn_stats.sh "data/${SET}" "exp/make_mfcc/${SET}" mfcc
  utils/fix_data_dir.sh "data/${SET}"
fi

# --- stage 1: segment the CURRENT audio against the transcript (tri2b_comb seed) ---
if [ "$stage" -le 1 ]; then
  echo "=== stage 1: segment (seed $SEED) ==="
  if [ -f "data/${SET}_seg/feats.scp" ] && [ -s "data/${SET}_seg/segments" ]; then
    echo "  [skip] data/${SET}_seg already built ($(wc -l <data/${SET}_seg/segments) segments) — reusing"
  else
    steps/cleanup/segment_long_utterances.sh --nj $seg_nj --cmd "$train_cmd" \
        "$SEED" data/lang "data/${SET}" "data/${SET}_seg" "exp/segment_${SET}"
    utils/fix_data_dir.sh "data/${SET}_seg"
    awk '{n++; d+=$4-$3} END {printf "segmented: %d segments, %.1f h retained\n", n, d/3600}' \
        "data/${SET}_seg/segments"
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" "data/${SET}_seg" "exp/make_mfcc/${SET}_seg" mfcc
    steps/compute_cmvn_stats.sh "data/${SET}_seg" "exp/make_mfcc/${SET}_seg" mfcc
    utils/fix_data_dir.sh "data/${SET}_seg"
  fi
fi

# --- stage 2: fMLLR-align (tri4b) + clean_and_segment -> ASR-ready _cleaned dir ---
if [ "$stage" -le 2 ]; then
  echo "=== stage 2: align_fmllr ($GMM) + clean_and_segment ==="
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      "data/${SET}_seg" data/lang "$GMM" "exp/tri4b_ali_${SET}"
  steps/cleanup/clean_and_segment_data.sh --nj $nj --cmd "$train_cmd" \
      "data/${SET}_seg" data/lang "exp/tri4b_ali_${SET}" "exp/tri4b_cleanup_${SET}" "data/${SET}_cleaned"
  utils/fix_data_dir.sh "data/${SET}_cleaned"
fi

# --- report: hours recovered + how many of the 110 sessions came back ---
echo
awk '{n++; d+=$4-$3} END {printf "ASR-ready: %d segments, %.1f h in data/%s_cleaned\n", n, d/3600, "'"$SET"'"}' \
    "data/${SET}_cleaned/segments"
recovered=$(awk '{sub(/-[0-9]+$/,"",$2); print $2}' "data/${SET}_cleaned/segments" | sort -u | wc -l)
echo "recovered ${recovered}/${n_stale} sessions (rest had no transcript-matching audio — stay quarantined)"
echo
echo "-> exp/kaldi_pilot/data/${SET}_cleaned  (segments/text/wav.scp — lhotse-importable)"
echo "   v1 dirs untouched; aligner (tri2b_comb/tri4b) read-only."
echo
echo "NEXT (v2 corpus — TRAIN ONLY, frozen dev/test unchanged):"
echo "  1. prep_lhotse_manifests.py --reseg-dir exp/kaldi_pilot/data/${SET}_cleaned"
echo "     (recovered segments OVERRIDE the stale ones; still-broken sessions stay excluded)"
echo "  2. compute_fbank_bg.py --splits train   (dev/test fbank already frozen)"
echo "  3. retrain v2 from scratch on the larger train set; decode the SAME frozen test set"
