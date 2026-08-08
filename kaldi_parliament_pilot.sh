#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Parliament VIABILITY PILOT — run the paper's §3 annotation loop on the
# ~100 h of sessions already banked by the crawl, BEFORE the full corpus is
# in. Bootstraps from the audiobook GMMs trained by kaldi_pilot.sh:
#
#   long sessions + stenographic transcripts
#     -> segment_long_utterances.sh (biased-LM decode, seed = tri2b, SI)
#     -> align_fmllr with tri3b (speaker = session -> channel adaptation)
#     -> get_prons + harvest_stress_labels.py
#     -> data/kaldi/stress_labels_parliament.jsonl   (human homograph labels)
#
# PREREQUISITE: kaldi_pilot.sh completed through stage 3 (exp/tri2b,
# exp/tri3b exist in exp/kaldi_pilot).
#
# Memory note (see docs/BG_PIPELINE_GUIDE.md "Pitfalls"): biased-LM graphs
# hold a whole session transcript each (~30K words), so segmentation runs at
# seg_nj=8, not nproc. MFCC/align on the short segments use full nj.
#
# Usage:   KALDI_ROOT=~/kaldi bash kaldi_parliament_pilot.sh
# Resume:  KALDI_ROOT=~/kaldi stage=N bash kaldi_parliament_pilot.sh

set -euo pipefail

stage=${stage:-0}
stop_stage=${stop_stage:-99}
nj=${nj:-$(nproc)}
seg_nj=${seg_nj:-8}                  # memory-bound (one session transcript per job)
KALDI_ROOT=${KALDI_ROOT:?set KALDI_ROOT to your kaldi checkout}
repo=$(cd "$(dirname "$0")" && pwd)
work=$repo/exp/kaldi_pilot           # SAME workdir as kaldi_pilot.sh (reuses lang + GMMs)
kdata=$repo/data/kaldi

[ -f "$work/exp/tri3b/final.mdl" ] || {
  echo "ERROR: run kaldi_pilot.sh through stage 3 first (need exp/tri2b + exp/tri3b)"; exit 1; }

cd "$work"
. ./path.sh; . ./cmd.sh

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "=== stage 0: parliament long-utterance dir + dict/lang with combined vocab ==="
  (cd "$repo" && ./venv/bin/python prep_kaldi_data.py --out data/kaldi \
      --set sivov=data/bg/train.jsonl \
      --set bg400=data/bg400/train.jsonl \
      --long-set parl=data/bg_parliament/manifest.jsonl)
  rm -rf data/local data/lang
  cp -r "$kdata/local" data/
  rm -rf data/parl                 # rerun refresh: crawl keeps adding sessions
  cp -r "$kdata/parl" data/parl
  utils/fix_data_dir.sh data/parl || true   # long utts: no feats yet, ok
  # rebuild lang: same phone set -> tri2b/tri3b stay compatible; words.txt grows
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "=== stage 1: segment long sessions (biased-LM decode, seed=tri2b) ==="
  # segment_long_utterances.sh requires feats on the LONG recordings
  # (it sub-segments them internally). One 2 h session ~= 40 MB of MFCC.
  steps/make_mfcc.sh --nj $seg_nj --cmd "$train_cmd" data/parl exp/make_mfcc/parl mfcc
  steps/compute_cmvn_stats.sh data/parl exp/make_mfcc/parl mfcc
  utils/fix_data_dir.sh data/parl
  # Speaker-independent seed on purpose: unknown speakers at decode time.
  # Discards audio/transcript regions that don't match (interjections,
  # unverbalized numbers) — that lost yield is expected, not a bug.
  steps/cleanup/segment_long_utterances.sh --nj $seg_nj --cmd "$train_cmd" \
      exp/tri2b data/lang data/parl data/parl_seg exp/segment_parl
  utils/fix_data_dir.sh data/parl_seg
  echo "--- segmentation yield ---"
  awk '{n++; d+=$4-$3} END {printf "%d segments, %.1f h retained\n", n, d/3600}' \
      data/parl_seg/segments
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "=== stage 2: MFCC + fMLLR alignment of the segments (tri3b) ==="
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/parl_seg exp/make_mfcc/parl_seg mfcc
  steps/compute_cmvn_stats.sh data/parl_seg exp/make_mfcc/parl_seg mfcc
  utils/fix_data_dir.sh data/parl_seg
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/parl_seg data/lang exp/tri3b exp/tri3b_parl_ali
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "=== stage 3: harvest parliament stress labels ==="
  steps/get_prons.sh --cmd "$train_cmd" data/parl_seg data/lang exp/tri3b_parl_ali
  echo "--- stress splits observed for battery homographs (parliament) ---"
  for w in вълна пара говори чета коса; do
    grep " $w " exp/tri3b_parl_ali/pron_counts_nowb.txt || true
  done
  (cd "$repo" && ./venv/bin/python harvest_stress_labels.py \
      --pilot exp/kaldi_pilot --ali exp/tri3b_parl_ali --data data/parl_seg \
      --out data/kaldi/stress_labels_parliament.jsonl)
  echo "Viability check: (a) segmentation yield >= ~50% of 96 h,"
  echo "(b) homograph splits look context-plausible, (c) native vetting of"
  echo "~20 tokens passes. Then the same loop scales to the full crawl."
fi

# ---------------------------------------------------------------------------
# ROUND 2 — in-domain bootstrap iteration. Round 1 (2026-07-05) yielded only
# 20% / 23.8 h with 10-35% stress-label noise (minority counts on known
# single-stress words like тези/или/това): the audiobook-only GMMs had never
# heard parliament acoustics. Standard fix: retrain on audiobooks + the
# round-1 parliament segments, re-segment, re-align with the in-domain models.
# ---------------------------------------------------------------------------

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "=== stage 4: in-domain GMMs (audiobooks 63h + parl round-1 23.8h) ==="
  utils/combine_data.sh data/train_comb data/train data/parl_seg
  utils/fix_data_dir.sh data/train_comb
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train_comb data/lang exp/tri3b exp/tri3b_comb_ali
  # SI LDA+MLLT model = round-2 segmentation seed (unknown speakers at decode)
  steps/train_lda_mllt.sh --cmd "$train_cmd" 4500 60000 \
      data/train_comb data/lang exp/tri3b_comb_ali exp/tri2b_comb
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/train_comb data/lang exp/tri2b_comb exp/tri2b_comb_ali
  # SAT model = round-2 final aligner
  steps/train_sat.sh --cmd "$train_cmd" 5500 120000 \
      data/train_comb data/lang exp/tri2b_comb_ali exp/tri3b_comb
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "=== stage 5: re-segment the long sessions with the in-domain seed ==="
  # (re)compute long-session MFCC: a stage-0 rerun refreshes data/parl with
  # the sessions the crawl banked since — they arrive here without feats.
  steps/make_mfcc.sh --nj $seg_nj --cmd "$train_cmd" data/parl exp/make_mfcc/parl mfcc
  steps/compute_cmvn_stats.sh data/parl exp/make_mfcc/parl mfcc
  utils/fix_data_dir.sh data/parl
  steps/cleanup/segment_long_utterances.sh --nj $seg_nj --cmd "$train_cmd" \
      exp/tri2b_comb data/lang data/parl data/parl_seg2 exp/segment_parl2
  utils/fix_data_dir.sh data/parl_seg2
  echo "--- round-2 segmentation yield (round 1 was 23.8 h / 20%) ---"
  awk '{n++; d+=$4-$3} END {printf "%d segments, %.1f h retained\n", n, d/3600}' \
      data/parl_seg2/segments
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "=== stage 6: round-2 alignment + harvest ==="
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/parl_seg2 exp/make_mfcc/parl_seg2 mfcc
  steps/compute_cmvn_stats.sh data/parl_seg2 exp/make_mfcc/parl_seg2 mfcc
  utils/fix_data_dir.sh data/parl_seg2
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/parl_seg2 data/lang exp/tri3b_comb exp/tri3b_parl_ali2
  steps/get_prons.sh --cmd "$train_cmd" data/parl_seg2 data/lang exp/tri3b_parl_ali2
  echo "--- NOISE METER: minority rate on known single-stress words ---"
  for w in тези или това бъде вече време има между; do
    grep " $w " exp/tri3b_parl_ali2/pron_counts_nowb.txt || true
  done
  (cd "$repo" && ./venv/bin/python harvest_stress_labels.py \
      --pilot exp/kaldi_pilot --ali exp/tri3b_parl_ali2 --data data/parl_seg2 \
      --out data/kaldi/stress_labels_parliament2.jsonl)
fi
