#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Stage 8 (RLAIF ladder), step 2: judge the best-of-N candidates with the
# stress-aware forced aligner. Aligns every synthetic candidate against the
# multi-pronunciation lexicon with tri3b_comb; get_prons then records which
# stress variant each candidate actually RENDERED — the same mechanism that
# harvested the 1.22M training labels, pointed at TTS output instead of
# human audio.
#
# Prereq: bon_generate.py wrote exp/kaldi_pilot/data/bon
# Usage:  bash bon_judge.sh
# Then:   ./venv/bin/python bon_score.py
set -euo pipefail

repo=$(cd "$(dirname "$0")" && pwd)
cd "$repo/exp/kaldi_pilot"
. ./path.sh
. ./cmd.sh

data=data/bon
ali=exp/tri3b_comb_bon_ali

[ -f $data/wav.scp ] || { echo "ERROR: $data missing — run bon_generate.py first"; exit 1; }
nspk=$(wc -l < $data/spk2utt)
nj=$(( ${nj:-4} < nspk ? ${nj:-4} : nspk ))

utils/fix_data_dir.sh $data
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $data exp/make_mfcc/bon mfcc
steps/compute_cmvn_stats.sh $data exp/make_mfcc/bon mfcc
utils/fix_data_dir.sh $data

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $data data/lang exp/tri3b_comb $ali

# Alignment-failure count: failed utts get no prons row -> bon_score.py
# reports them as unjudged, but surface the number here too.
fails=$(grep -l "Did not successfully decode" $ali/log/align_pass*.log 2>/dev/null | wc -l || true)
echo "align done (failed logs: $fails)"

steps/get_prons.sh --cmd "$train_cmd" $data data/lang $ali

echo "judge done -> $ali/prons.*.gz"
echo "next: ./venv/bin/python bon_score.py"
