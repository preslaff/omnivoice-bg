#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Kaldi alignment pilot — Phase 2 groundwork for the Accentor reproduction.
# Trains a GMM aligner (mono -> tri -> LDA+MLLT -> SAT) on the real audiobook
# corpora with the stressed/unstressed-vowel lexicon, producing phone-level
# alignments — the prerequisite for the stress-annotation TDNN (gap item 2)
# and the per-sentence stress-decoding graphs.
#
# One-time install (CPU build is enough for GMM; ~30-60 min compile):
#   git clone https://github.com/kaldi-asr/kaldi /opt/kaldi   # or ~/kaldi
#   cd /opt/kaldi/tools  && extras/check_dependencies.sh && make -j $(nproc)
#   cd /opt/kaldi/src    && ./configure --shared && make -j $(nproc)
#
# Then:  KALDI_ROOT=/opt/kaldi bash kaldi_pilot.sh
# Re-run with  stage=N  to resume:  KALDI_ROOT=... stage=3 bash kaldi_pilot.sh

set -euo pipefail

stage=${stage:-0}
stop_stage=${stop_stage:-99}
nj=${nj:-$(nproc)}
KALDI_ROOT=${KALDI_ROOT:?set KALDI_ROOT to your kaldi checkout}
repo=$(cd "$(dirname "$0")" && pwd)
work=$repo/exp/kaldi_pilot           # kaldi wants to own its cwd layout
kdata=$repo/data/kaldi

mkdir -p "$work" && cd "$work"
# Standard recipe scaffolding: steps/utils symlinks + path.sh + cmd.sh
[ -e steps ] || ln -s "$KALDI_ROOT/egs/wsj/s5/steps" steps
[ -e utils ] || ln -s "$KALDI_ROOT/egs/wsj/s5/utils" utils
cat > path.sh <<EOF
export KALDI_ROOT=$KALDI_ROOT
export PATH=\$KALDI_ROOT/src/bin:\$KALDI_ROOT/src/gmmbin:\$KALDI_ROOT/src/featbin:\$KALDI_ROOT/src/fstbin:\$KALDI_ROOT/src/latbin:\$KALDI_ROOT/tools/openfst/bin:\$PWD/utils:\$PWD/shims:\$PATH
export LC_ALL=C
EOF
# Kaldi diagnostic scripts want a bare `python` (absent on Ubuntu 24.04).
mkdir -p shims && ln -sf "$(command -v python3)" shims/python
cat > cmd.sh <<'EOF'
export train_cmd=run.pl
export decode_cmd=run.pl
EOF
mkdir -p conf
cat > conf/mfcc.conf <<'EOF'
--use-energy=false
--sample-frequency=16000
EOF
. ./path.sh; . ./cmd.sh

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "=== stage 0: format lexicon + audiobook sets ==="
  (cd "$repo" && ./venv/bin/python prep_kaldi_data.py --out data/kaldi \
      --set sivov=data/bg/train.jsonl \
      --set bg400=data/bg400/train.jsonl)
  mkdir -p data
  rm -rf data/local            # cp -r into an existing dir would nest local/local
  cp -r "$kdata/local" data/
  utils/combine_data.sh data/train "$kdata/sivov" "$kdata/bg400"
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "=== stage 1: prepare_lang (2.5M-word lexicon — takes a while) ==="
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "=== stage 2: MFCC + CMVN ==="
  steps/make_mfcc.sh --nj $nj data/train exp/make_mfcc/train mfcc
  steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "=== stage 3: GMM ladder (mono -> tri1 -> tri2b LDA+MLLT -> tri3b SAT) ==="
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
  steps/train_mono.sh  --nj $nj --cmd "$train_cmd" data/train_10kshort data/lang exp/mono
  steps/align_si.sh    --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" 2500 30000 data/train data/lang exp/mono_ali exp/tri1
  steps/align_si.sh    --nj $nj --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali
  steps/train_lda_mllt.sh --cmd "$train_cmd" 4000 50000 data/train data/lang exp/tri1_ali exp/tri2b
  steps/align_si.sh    --nj $nj --cmd "$train_cmd" data/train data/lang exp/tri2b exp/tri2b_ali
  steps/train_sat.sh   --cmd "$train_cmd" 5000 100000 data/train data/lang exp/tri2b_ali exp/tri3b
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/tri3b exp/tri3b_ali
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "=== stage 4: harvest pronunciation (= stress-variant) choices ==="
  # Which stress variant did the audio pick for each spoken token? This is
  # the first HUMAN ground truth for ambiguous words — get_prons.sh dumps
  # per-utterance pronunciations + corpus-wide counts.
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang exp/tri3b_ali
  echo "--- stress splits observed for battery homographs ---"
  for w in вълна пара говори чета коса; do
    grep " $w " exp/tri3b_ali/pron_counts_nowb.txt || true
  done
  echo "Per-utt choices: exp/tri3b_ali/prons.*.gz ; counts: pron_counts_nowb.txt"
  echo "Next (post-milestone-A): segment parliament sessions (steps/cleanup),"
  echo "fold them into data/train, then train the stress-annotation TDNN on"
  echo "the combined ~750 h and decode per-sentence stress graphs."
fi
