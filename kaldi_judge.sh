#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Kaldi stress-judge for the teacher-distill clips: force-align each BgTTS clip
# to its text with the pilot's tri2b (LDA+MLLT) model, using a small lang whose
# TARGET VERBS carry every stress variant, so the aligner PICKS which vowel is
# stressed. teacher_distill.py's `judge --judge kaldi` builds the dict + data
# dirs and parses the resulting prons; this script is the Kaldi half.
#
# Inputs (absolute paths, exported by the caller):
#   DICT   dict dir (lexicon.txt with all-variant prons + copied phone lists)
#   DATA   data dir (wav.scp resampled to 16k, text, utt2spk, spk2utt)
#   OUT    output dir (gets lang/, mfcc/, ali.*.gz, prons.*.gz)
#   MODEL  GMM dir (default exp/kaldi_pilot/exp/tri2b)
#   nj     parallel jobs (caller caps to #clips)
#
# Reuses the pilot scaffolding (steps/ utils/ path.sh cmd.sh conf/) that
# kaldi_pilot.sh already created under exp/kaldi_pilot — run that first if the
# pilot dir is fresh. tri2b needs only align_si (no speaker adaptation), which
# is the right choice for 1-2 s single-utterance "speakers".
set -euo pipefail

KALDI_ROOT=${KALDI_ROOT:-/home/preslaff/kaldi}
repo=$(cd "$(dirname "$0")" && pwd)
work=$repo/exp/kaldi_pilot
model=${MODEL:-$work/exp/tri2b}
dict=${DICT:?export DICT to the judge dict dir}
data=${DATA:?export DATA to the judge data dir}
out=${OUT:?export OUT to the output ali dir}
nj=${nj:-$(nproc)}

[ -d "$work/steps" ] || { echo "run kaldi_pilot.sh first ($work has no steps/)"; exit 1; }
[ -f "$model/final.mdl" ] || { echo "no model at $model/final.mdl"; exit 1; }
export KALDI_ROOT
cd "$work"
. ./path.sh
. ./cmd.sh

mkdir -p "$out"
echo "=== judge stage 1/4: prepare_lang (all-variant verb lexicon) ==="
rm -rf "$out/lang" "$out/lang_tmp"
utils/prepare_lang.sh "$dict" "<unk>" "$out/lang_tmp" "$out/lang"

echo "=== judge stage 2/4: MFCC + CMVN (clips resampled to 16k in wav.scp) ==="
steps/make_mfcc.sh --nj "$nj" "$data" "$out/log_mfcc" "$out/mfcc"
steps/compute_cmvn_stats.sh "$data" "$out/log_mfcc" "$out/mfcc"
utils/fix_data_dir.sh "$data"

echo "=== judge stage 3/4: align_si with $(basename "$model") ==="
steps/align_si.sh --nj "$nj" --cmd "$train_cmd" "$data" "$out/lang" "$model" "$out"

echo "=== judge stage 4/4: get_prons (which stress variant the audio chose) ==="
steps/get_prons.sh --cmd "$train_cmd" "$data" "$out/lang" "$out"
echo "OK -> prons: $out/prons.*.gz ; lang: $out/lang"
