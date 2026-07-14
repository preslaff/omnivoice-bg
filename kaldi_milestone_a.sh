#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# MILESTONE A — the ONE combined retrain + the annotation TDNN (guide §6a-6c).
# Run when the 2020-2026 crawl has banked ~500 h raw parliament AND the full
# bg400 audiobook ingest is done. This is the paper's §3 at scale:
#
#   1. re-segment the FULL parliament corpus with the in-domain round-2 seed
#   2. retrain the GMM ladder ONCE on everything (audiobooks ~400 h +
#      parliament segments ~"the paper's 749 h recipe")  -> tri4b
#   3. clean_and_segment pass (paper's data cleanup)     -> train_cleaned
#   4. train the ANNOTATION TDNN (nnet3, cross-entropy, hires MFCC)
#   5. TDNN alignment + get_prons + harvest              -> the trusted labels
#
# Design notes:
#   - CE (non-chain) TDNN on purpose: chain models subsample frames 3x, which
#     would silently break get_prons/harvest timestamps; the CE model aligns
#     at the GMM's 10 ms frame rate and is a drop-in for the whole harvest
#     tool chain. Annotation needs trustworthy ALIGNMENT, not decode WER.
#   - Acceptance test (guide F5): the GMM final-stress bias on prosodically
#     weak words (може/всички/съюз...) must FLIP to the canonical majority in
#     the TDNN's pron counts — otherwise the TDNN inherited the bias and the
#     tier-2 exclusion list stays in force. Stage 7 prints this verdict.
#
# PREREQUISITES:
#   - kaldi_pilot.sh + kaldi_parliament_pilot.sh completed (needs
#     exp/tri3b_comb + exp/tri2b_comb in exp/kaldi_pilot)
#   - a CUDA Kaldi build for stage 6+ (the current CPU build trains a 750 h
#     TDNN in weeks, not days). CUDA 13.2 is unsupported by Kaldi — install a
#     CUDA 12.x toolkit alongside and rebuild:
#       cd $KALDI_ROOT/src && ./configure --shared --use-cuda=yes \
#           --cudatk-dir=/usr/local/cuda-12.4 && make -j $(nproc)
#     Stages 0-5 (GMM territory) run fine on the CPU build.
#
# Memory (60 GB box — guide Pitfall 1): segmentation stays at seg_nj=8 (one
# session transcript per biased-LM job); the GMM ladder and MFCC use full nj
# (dict is vocab-limited); TDNN egs generation is bounded via --max-jobs-run.
#
# Usage:   KALDI_ROOT=~/kaldi bash kaldi_milestone_a.sh
# Resume:  KALDI_ROOT=~/kaldi stage=N bash kaldi_milestone_a.sh

set -euo pipefail

stage=${stage:-0}
stop_stage=${stop_stage:-99}
nj=${nj:-$(nproc)}
seg_nj=${seg_nj:-8}                    # memory-bound (guide Pitfall 1)
egs_jobs=${egs_jobs:-8}                # egs generation is IO+RAM heavy
min_parl_hours=${min_parl_hours:-400}  # volume gate (milestone A ~ 500 h raw)
KALDI_ROOT=${KALDI_ROOT:?set KALDI_ROOT to your kaldi checkout}
repo=$(cd "$(dirname "$0")" && pwd)
work=$repo/exp/kaldi_pilot             # SAME workdir as the pilots (reuses data/, mfcc, GMMs)
kdata=$repo/data/kaldi

[ -f "$work/exp/tri3b_comb/final.mdl" ] || {
  echo "ERROR: run kaldi_parliament_pilot.sh through stage 4 first"; exit 1; }

cd "$work"
. ./path.sh; . ./cmd.sh

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "=== stage 0: volume gate + refresh data dirs (full bg400 + full crawl) ==="
  hours=$(cd "$repo" && ./venv/bin/python - <<'EOF'
import json
print(int(sum(float(json.loads(l)["duration"])
              for l in open("data/bg_parliament/manifest.jsonl")) / 3600))
EOF
)
  echo "parliament banked: ${hours} h raw (gate: >= ${min_parl_hours} h)"
  [ "$hours" -ge "$min_parl_hours" ] || {
    echo "GATE FAILED — let the crawl run further, or override min_parl_hours"; exit 1; }
  # Rebuild all Kaldi data dirs + the vocab-limited dict over the FULL corpora.
  (cd "$repo" && ./venv/bin/python prep_kaldi_data.py --out data/kaldi \
      --set sivov=data/bg/train.jsonl \
      --set bg400=data/bg400/train.jsonl \
      --long-set parl=data/bg_parliament/manifest.jsonl)
  rm -rf data/local data/lang
  cp -r "$kdata/local" data/
  rm -rf data/parl && cp -r "$kdata/parl" data/parl
  utils/fix_data_dir.sh data/parl || true      # long utts: no feats yet, ok
  # Same phone set -> tri2b_comb/tri3b_comb stay compatible; words.txt grows.
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "=== stage 1: MFCC for the full audiobook corpus (~400+ h) ==="
  rm -rf data/train_aud
  utils/combine_data.sh data/train_aud "$kdata/sivov" "$kdata/bg400"
  utils/fix_data_dir.sh data/train_aud
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/train_aud exp/make_mfcc/train_aud mfcc
  steps/compute_cmvn_stats.sh data/train_aud exp/make_mfcc/train_aud mfcc
  utils/fix_data_dir.sh data/train_aud
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "=== stage 2: segment the FULL parliament crawl (seed = tri2b_comb) ==="
  steps/make_mfcc.sh --nj $seg_nj --cmd "$train_cmd" data/parl exp/make_mfcc/parl mfcc
  steps/compute_cmvn_stats.sh data/parl exp/make_mfcc/parl mfcc
  utils/fix_data_dir.sh data/parl
  steps/cleanup/segment_long_utterances.sh --nj $seg_nj --cmd "$train_cmd" \
      exp/tri2b_comb data/lang data/parl data/parl_seg3 exp/segment_parl3
  utils/fix_data_dir.sh data/parl_seg3
  echo "--- segmentation yield (round-2 benchmark: 79%) ---"
  awk '{n++; d+=$4-$3} END {printf "%d segments, %.1f h retained\n", n, d/3600}' \
      data/parl_seg3/segments
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "=== stage 3: THE combined GMM retrain (audiobooks + parliament) -> tri4b ==="
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/parl_seg3 exp/make_mfcc/parl_seg3 mfcc
  steps/compute_cmvn_stats.sh data/parl_seg3 exp/make_mfcc/parl_seg3 mfcc
  utils/fix_data_dir.sh data/parl_seg3
  utils/combine_data.sh data/train_all data/train_aud data/parl_seg3
  utils/fix_data_dir.sh data/train_all
  # Seed alignments from the best existing model, then one LDA+MLLT + SAT
  # pass sized for ~750 h (leaf/gauss counts scaled up from the pilots).
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train_all data/lang exp/tri3b_comb exp/tri3b_comb_ali_all
  steps/train_lda_mllt.sh --cmd "$train_cmd" 6000 90000 \
      data/train_all data/lang exp/tri3b_comb_ali_all exp/tri2b_a
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/train_all data/lang exp/tri2b_a exp/tri2b_a_ali
  steps/train_sat.sh --cmd "$train_cmd" 7000 150000 \
      data/train_all data/lang exp/tri2b_a_ali exp/tri4b
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train_all data/lang exp/tri4b exp/tri4b_ali_all
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "=== stage 4: cleanup pass (paper's clean_and_segment) -> train_cleaned ==="
  # Removes bad transcript regions the biased-LM segmentation let through;
  # the TDNN trains on the cleaned set only.
  steps/cleanup/clean_and_segment_data.sh --nj $nj --cmd "$train_cmd" \
      data/train_all data/lang exp/tri4b exp/tri4b_cleanup data/train_cleaned
  utils/fix_data_dir.sh data/train_cleaned
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train_cleaned data/lang exp/tri4b exp/tri4b_ali_cleaned
  echo "--- retained after cleanup ---"
  awk '{n++; d+=$4-$3} END {printf "%d utts, %.1f h\n", n, d/3600}' \
      data/train_cleaned/segments 2>/dev/null || \
      wc -l < data/train_cleaned/utt2spk
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "=== stage 5: GMM-level noise meter + F5 bias report (informational) ==="
  steps/get_prons.sh --cmd "$train_cmd" data/train_cleaned data/lang exp/tri4b_ali_cleaned
  echo "--- noise meter: minority rate on known single-stress words ---"
  for w in тези или това бъде вече време има между; do
    grep " $w " exp/tri4b_ali_cleaned/pron_counts_nowb.txt || true
  done
  echo "--- F5 bias words (GMM is EXPECTED to still misjudge these) ---"
  for w in може всички съюз още един една колеги; do
    grep " $w " exp/tri4b_ali_cleaned/pron_counts_nowb.txt || true
  done
  echo "(the TDNN in stage 7 must flip these to the canonical majority)"
fi

# ---------------------------------------------------------------------------
# TDNN territory — needs the CUDA Kaldi build (see header).
# ---------------------------------------------------------------------------

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "=== stage 6: annotation TDNN (nnet3 CE, hires MFCC, GPU) ==="
  # cuda-compiled lives in nnet3bin, which the workspace path.sh omits.
  "$KALDI_ROOT/src/nnet3bin/cuda-compiled" || {
    echo "ERROR: this Kaldi build has no CUDA — rebuild with --use-cuda=yes"
    echo "       (CUDA 13.1 works: local configure patch adds the 13_* case;"
    echo "        use --cuda-arch=\"-gencode arch=compute_120,code=sm_120\")"; exit 1; }
  export PATH=$KALDI_ROOT/src/nnet3bin:$PATH
  cat > conf/mfcc_hires.conf <<'EOF'
--use-energy=false
--sample-frequency=16000
--num-mel-bins=40
--num-ceps=40
--low-freq=20
--high-freq=-400
EOF
  utils/copy_data_dir.sh data/train_cleaned data/train_cleaned_hires
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/train_cleaned_hires exp/make_mfcc/train_cleaned_hires mfcc
  steps/compute_cmvn_stats.sh data/train_cleaned_hires exp/make_mfcc/train_cleaned_hires mfcc
  utils/fix_data_dir.sh data/train_cleaned_hires

  # TDNN xconfig — modest (~10M params): the job is alignment fidelity on
  # in-domain speech, not open-vocabulary decoding. Splicing follows the
  # classic librispeech nnet3 tdnn recipe.
  tdnn_dir=exp/nnet3/tdnn_a
  mkdir -p $tdnn_dir/configs
  num_targets=$(tree-info exp/tri4b_ali_cleaned/tree | grep num-pdfs | awk '{print $2}')
  cat > $tdnn_dir/configs/network.xconfig <<EOF
  input dim=40 name=input
  relu-batchnorm-layer name=tdnn1 dim=768 input=Append(-2,-1,0,1,2)
  relu-batchnorm-layer name=tdnn2 dim=768 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 dim=768 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn4 dim=768 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn5 dim=768 input=Append(-6,-3,0)
  relu-batchnorm-layer name=tdnn6 dim=768
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py \
      --xconfig-file $tdnn_dir/configs/network.xconfig --config-dir $tdnn_dir/configs
  steps/nnet3/train_dnn.py --stage=-10 \
      --cmd="$train_cmd" \
      --feat.cmvn-opts="--norm-means=true --norm-vars=false" \
      --trainer.num-epochs 3 \
      --trainer.optimization.num-jobs-initial 1 \
      --trainer.optimization.num-jobs-final 1 \
      --trainer.optimization.initial-effective-lrate 0.0015 \
      --trainer.optimization.final-effective-lrate 0.00015 \
      --trainer.optimization.minibatch-size 256,128 \
      --egs.dir "" --egs.opts "--max-jobs-run $egs_jobs" \
      --use-gpu=true \
      --feat-dir data/train_cleaned_hires \
      --ali-dir exp/tri4b_ali_cleaned \
      --lang data/lang \
      --dir $tdnn_dir
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  echo "=== stage 7: TDNN alignment + F5 ACCEPTANCE TEST ==="
  # CE nnet3 aligns at the same 10 ms frame rate as the GMMs -> get_prons and
  # the harvester work unchanged (this is why the model is not a chain model).
  steps/nnet3/align.sh --nj $nj --cmd "$train_cmd" --use-gpu false \
      data/train_cleaned_hires data/lang exp/nnet3/tdnn_a exp/nnet3/tdnn_a_ali
  steps/get_prons.sh --cmd "$train_cmd" \
      data/train_cleaned_hires data/lang exp/nnet3/tdnn_a_ali
  echo "--- ACCEPTANCE (guide F5): canonical variant must now be the clear ---"
  echo "--- majority; otherwise the TDNN inherited the GMM bias and the    ---"
  echo "--- tier-2 exclusion list STAYS.  (canonical: мо́же вси́чки съю́з ...) ---"
  for w in може всички съюз вече още един една или колеги между; do
    grep " $w " exp/nnet3/tdnn_a_ali/pron_counts_nowb.txt || true
  done
  echo "--- battery homograph splits (TDNN) ---"
  for w in вълна пара говори чета коса прави носи учи ходи моли купи; do
    grep " $w " exp/nnet3/tdnn_a_ali/pron_counts_nowb.txt || true
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  echo "=== stage 8: harvest the milestone-A stress labels ==="
  (cd "$repo" && ./venv/bin/python harvest_stress_labels.py \
      --pilot exp/kaldi_pilot --ali exp/nnet3/tdnn_a_ali \
      --data data/train_cleaned_hires \
      --out data/kaldi/stress_labels_milestone_a.jsonl)
  echo "Next: native spot-vet ~20 tokens of the F5 words (vet_stress_cli.py),"
  echo "then rebuild the Accentor fine-tune set with the new labels"
  echo "(build_accentor_finetune.py) — if the F5 test passed, tier-2 words"
  echo "can be promoted to tier-1 supervision."
fi
