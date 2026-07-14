#!/usr/bin/env bash
# Tokenize the FULL ~400 h bg400 corpus for TTS training (guide 6f scale-up).
#
# Why: the ft4->ft5b A/B showed marking consistency is saturated — the
# remaining stress-rendering imprecision is an ACOUSTIC DATA VOLUME problem
# (TTS trained on 63 h while 400 h of the same narrator sits on disk).
#
# Output: data/bg400full/tokens/{train,dev}/ — a NEW dir; the old 50 h
# data/bg400/tokens/ stays untouched (rollback safety). The old subset's
# 27K clips are re-tokenized inside the full run (~15% redundant GPU work,
# accepted for uniformity).
#
# GPU time: Higgs codec encode of ~400 h — expect several hours to overnight
# on the 16 GB card. Run when nothing else needs the GPU.
#
# Usage:   bash tokenize_bg400_full.sh            # all stages
# Resume:  stage=N bash tokenize_bg400_full.sh    # 0=train 1=dev 2=mark+manifests
set -euo pipefail
cd "$(dirname "$0")"

stage=${stage:-0}
stop_stage=${stop_stage:-99}
out=data/bg400full/tokens

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "=== stage 0: tokenize TRAIN (193,768 clips — the long job) ==="
  mkdir -p $out/train/audios $out/train/txts
  CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m omnivoice.scripts.extract_audio_tokens \
      --input_jsonl data/bg400/train.jsonl \
      --tar_output_pattern $out/train/audios/shard-%06d.tar \
      --jsonl_output_pattern $out/train/txts/shard-%06d.jsonl \
      --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
      --nj_per_gpu 2 --shuffle True --skip_errors \
      --min_length 1.0 --max_length 25.0
  echo "train errors: $(wc -l < $out/train/errors.jsonl 2>/dev/null || echo 0)"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "=== stage 1: tokenize DEV (1,933 clips) ==="
  mkdir -p $out/dev/audios $out/dev/txts
  CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m omnivoice.scripts.extract_audio_tokens \
      --input_jsonl data/bg400/dev.jsonl \
      --tar_output_pattern $out/dev/audios/shard-%06d.tar \
      --jsonl_output_pattern $out/dev/txts/shard-%06d.jsonl \
      --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
      --nj_per_gpu 2 --shuffle True --skip_errors \
      --min_length 1.0 --max_length 25.0
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "=== stage 2: mark with the SERVED Accentor + build marked manifests ==="
  # F9 marking consistency: same checkpoint as serve.py --accentor.
  ./venv/bin/python mark_tts_txts.py --ckpt exp/accentor_ft5b/best.pt \
      --overrides config/bg/bg_stress_overrides.json \
      --dirs $out/train $out/dev
  for d in $out/train $out/dev; do
    sed 's#/txts/#/txts_marked/#' $d/data.lst > $d/data_marked.lst
    echo "$d: $(wc -l < $d/data.lst) shards, marked manifest written"
  done
  echo "done — train with config/bg/{train,data}_config_bg_full.json"
fi
