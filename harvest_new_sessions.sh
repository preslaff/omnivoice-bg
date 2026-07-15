#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# INCREMENTAL parliament harvest — add labels from NEW crawl sessions without
# retraining the acoustic model or re-touching already-harvested audio.
#
# Rationale (see BG_FINAL_EFFORT_PLAN.md / the "incremental" discussion):
# alignment models saturate at a few hundred hours (finding F2 — the 63 h GMM
# aligned 400 h cleanly), so once milestone A has produced the acoustic model,
# it is a FIXED label factory. More parliament data => more homograph tokens,
# obtained by running only the new sessions through the existing models:
#
#   new sessions -> segment (tri2b_comb seed) -> clean (tri4b) -> hires MFCC
#   -> TDNN align (tdnn_a) -> get_prons + harvest -> tagged labels JSONL
#
# The output is DISJOINT from stress_labels_milestone_a.jsonl (new session ids
# only), so the two stack safely for build_accentor_finetune.py.
#
# Prereqs: milestone A finished (exp/tri2b_comb, exp/tri4b, exp/nnet3/tdnn_a
# exist) and the crawl has banked sessions beyond the last harvest.
#
# Usage:   KALDI_ROOT=~/kaldi bash harvest_new_sessions.sh
#          KALDI_ROOT=~/kaldi INCR_TAG=2026q3 stage=2 bash harvest_new_sessions.sh
set -euo pipefail

stage=${stage:-0}
nj=${nj:-$(nproc)}
seg_nj=${seg_nj:-8}
KALDI_ROOT=${KALDI_ROOT:?set KALDI_ROOT to your kaldi checkout}
repo=$(cd "$(dirname "$0")" && pwd)
work=$repo/exp/kaldi_pilot
kdata=$repo/data/kaldi
INCR_TAG=${INCR_TAG:-$(date +%Y%m%d)}

# Fixed milestone-A models (the label factory — never retrained here).
SEED=exp/tri2b_comb          # segmentation biased-LM decode seed
GMM=exp/tri4b                # cleanup + fMLLR
TDNN=exp/nnet3/tdnn_a        # final alignment for labels

manifest=data/bg_parliament/manifest.jsonl
prev_labels=data/kaldi/stress_labels_milestone_a.jsonl   # + any earlier incr
out_labels=data/kaldi/stress_labels_incr_${INCR_TAG}.jsonl

cd "$work"
[ -e steps ] || ln -s "$KALDI_ROOT/egs/wsj/s5/steps" steps
[ -e utils ] || ln -s "$KALDI_ROOT/egs/wsj/s5/utils" utils
. ./path.sh; . ./cmd.sh
export PATH=$KALDI_ROOT/src/nnet3bin:$PATH

for d in "$SEED" "$GMM" "$TDNN"; do
  [ -d "$d" ] || { echo "ERROR: $d missing — run milestone A first"; exit 1; }
done

# --- stage 0: which sessions are new? (in the manifest, not yet harvested) ---
if [ $stage -le 0 ]; then
  echo "=== stage 0: diff crawl manifest vs already-harvested sessions ==="
  (cd "$repo" && ./venv/bin/python - "$manifest" "$out_labels" \
       data/kaldi/stress_labels_milestone_a.jsonl \
       $(ls data/kaldi/stress_labels_incr_*.jsonl 2>/dev/null) <<'PY'
import json, re, sys, pathlib
manifest, out_labels, *label_files = sys.argv[1:]
done = set()
for lf in label_files:
    p = pathlib.Path(lf)
    if not p.exists():
        continue
    for line in open(p, encoding='utf-8'):
        utt = json.loads(line).get('utt', '')
        m = re.match(r'(plsten_\d+)', utt)          # session id prefix
        if m:
            done.add(m.group(1))
new = []
for line in open(manifest, encoding='utf-8'):
    r = json.loads(line)
    sid = r['id']
    if sid not in done:
        new.append(line)
print(f'harvested sessions: {len(done)};  new: {len(new)}', file=sys.stderr)
if not new:
    sys.exit(9)                                     # nothing to do
mn = pathlib.Path(manifest).with_name('manifest_new.jsonl')
mn.write_text(''.join(new), encoding='utf-8')
print(f'wrote {mn} ({len(new)} sessions)', file=sys.stderr)
PY
  ) || { rc=$?; [ $rc -eq 9 ] && { echo "No new sessions — nothing to harvest."; exit 0; }; exit $rc; }
fi

# --- stage 1: build a Kaldi long-utterance dir for the NEW sessions only ---
if [ $stage -le 1 ]; then
  echo "=== stage 1: prep data/parl_new from the new-sessions manifest ==="
  (cd "$repo" && ./venv/bin/python prep_kaldi_data.py --out data/kaldi \
      --long-set parl_new=data/bg_parliament/manifest_new.jsonl)
  rm -rf data/parl_new && cp -r "$kdata/parl_new" data/parl_new
  utils/fix_data_dir.sh data/parl_new || true
  steps/make_mfcc.sh --nj $seg_nj --cmd "$train_cmd" data/parl_new exp/make_mfcc/parl_new mfcc
  steps/compute_cmvn_stats.sh data/parl_new exp/make_mfcc/parl_new mfcc
  utils/fix_data_dir.sh data/parl_new
fi

# --- stage 2: segment + clean the new sessions (fixed seed/GMM) ---
if [ $stage -le 2 ]; then
  echo "=== stage 2: segment (seed $SEED) + clean (fMLLR $GMM) ==="
  steps/cleanup/segment_long_utterances.sh --nj $seg_nj --cmd "$train_cmd" \
      "$SEED" data/lang data/parl_new data/parl_new_seg exp/segment_parl_new
  utils/fix_data_dir.sh data/parl_new_seg
  awk '{n++; d+=$4-$3} END {printf "new: %d segments, %.1f h retained\n", n, d/3600}' \
      data/parl_new_seg/segments
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/parl_new_seg exp/make_mfcc/parl_new_seg mfcc
  steps/compute_cmvn_stats.sh data/parl_new_seg exp/make_mfcc/parl_new_seg mfcc
  utils/fix_data_dir.sh data/parl_new_seg
  steps/cleanup/clean_and_segment_data.sh --nj $nj --cmd "$train_cmd" \
      data/parl_new_seg data/lang "$GMM" exp/tri4b_cleanup_new data/parl_new_cleaned
  utils/fix_data_dir.sh data/parl_new_cleaned
fi

# --- stage 3: hires MFCC + TDNN align (the label-quality path) ---
if [ $stage -le 3 ]; then
  echo "=== stage 3: hires MFCC + TDNN align ($TDNN) ==="
  [ -f conf/mfcc_hires.conf ] || cat > conf/mfcc_hires.conf <<'EOF'
--use-energy=false
--sample-frequency=16000
--num-mel-bins=40
--num-ceps=40
--low-freq=20
--high-freq=-400
EOF
  utils/copy_data_dir.sh data/parl_new_cleaned data/parl_new_hires
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/parl_new_hires exp/make_mfcc/parl_new_hires mfcc
  steps/compute_cmvn_stats.sh data/parl_new_hires exp/make_mfcc/parl_new_hires mfcc
  utils/fix_data_dir.sh data/parl_new_hires
  steps/nnet3/align.sh --nj $nj --cmd "$train_cmd" --use-gpu false \
      data/parl_new_hires data/lang "$TDNN" exp/nnet3/tdnn_a_ali_new
fi

# --- stage 4: harvest -> tagged, disjoint labels file ---
if [ $stage -le 4 ]; then
  echo "=== stage 4: harvest new labels -> $out_labels ==="
  steps/get_prons.sh --cmd "$train_cmd" \
      data/parl_new_hires data/lang exp/nnet3/tdnn_a_ali_new
  (cd "$repo" && ./venv/bin/python harvest_stress_labels.py \
      --pilot exp/kaldi_pilot --ali exp/nnet3/tdnn_a_ali_new \
      --data data/parl_new_hires \
      --out "$out_labels")
  echo
  echo "Done. New labels: $repo/$out_labels (disjoint from milestone_a — stack them)."
  echo "Next: native spot-vet the new boundary-word tokens (vet_stress_cli.py),"
  echo "then rebuild the Accentor set over ALL stress_labels_*.jsonl and retrain"
  echo "(build_accentor_finetune.py) — acoustic model untouched, ~40 min Accentor."
fi
