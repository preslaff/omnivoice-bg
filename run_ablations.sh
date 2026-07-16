#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Ablation study for the paper (§6.x): three subtractive cells against the
# full ft6b recipe. The other table rows already exist as generations
# (text-only=accentor_text3 11/25; no-veto=ft5-era; +veto=ft5b 15/25;
# no-oversample=ft6 15/25; full=ft6b 16/25).
#
#   A1  -teacher      ft6b recipe without --extra-marked (BgGPT rows)
#   A2  -veto         ft6b recipe with an empty vet-results file
#   A3  -parliament   ft6b recipe with plsten_* label rows removed
#
# Each cell: build (~5 min) -> 20K-step train (~57 min GPU) -> battery + OOD.
# Total ~3.5 h sequential; run in tmux overnight. Evals use last.pt for every
# cell (the ft6b serve choice) — consistency across cells beats per-cell
# optimum. Summary lands in exp/ablations/summary.txt.
#
# Usage:  tmux new -s abl; bash run_ablations.sh
#         only=A2 bash run_ablations.sh          # single cell
set -euo pipefail
cd "$(dirname "$0")"

only=${only:-}
LABELS=data/kaldi/stress_labels_milestone_a.jsonl
VET=data/kaldi/vetting_milestone_a.jsonl
TEACHER=data/bggpt_teacher/teacher_marked.jsonl
mkdir -p exp/ablations data/ablations

run_cell () {  # name, build-args...
  local name=$1; shift
  [ -n "$only" ] && [ "$only" != "$name" ] && return 0
  echo "=== ablation $name: build ==="
  ./venv/bin/python build_accentor_finetune.py "$@" \
      --oversample-minority 16 --out "data/ablations/${name}.jsonl"
  echo "=== ablation $name: train (20K steps) ==="
  ./venv/bin/python train_accentor.py --data "data/ablations/${name}.jsonl" \
      --init exp/accentor_text3/best.pt --out "exp/ablations/${name}" \
      --steps 20000 --batch-size 256
  echo "=== ablation $name: eval ==="
  ./venv/bin/python eval_accentor.py --ckpt "exp/ablations/${name}/last.pt" \
      --battery config/bg/homograph_battery_sentences.json \
      > "exp/ablations/${name}.battery.txt" 2>&1
  ./venv/bin/python eval_accentor_ood.py --ckpt "exp/ablations/${name}/last.pt" \
      > "exp/ablations/${name}.ood.txt" 2>&1
  grep -h "battery:" "exp/ablations/${name}.battery.txt" | tail -1
  grep -h "stress error rate" "exp/ablations/${name}.ood.txt" | head -1
}

# A1 — no teacher rows (isolates the BgGPT sense-teacher contribution)
run_cell A1_no_teacher \
    --labels "$LABELS" --vet-results "$VET"

# A2 — no vetting veto (isolates the per-token native veto)
: > data/ablations/empty_vet.jsonl
run_cell A2_no_veto \
    --labels "$LABELS" --vet-results data/ablations/empty_vet.jsonl \
    --extra-marked "$TEACHER"

# A3 — no parliament labels (isolates the 41% parliament share, §4.2.1)
if [ "$only" = "" ] || [ "$only" = "A3_no_parliament" ]; then
  ./venv/bin/python - <<'PY'
import json
n=k=0
with open('data/kaldi/stress_labels_milestone_a.jsonl', encoding='utf-8') as f, \
     open('data/ablations/labels_no_parl.jsonl', 'w', encoding='utf-8') as o:
    for line in f:
        n += 1
        if not json.loads(line)['utt'].startswith('plsten'):
            o.write(line); k += 1
print(f'labels: {n:,} -> {k:,} after removing parliament rows')
PY
fi
run_cell A3_no_parliament \
    --labels data/ablations/labels_no_parl.jsonl --vet-results "$VET" \
    --extra-marked "$TEACHER"

echo
echo "=== ABLATION SUMMARY ===" | tee exp/ablations/summary.txt
{
  echo "full ft6b (reference):      battery 16/25   OOD SER 0.005%"
  echo "no-oversample = ft6:        battery 15/25   (already trained)"
  echo "text-only = accentor_text3: battery 11/25   (already trained)"
  for n in A1_no_teacher A2_no_veto A3_no_parliament; do
    b=$(grep -h "battery:" "exp/ablations/${n}.battery.txt" 2>/dev/null | tail -1)
    s=$(grep -h "stress error rate" "exp/ablations/${n}.ood.txt" 2>/dev/null | head -1)
    echo "$n: $b | $s"
  done
} | tee -a exp/ablations/summary.txt
echo "battery details per cell: exp/ablations/*.battery.txt"
echo "Key probes to read manually: говори́/прави́ imperatives in A1 (teacher"
echo "causality), чета/купи inversion in A2 (veto), OOD SER in A3 (parliament)."
