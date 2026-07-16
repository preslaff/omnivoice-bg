#!/bin/bash
# Launch the OmniVoice-BG production TTS server:
#   fine-tuned Bulgarian model + server-side stress accenting + deterministic
#   cloning. See OMNIVOICE_BG.md for the full picture.
#
# Usage:   bash serve_bg.sh
# Override any setting via env var, e.g.:  OMNIVOICE_BG_PORT=9000 bash serve_bg.sh
set -euo pipefail
cd "$(dirname "$0")"

# ====== settings (override via env) ======
MODEL="${OMNIVOICE_BG_MODEL:-exp/omnivoice_bg_ft6b/checkpoint-3000}"
# Contextual stress front-end (milestone-A cycle, shipped 2026-07-17):
# ft6b Accentor — battery 16/25, OOD SER 0.005%; the TTS checkpoint above is
# the F9-consistent refresh trained on THESE marks. Set empty to fall back
# to rechko-lexicon marking.
ACCENTOR="${OMNIVOICE_BG_ACCENTOR:-exp/accentor_ft6b/last.pt}"
LEXICON="${OMNIVOICE_BG_LEXICON:-data/bg_lexicon/bg_stress.sqlite}"
OVERRIDES="${OMNIVOICE_BG_OVERRIDES:-config/bg/bg_stress_overrides.json}"
HOST="${OMNIVOICE_BG_HOST:-0.0.0.0}"
PORT="${OMNIVOICE_BG_PORT:-8002}"
SEED="${OMNIVOICE_BG_SEED:-42}"
# Optional server-wide fixed voice. Leave REF_AUDIO empty to rely on the
# fine-tuned model's default narrator voice and/or per-request client references.
REF_AUDIO="${OMNIVOICE_BG_REF_AUDIO:-}"
REF_TEXT="${OMNIVOICE_BG_REF_TEXT:-}"          # exact transcript of REF_AUDIO
# =========================================

PY=./venv/bin/python

# Sanity checks
[ -d "${MODEL}" ] || [ "${MODEL}" = "k2-fsa/OmniVoice" ] || {
  echo "WARNING: model '${MODEL}' not found locally — will try to download." >&2; }
[ -f "${LEXICON}" ] || { echo "ERROR: lexicon not found: ${LEXICON}" >&2
  echo "Build it: ${PY} build_bg_lexicon.py --dump ../rechko/db.sql --out ${LEXICON}" >&2; exit 1; }

# Warn if something is already on the port
if command -v ss >/dev/null && ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q ":${PORT}"; then
  echo "WARNING: port ${PORT} already in use — stop the old server first (pkill -f omnivoice.cli.serve)." >&2
fi

ARGS=(--model "${MODEL}" --lexicon "${LEXICON}" --overrides "${OVERRIDES}"
      --host "${HOST}" --port "${PORT}" --seed "${SEED}")
[ -n "${ACCENTOR}" ] && ARGS+=(--accentor "${ACCENTOR}")
if [ -n "${REF_AUDIO}" ]; then
  ARGS+=(--ref-audio "${REF_AUDIO}")
  [ -n "${REF_TEXT}" ] && ARGS+=(--ref-text "${REF_TEXT}")
fi

echo "Launching OmniVoice-BG server: model=${MODEL} accentor=${ACCENTOR:-rechko} port=${PORT} fixed_voice=${REF_AUDIO:-none}"
# expandable_segments reduces allocator fragmentation on the 16 GB card (harmless).
exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -m omnivoice.cli.serve "${ARGS[@]}"
