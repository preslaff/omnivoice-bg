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
# Production stack (E3 sparse dialect, shipped 2026-07-18):
#   MODEL    = bright T5 recipe refreshed on SPARSE marks (natural prosody,
#              no mark-triggered ч/щ/с over-articulation).
#   ACCENTOR = ft6b (battery 16/25, OOD SER 0.005%).
#   SPARSE_KEEP = marks only homographs + hard words + override pins; the
#              F9-matched dialect for the model above. Empty SPARSE_KEEP =>
#              dense marking (only correct with a dense-trained model).
MODEL="${OMNIVOICE_BG_MODEL:-exp/omnivoice_bg_e3_sparse/checkpoint-3000}"
ACCENTOR="${OMNIVOICE_BG_ACCENTOR:-exp/accentor_ft6b/last.pt}"
SPARSE_KEEP="${OMNIVOICE_BG_SPARSE_KEEP:-config/bg/sparse_keep.json}"
LEXICON="${OMNIVOICE_BG_LEXICON:-data/bg_lexicon/bg_stress.sqlite}"
OVERRIDES="${OMNIVOICE_BG_OVERRIDES:-config/bg/bg_stress_overrides.json}"
HOST="${OMNIVOICE_BG_HOST:-0.0.0.0}"
PORT="${OMNIVOICE_BG_PORT:-8002}"
SEED="${OMNIVOICE_BG_SEED:-42}"
# Optional server-wide fixed voice. Leave REF_AUDIO empty to rely on the
# fine-tuned model's default narrator voice and/or per-request client references.
# Server-wide fixed voice (2026-07-18): the HOMEMAX ad announcer, validated
# for ad narration. Raw master on purpose — the model auto-normalizes its
# LEVEL (kills the scratch), and the deep produced character is DESIRABLE for
# ads, so we do NOT neutral-condition it (that would strip the ad voice).
REF_AUDIO="${OMNIVOICE_BG_REF_AUDIO:-examples/reklama_home_max.wav}"
REF_TEXT="${OMNIVOICE_BG_REF_TEXT:-$(cat examples/reklama_home_max.txt 2>/dev/null)}"
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
[ -n "${SPARSE_KEEP}" ] && ARGS+=(--sparse-keep "${SPARSE_KEEP}")
if [ -n "${REF_AUDIO}" ]; then
  ARGS+=(--ref-audio "${REF_AUDIO}")
  [ -n "${REF_TEXT}" ] && ARGS+=(--ref-text "${REF_TEXT}")
fi

echo "Launching OmniVoice-BG server: model=${MODEL} accentor=${ACCENTOR:-rechko} marking=${SPARSE_KEEP:+sparse} port=${PORT} fixed_voice=${REF_AUDIO:-none}"
# expandable_segments reduces allocator fragmentation on the 16 GB card (harmless).
exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" -m omnivoice.cli.serve "${ARGS[@]}"
