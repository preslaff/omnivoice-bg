#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Pre-flight wrapper for the incremental parliament harvest.
#
# WHY THIS EXISTS
# ---------------
# On 2026-07-28 the box hard-froze at ~07:07 during the stage-2 CPU lattice
# decode. It was NOT a resource problem — 18 GB RAM free, swap idle, iowait
# ~0 %, load actually *falling* as jobs finished — and nothing was captured
# (no OOM line, no kernel panic in pstore, no GPU/thermal/MCE log). An abrupt,
# unlogged, total hang like that is a platform-level lockup (kernel/driver/
# hardware), incidental to the harvest. The suspect is the new kernel
# (7.0.0-28-generic, first booted Jul 23); the prior 6.17.0-* kernels have
# long clean-uptime records.
#
# So the real fix is to make the NEXT run diagnosable and its progress durable:
#   1. warn (and offer to abort) if we're on the suspect kernel;
#   2. run a hardware watchdog (temps / power / mem / load every 30 s) alongside
#      the harvest, so a freeze leaves a temperature+power trail sar never had;
#   3. launch the harvest inside tmux with a top-level tee'd log, so an SSH drop
#      doesn't kill it and we get the harvest's own progress trail next time.
#
# The harvest itself is resumable by stage (Kaldi checkpoints per stage). After
# the 2026-07-28 freeze, stage 4 never ran (no labels written, corpus intact) —
# resume with stage=2.
#
# For the ONE-TIME host setup that captures the *cause* of a future freeze
# (kdump / pstore / hung-task panic / netconsole), see
# docs/HARVEST_CRASH_CAPTURE.md.
#
# USAGE
# -----
#   KALDI_ROOT=~/kaldi ./run_harvest.sh              # preflight -> launch (stage 0)
#   KALDI_ROOT=~/kaldi stage=2 ./run_harvest.sh      # resume the interrupted run
#   KALDI_ROOT=~/kaldi ./run_harvest.sh check        # preflight only, no launch
#
# Pass-through env (same contract as harvest_new_sessions.sh):
#   KALDI_ROOT (required), stage, nj, seg_nj, INCR_TAG
# Wrapper env:
#   HARVEST_TMUX          tmux session name           (default: harvest)
#   WATCH_INTERVAL        watchdog sample seconds      (default: 30)
#   ALLOW_SUSPECT_KERNEL  =1 to skip the kernel prompt (for non-interactive use)
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
HARVEST="$REPO/harvest_new_sessions.sh"
SESSION=${HARVEST_TMUX:-harvest}
WATCH_INTERVAL=${WATCH_INTERVAL:-30}
SUSPECT_KERNEL="7.0.0-28-generic"

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
RUNLOG="$LOGDIR/harvest_incr_$TS.log"
HWLOG="$LOGDIR/harvest_hw_$TS.log"
HWPIDFILE="$LOGDIR/harvest_hw_$TS.pid"
LAUNCHER="$LOGDIR/.harvest_cmd_$TS.sh"

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_dim=$'\033[2m'; c_0=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '  %s✓%s %s\n' "$c_grn" "$c_0" "$*"; }
warn() { printf '  %s!%s %s\n' "$c_yel" "$c_0" "$*"; }
die()  { printf '  %s✗ %s%s\n' "$c_red" "$*" "$c_0" >&2; exit 1; }

# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
preflight() {
  say "${c_dim}=== pre-flight ===${c_0}"

  [ -n "${KALDI_ROOT:-}" ] || die "KALDI_ROOT is not set (e.g. KALDI_ROOT=~/kaldi)"
  [ -d "$KALDI_ROOT" ]     || die "KALDI_ROOT=$KALDI_ROOT does not exist"
  ok "KALDI_ROOT = $KALDI_ROOT"

  [ -f "$HARVEST" ] || die "harvest_new_sessions.sh not found next to this wrapper"
  ok "harvest script present"

  command -v tmux >/dev/null || die "tmux not installed (apt install tmux)"
  ok "tmux $(tmux -V | awk '{print $2}')"

  # Not already running?
  if pgrep -af 'harvest_new_sessions|segment_long_utterances|gmm-latgen-faster' \
       | grep -qv "$0"; then
    die "a harvest already appears to be running — check: tmux ls ; pgrep -af gmm-latgen"
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    die "tmux session '$SESSION' already exists — attach: tmux attach -t $SESSION"
  fi
  ok "no harvest currently running"

  # Label-factory models present (fail fast before a 7 h decode).
  local work="$REPO/exp/kaldi_pilot"
  local missing=0
  for d in exp/tri2b_comb exp/tri4b exp/nnet3/tdnn_a; do
    [ -d "$work/$d" ] || { warn "missing $work/$d"; missing=1; }
  done
  [ "$missing" = 0 ] && ok "Kaldi label factory present (tri2b_comb / tri4b / tdnn_a)" \
                     || die "milestone-A models missing — run milestone A first"

  # Disk headroom.
  local avail_kb; avail_kb=$(df --output=avail "$REPO" | tail -1)
  local avail_gb=$(( avail_kb / 1024 / 1024 ))
  if [ "$avail_gb" -lt 50 ]; then
    warn "only ${avail_gb} GB free on $REPO — stage-2 lattices need room"
  else
    ok "disk headroom ${avail_gb} GB"
  fi

  # Watchdog tooling (non-fatal — the watchdog degrades gracefully).
  command -v sensors    >/dev/null && ok "lm-sensors present"  || warn "no lm-sensors (apt install lm-sensors) — CPU/VRM temps won't be logged"
  command -v nvidia-smi >/dev/null && ok "nvidia-smi present"  || warn "no nvidia-smi — GPU temps won't be logged"

  # The main event: kernel check.
  local kr; kr=$(uname -r)
  if [ "$kr" = "$SUSPECT_KERNEL" ]; then
    warn "running the SUSPECT kernel $kr (the 2026-07-28 freeze was on this kernel)"
    say  "      A known-good kernel is installed. To switch: reboot, hold Shift for GRUB,"
    say  "      Advanced options -> pick 6.17.0-* , then re-run this wrapper."
    if [ "${ALLOW_SUSPECT_KERNEL:-0}" != 1 ]; then
      if [ -t 0 ]; then
        printf '  %sProceed on this kernel anyway? [y/N] %s' "$c_yel" "$c_0"
        read -r reply
        case "$reply" in [yY]*) warn "proceeding on suspect kernel at your request";;
                          *) die "aborted — reboot into a 6.17.0-* kernel and re-run";; esac
      else
        die "on suspect kernel and non-interactive — set ALLOW_SUSPECT_KERNEL=1 to override"
      fi
    else
      warn "ALLOW_SUSPECT_KERNEL=1 — proceeding on suspect kernel"
    fi
  else
    ok "kernel $kr (not the suspect kernel)"
  fi
}

# --------------------------------------------------------------------------
# Hardware watchdog — one timestamped block per interval.
# Captures exactly what sar could not: CPU/GPU temps + GPU power, alongside a
# mem/load backstop, so the last block before a freeze names the conditions.
# --------------------------------------------------------------------------
start_watchdog() {
  {
    say "# OmniVoice-BG harvest hardware watchdog"
    say "# started $(date '+%F %T')  interval=${WATCH_INTERVAL}s  kernel=$(uname -r)"
    say "# columns per block: time | load | mem | cpu-temps | gpu(temp,power)"
  } > "$HWLOG"

  (
    while :; do
      {
        printf '=== %s ===\n' "$(date '+%F %T')"
        printf 'load : %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
        printf 'mem  : %s\n' "$(free -m | awk '/^Mem:/{printf "used %s / %s MB, avail %s MB", $3,$2,$7}')"
        if command -v sensors >/dev/null; then
          sensors 2>/dev/null | grep -iE 'Package id|Tctl|Tdie|Composite|Core [0-9]' \
            | sed 's/^/cpu  : /'
        fi
        if command -v nvidia-smi >/dev/null; then
          nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used \
                     --format=csv,noheader,nounits 2>/dev/null \
            | sed 's/^/gpu  : /'
        fi
      } >> "$HWLOG" 2>&1
      sleep "$WATCH_INTERVAL"
    done
  ) &
  echo $! > "$HWPIDFILE"
  ok "watchdog PID $(cat "$HWPIDFILE") -> $(basename "$HWLOG") (every ${WATCH_INTERVAL}s)"
}

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
preflight

if [ "${1:-}" = "check" ]; then
  say "${c_grn}pre-flight OK${c_0} — 'check' mode, nothing launched."
  exit 0
fi

start_watchdog

# A generated launcher keeps quoting sane inside tmux, and guarantees the
# watchdog is stopped the instant the harvest exits (success, error, or Ctrl-C).
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
export KALDI_ROOT="$KALDI_ROOT"
${stage:+export stage="$stage"}
${nj:+export nj="$nj"}
${seg_nj:+export seg_nj="$seg_nj"}
${INCR_TAG:+export INCR_TAG="$INCR_TAG"}

trap 'kill "\$(cat "$HWPIDFILE" 2>/dev/null)" 2>/dev/null || true' EXIT

echo "=== harvest launched $(date "+%F %T") | kernel $(uname -r) | stage=\${stage:-0} nj=\${nj:-auto} seg_nj=\${seg_nj:-8} ==="
set -o pipefail
bash "$HARVEST" 2>&1 | tee "$RUNLOG"
rc=\${PIPESTATUS[0]}
echo "=== harvest exited rc=\$rc at $(date "+%F %T") ==="
echo "\$rc" > "$LOGDIR/harvest_last_rc_$TS"
# keep the pane open so the exit status is visible on re-attach
echo "(press any key to close this tmux window)"; read -r -n1 || true
EOF
chmod +x "$LAUNCHER"

tmux new-session -d -s "$SESSION" "bash '$LAUNCHER'"
ok "harvest running in tmux session '$SESSION'"

say ""
say "${c_grn}launched.${c_0}"
say "  watch live   : ${c_dim}tmux attach -t $SESSION${c_0}    (detach: Ctrl-b then d)"
say "  harvest log  : ${c_dim}tail -f $RUNLOG${c_0}"
say "  hardware log : ${c_dim}tail -f $HWLOG${c_0}"
say "  stop it      : ${c_dim}tmux kill-session -t $SESSION${c_0}  (watchdog stops with it)"
say ""
say "  If it freezes again, the LAST block in the hardware log names the temps/"
say "  power at the wall — that's the evidence we were missing on 2026-07-28."
