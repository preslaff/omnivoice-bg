#!/usr/bin/env bash
# Nightly backup of the volunteer recordings — the irreproducible artifact
# (raw/ + submissions.jsonl + flags.jsonl). Runs on the VPS from cron.
#
# It does TWO things, so a single disk failure never loses data:
#   1. MIRROR  — rsync an always-current copy to $DEST/mirror (incremental,
#      cheap; the data is append-only so no --delete needed).
#   2. SNAPSHOT — a dated tar.gz in $DEST/snapshots, rotated to the last $KEEP,
#      for point-in-time recovery (e.g. accidental corruption).
#
# Put $DEST on a DIFFERENT disk/volume, or set $REMOTE to copy offsite.
#
# Configure via env (all optional — defaults shown):
#   OMNIVOICE_VOL_DIR     source dir           (/opt/omnivoice/data/bg/imperatives_volunteer)
#   OMNIVOICE_BACKUP_DIR  backup root          (/opt/omnivoice/backups)
#   OMNIVOICE_BACKUP_KEEP snapshots to retain  (14)
#   OMNIVOICE_BACKUP_REMOTE  offsite rsync tgt (user@host:/path) — needs key-based ssh
set -euo pipefail

SRC="${OMNIVOICE_VOL_DIR:-/opt/omnivoice/data/bg/imperatives_volunteer}"
DEST="${OMNIVOICE_BACKUP_DIR:-/opt/omnivoice/backups}"
KEEP="${OMNIVOICE_BACKUP_KEEP:-14}"
REMOTE="${OMNIVOICE_BACKUP_REMOTE:-}"

log() { echo "$(date -Is) $*"; }

[ -d "$SRC" ] || { log "ERROR: source $SRC not found"; exit 1; }
mkdir -p "$DEST/mirror" "$DEST/snapshots"

# 1. incremental mirror (always-current second copy)
rsync -a "$SRC/" "$DEST/mirror/"
log "mirror updated -> $DEST/mirror"

# 2. dated snapshot, written atomically (.partial -> rename), then rotated
ts=$(date +%Y%m%d-%H%M%S)
snap="$DEST/snapshots/imperatives_volunteer_${ts}.tar.gz"
tar -czf "${snap}.partial" -C "$(dirname "$SRC")" "$(basename "$SRC")"
mv "${snap}.partial" "$snap"
log "snapshot -> $snap ($(du -h "$snap" | cut -f1))"

ls -1t "$DEST"/snapshots/imperatives_volunteer_*.tar.gz 2>/dev/null \
    | tail -n +$((KEEP + 1)) | while read -r old; do
  rm -f "$old"; log "rotated out $old"
done

# 3. optional offsite copy of the latest snapshot
if [ -n "$REMOTE" ]; then
  rsync -a "$snap" "$REMOTE"/ && log "mirrored snapshot -> $REMOTE"
fi

log "backup complete"
