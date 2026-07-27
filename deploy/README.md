# Deploying the volunteer collector at `omnivoice.cdsv.dev`

The volunteer recording app (`omnivoice/cli/collect.py`) runs on the VPS behind
nginx with TLS. It needs **no GPU and no ML dependencies** — just FastAPI +
uvicorn + python-multipart, ffmpeg (optional), and the two prompt text files.

## Topology

```
volunteer browser ──HTTPS──> nginx (omnivoice.cdsv.dev) ──http──> 127.0.0.1:8003 (collect.py)
                                                                        │
                                                       data/bg/imperatives_volunteer/
                                                         raw/<speaker>/*.webm + submissions.jsonl
```

Recorded clips land on the **VPS**. Periodically pull them to the GPU box for
ingestion (`convert_volunteer.py` → tokenize → ft9).

## Steps

1. **DNS** — point `omnivoice.cdsv.dev` (A/AAAA) at the VPS.
2. **App** — install & start the collector (see `omnivoice-collect.service` for
   the exact commands). It binds to `127.0.0.1:8003`, reachable only via nginx.
3. **nginx + TLS** — install `omnivoice.cdsv.dev.nginx.conf`, then
   `sudo certbot --nginx -d omnivoice.cdsv.dev`. **HTTPS is mandatory** — the
   mic (getUserMedia) is blocked on plain http.
4. **Verify** — open `https://omnivoice.cdsv.dev/`, allow the mic, record one
   clip; check `https://omnivoice.cdsv.dev/health` and that a file appears under
   `raw/`. `curl https://omnivoice.cdsv.dev/stats` shows coverage.

## Where recordings are stored on the server

Everything lands under the collector's `--out-dir`, which the systemd unit sets
to `/opt/omnivoice/data/bg/imperatives_volunteer/`:

```
/opt/omnivoice/data/bg/imperatives_volunteer/
├── raw/
│   ├── mila_v/
│   │   ├── imp_001350_t0.webm      # one file per take
│   │   ├── imp_001350_t1.webm
│   │   └── …
│   └── ivan_p/ …                    # one subdir per speaker
├── submissions.jsonl                # one row per take: speaker, prompt_id, marked
│                                     # text (ground truth), duration, accepted, …
└── flags.jsonl                      # sentences volunteers flagged as wrong
```

- **Format:** raw browser audio (webm/opus). Decoding to 24 kHz WAV happens later
  on the GPU box (`convert_volunteer.py`), **not** on the VPS.
- **The index is `submissions.jsonl`** — it ties each file to its ground-truth
  marked text. `raw/` + `submissions.jsonl` + `flags.jsonl` are the complete,
  irreproducible artifact; back them up.
- **Size is small:** ~15–30 KB per 1–2 s clip. 100 volunteers × ~1,640 prompts ×
  2 takes ≈ a few GB. No storage concern.
- **Permissions:** the service runs as `www-data`, so the dir must be writable by
  it — create it once and hand it over:
  ```bash
  sudo mkdir -p /opt/omnivoice/data/bg/imperatives_volunteer
  sudo chown -R www-data:www-data /opt/omnivoice/data
  ```
- **Relocating:** to store on a mounted volume, change only the `--out-dir` line
  in `omnivoice-collect.service` (e.g. `/mnt/data/omnivoice/imperatives_volunteer`);
  nothing else depends on the path.

## Nightly backup (cron)

`backup_volunteer.sh` runs from cron and keeps the recordings off a single disk:

1. **Mirror** — incremental `rsync` of an always-current copy to
   `$OMNIVOICE_BACKUP_DIR/mirror` (append-only data, so cheap; no `--delete`).
2. **Snapshots** — a dated `tar.gz` in `.../snapshots`, rotated to the last
   `$OMNIVOICE_BACKUP_KEEP` (default 14), for point-in-time recovery.

Install:

```bash
sudo mkdir -p /opt/omnivoice/deploy
sudo cp deploy/backup_volunteer.sh /opt/omnivoice/deploy/ && sudo chmod +x /opt/omnivoice/deploy/backup_volunteer.sh
sudo cp deploy/omnivoice-backup.cron /etc/cron.d/omnivoice-backup
sudo /opt/omnivoice/deploy/backup_volunteer.sh      # test run once, by hand
```

Runs as root at **03:30** nightly (`/etc/cron.d/omnivoice-backup`), logging to
`/var/log/omnivoice-backup.log`. Tune via env in the cron entry:

- `OMNIVOICE_BACKUP_DIR` — put it on a **different disk/volume** than the source.
- `OMNIVOICE_BACKUP_REMOTE=user@host:/path` — copy each snapshot **offsite**
  (needs key-based ssh); this is the real "never on one machine" guarantee.
- `OMNIVOICE_BACKUP_KEEP` — how many dated snapshots to retain.

## Pulling recordings to the GPU box

```bash
# on the GPU box — mirror the collected audio + manifest
rsync -av vps:/opt/omnivoice/data/bg/imperatives_volunteer/ \
          data/bg/imperatives_volunteer/
./venv/bin/python convert_volunteer.py --min-per-verb 8   # decode + QC + manifest
# then the printed GPU tokenize command -> tokens/train/data.lst -> ft9
```

## Notes

- **No auth by design** — anyone with the URL can record (that's the point of a
  volunteer drive). To restrict to an invited group, enable the nginx
  `auth_basic` block in the conf (shared password).
- Optionally add nginx `limit_req` if you want basic abuse protection.
- Back up `submissions.jsonl`, `flags.jsonl`, and `raw/` — the irreproducible
  artifacts. `flags.jsonl` (sentences volunteers marked as wrong) is your
  curation feedback: review it to fix/drop bad verb or frame entries.
- Volunteers can record **multiple takes** of a sentence (each saved as
  `…_t0,_t1,…`) and **flag** wrong sentences — no action needed server-side.
