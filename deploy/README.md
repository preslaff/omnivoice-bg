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

## Getting the files onto the VPS (rsync — no repo needed)

`collect.py` is self-contained (stdlib + fastapi + uvicorn; **no torch, no repo,
no editable install**). You only need to copy a handful of files, preserving the
tree so `collect.py` finds its config/logo (it resolves them from `parents[2]` of
its own path — i.e. the deploy root `/opt/omnivoice`):

```
/opt/omnivoice/
├── omnivoice/cli/collect.py
├── config/bg/imperative_verbs.txt
├── config/bg/imperative_frames_clean.txt
└── deploy/logo.png                 # favicon.png is optional (falls back to logo)
```

From your **dev box** (`~/WORK/OmniVoice`), `rsync -R` preserves the relative
paths under a staging dir in your VPS home (no root needed for the copy):

```bash
cd ~/WORK/OmniVoice
rsync -avR \
    omnivoice/cli/collect.py \
    config/bg/imperative_verbs.txt \
    config/bg/imperative_frames_clean.txt \
    deploy/logo.png \
    deploy/omnivoice-collect.service \
    deploy/omnivoice.cdsv.dev.nginx.conf \
    deploy/backup_volunteer.sh \
    deploy/omnivoice-backup.cron \
    deploy/README.md \
    USER@cdsv.dev:omnivoice-stage/
```

Then on the **VPS**, move it into place and hand it to `www-data`:

```bash
sudo mkdir -p /opt/omnivoice
sudo cp -a ~/omnivoice-stage/. /opt/omnivoice/
sudo chown -R www-data:www-data /opt/omnivoice
```

(Substitute your real SSH target for `USER@cdsv.dev`, or an ssh-config alias.)
To update later, re-run the same `rsync` (it's incremental) and
`sudo cp -a ~/omnivoice-stage/. /opt/omnivoice/ && sudo systemctl restart omnivoice-collect`.

**Launch note:** the systemd unit runs `python omnivoice/cli/collect.py` (a script
by path), **not** `python -m omnivoice.cli.collect` — the `-m` form would import
the `omnivoice` package `__init__.py` and pull in torch, which isn't installed on
the VPS. Don't "fix" it back to `-m`.

## Steps

1. **DNS** — point `omnivoice.cdsv.dev` (A/AAAA) at the VPS.
2. **App (systemd)** — see below. It binds to `127.0.0.1:8004`, reachable only
   via nginx.
3. **nginx + TLS** — see below. **HTTPS is mandatory** — the mic (getUserMedia)
   is blocked on plain http.
4. **Verify** — open `https://omnivoice.cdsv.dev/`, allow the mic, record one
   clip; check `https://omnivoice.cdsv.dev/health` and that a file appears under
   `raw/`. `curl https://omnivoice.cdsv.dev/stats` shows coverage.

### 2 — systemd service

The collector listens on **127.0.0.1:8004** (port 8003 was already taken on this
VPS). After the rsync above:

```bash
sudo cp /opt/omnivoice/deploy/omnivoice-collect.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now omnivoice-collect
sudo systemctl status omnivoice-collect --no-pager
curl -s http://127.0.0.1:8004/health          # expect {"ok": true, ...}
journalctl -u omnivoice-collect -f            # follow logs
```

**Port conflict?** If the service crash-loops with `[Errno 98] address already in
use`, another process owns the port. Find and clear it, then reset the restart
counter:

```bash
sudo ss -ltnp | grep ':8004'                  # shows PID + command holding it
sudo systemctl reset-failed omnivoice-collect
sudo systemctl restart omnivoice-collect
```

**To change the port** (edit both the service and the nginx `proxy_pass`, then
reload both):

```bash
# 1. service: swap the --port value (example: 8004 -> 8005)
sudo sed -i 's/--port 8004/--port 8005/' \
    /etc/systemd/system/omnivoice-collect.service /opt/omnivoice/deploy/omnivoice-collect.service
sudo systemctl daemon-reload && sudo systemctl restart omnivoice-collect

# 2. nginx: point the proxy at the new port
sudo sed -i 's|127.0.0.1:8004|127.0.0.1:8005|' \
    /etc/nginx/sites-available/omnivoice.cdsv.dev /opt/omnivoice/deploy/omnivoice.cdsv.dev.nginx.conf
sudo nginx -t && sudo systemctl reload nginx
```

### 3 — nginx reverse proxy + TLS

The shipped conf is **HTTP-only on purpose** — it references no `ssl_certificate`,
because the cert doesn't exist until certbot creates it and nginx won't load a
missing cert (`nginx -t` fails → `certbot --nginx` bails on its own preflight).
Install HTTP-only first, then let certbot add SSL:

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
sudo cp /opt/omnivoice/deploy/omnivoice.cdsv.dev.nginx.conf \
        /etc/nginx/sites-available/omnivoice.cdsv.dev
sudo ln -sf /etc/nginx/sites-available/omnivoice.cdsv.dev \
            /etc/nginx/sites-enabled/omnivoice.cdsv.dev
sudo nginx -t                                  # passes: HTTP-only, no missing cert
sudo systemctl reload nginx
sudo certbot --nginx -d omnivoice.cdsv.dev     # obtains cert, injects 443 + :80->:443
sudo nginx -t && sudo systemctl reload nginx
```

certbot rewrites the installed file in place: adds a `listen 443 ssl` server with
the real cert paths, copies the `proxy_pass` location into it, and turns the `:80`
block into a redirect. Result: HTTPS → `127.0.0.1:8004`, `client_max_body_size
25m` for uploads.

**Prerequisites for certbot:** `omnivoice.cdsv.dev` must already resolve to this
VPS (A/AAAA record) and port 80 reachable — the ACME http-01 challenge needs both.

**If you re-copy the conf later** and `nginx -t` starts failing with `cannot load
certificate .../fullchain.pem`, you've overwritten certbot's SSL edits with the
HTTP-only template again — just re-run `sudo certbot --nginx -d omnivoice.cdsv.dev`
(the cert already exists, so it only re-injects the nginx SSL config).

**Unrelated warnings:** `protocol options redefined for ...:443` lines from your
other sites are harmless nginx warnings, not errors — ignore them.

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
