#!/usr/bin/env python3
# Copyright 2024 Xiaomi Corp. (authors: OmniVoice-BG)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Ingest Bulgarian Parliament plenary sessions → per-session audio + transcript pairs.

Phase 1 of the Accentor reproduction. For each plenary session it fetches one JSON record
(`GET /api/v1/pl-sten/{ID}`) carrying both the full stenographic transcript and the video
part URLs, downloads each MP4 part, extracts 24 kHz mono audio with ffmpeg (deleting the
video immediately), concatenates parts, cleans the transcript via `bg_sten_clean`, and
writes a manifest row consumable by `omnivoice.scripts.extract_audio_tokens`.

See docs/BG_PARLIAMENT_INGESTION.md for the full design.

    ./venv/bin/python prep_bg_parliament.py --out data/bg_parliament \
        --date-start 2010-01-01 --date-end 2018-12-31

Resumable: completed sessions are marked by state/plsten_<ID>.ok and skipped. The
enumeration {id,date} cache (enum_cache.jsonl) makes re-runs instant.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import bg_sten_clean

API = "https://www.parliament.bg/api/v1/pl-sten/{id}"
CONTACT = os.environ.get("OMNIVOICE_BG_CONTACT", "partners@aidentit.com")
USER_AGENT = f"OmniVoice-BG research crawler (contact: {CONTACT})"
TARGET_SR = 24000

# Enumeration: the stenogram IDs below this guard are a non-monotonic re-import of
# 1990s–2000s sessions; the monotonic "modern block" (≈2540–7400) covers 2010–2018.
ID_LO_GUARD = 2540
ID_HI_GUARD = 11000


# --------------------------------------------------------------------------- HTTP
def make_session(retries: int, backoff: float) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries, connect=retries, read=retries,
        backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504),
        respect_retry_after_header=True, raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def api_get(session: requests.Session, pl_id: int, timeout=60):
    """Return parsed session JSON, or None if missing / not JSON (SPA fallthrough)."""
    try:
        r = session.get(API.format(id=pl_id), timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()  # raises if HTML
    except (requests.RequestException, ValueError):
        return None


# --------------------------------------------------------------------- enumeration
def _date_of(session, pl_id, cache):
    if pl_id in cache:
        return cache[pl_id]
    rec = api_get(session, pl_id)
    d = rec.get("Pl_Sten_date") if isinstance(rec, dict) else None
    cache[pl_id] = d
    return d


def find_edge(session, target, cache, want, lo=ID_LO_GUARD, hi=ID_HI_GUARD, sleep=0.0):
    """Binary-search the monotonic modern block for the boundary ID.

    want='first_ge' → smallest id whose date >= target;
    want='last_le'  → largest id whose date <= target.
    Skips None-date IDs by probing neighbours.
    """
    def probe(mid):
        for off in range(0, 40):  # nudge past gap/None IDs
            for m in (mid + off, mid - off):
                if lo <= m <= hi:
                    d = _date_of(session, m, cache)
                    if sleep:
                        time.sleep(sleep)
                    if d:
                        return m, d
        return None, None

    best = None
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        m, d = probe(mid)
        if d is None:
            break
        if want == "first_ge":
            if d >= target:
                best = m; b = m - 1
            else:
                a = m + 1
        else:  # last_le
            if d <= target:
                best = m; a = m + 1
            else:
                b = m - 1
    return best


def enumerate_sessions(session, date_lo, date_hi, cache, sleep=0.0, id_lo=None, id_hi=None):
    """Yield (id, date) for sessions whose date ∈ [date_lo, date_hi].

    Stenogram IDs are only PIECEWISE monotonic — modern sessions (2010-2018) form
    locally-sorted runs INTERLEAVED with chunks of re-imported historical sessions
    (e.g. id 4000→2014 but 4240→1943). Binary search (find_edge) is therefore UNSAFE
    and would skip whole runs; we linear-scan the guarded range and keep by date. Cheap
    (metadata only) and cached in enum_cache.jsonl, so a full scan is a one-time cost.
    """
    if id_lo is None:
        id_lo = ID_LO_GUARD
    if id_hi is None:
        id_hi = ID_HI_GUARD
    print(f"[enum] walking IDs {id_lo}..{id_hi} for dates {date_lo}..{date_hi}", flush=True)
    prev = None
    kept = []
    for pid in range(id_lo, id_hi + 1):
        cached = pid in cache
        d = _date_of(session, pid, cache)
        if sleep and not cached:  # only rate-limit real API calls, not cache hits
            time.sleep(sleep)
        if not d:
            continue
        if prev and d < prev:
            print(f"[enum] WARN backward date jump at id {pid}: {d} < {prev} "
                  f"(continuing with linear date filter)", flush=True)
        prev = d
        if date_lo <= d <= date_hi:
            kept.append((pid, d))
    print(f"[enum] {len(kept)} sessions in range", flush=True)
    return kept


# ------------------------------------------------------------------- media helpers
def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def download_part(session, url, dest: pathlib.Path, timeout=120, max_tries=8):
    """Stream-download to dest, resuming (HTTP Range) until the FULL Content-Length is
    fetched. MP4 keeps its `moov` index atom at the end (non-faststart), so a truncated
    download — a mid-stream connection drop, which a single GET silently accepts — yields
    'moov atom not found' at extraction. We therefore loop, resuming from the partial file
    each time, until size == Content-Length, and verify EXACT size (no 99% tolerance: even
    a few missing KB at the tail can drop the moov)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = None
    for _ in range(max_tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} for {url}")
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if r.status_code == 206 and have else "wb"
                if mode == "wb" and tmp.exists():
                    tmp.unlink()  # server ignored Range → restart cleanly
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
        except requests.RequestException:
            continue  # transient drop → resume on next pass
        if not total:
            break  # server sent no Content-Length; can't verify, accept what we got
    if total and tmp.stat().st_size != total:
        raise RuntimeError(f"short download {tmp.stat().st_size}/{total} for {url}")
    os.replace(tmp, dest)


def probe_mp4_complete(session, url, timeout=60):
    """Cheaply decide if the server's MP4 is COMPLETE, from a 64 KB prefix.

    Some parliament recordings are truncated at source: the server reports a
    Content-Length far smaller than the file's own `mdat` box declares, so the media is
    cut off before the trailing `moov` index is written → 'moov atom not found' at
    extraction. We detect this from the box headers in the first 64 KB instead of
    downloading the whole (useless) part. Returns (ok, total_bytes).
    """
    r = session.get(url, headers={"Range": "bytes=0-65535"}, timeout=timeout)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    if r.status_code == 206 and "/" in r.headers.get("Content-Range", ""):
        total = int(r.headers["Content-Range"].split("/")[-1])
    else:
        total = int(r.headers.get("Content-Length", 0))
    data = r.content
    # 1) must be a real MP4: first box is 'ftyp' (malformed files start with garbage)
    if len(data) < 8 or data[4:8] != b"ftyp":
        return False, total
    off = 0
    while off + 8 <= len(data):
        size = int.from_bytes(data[off:off + 4], "big")
        typ = data[off + 4:off + 8]
        if size == 1:
            if off + 16 > len(data):
                break
            size = int.from_bytes(data[off + 8:off + 16], "big")
        if typ == b"moov":
            return True, total                       # 2) faststart → healthy
        if typ == b"mdat":
            mend = off + size
            if total and mend >= total:
                return False, total                  # 3a) mdat exceeds file → truncated
            # 3b) verify a real moov actually follows mdat within the file
            tail = session.get(url, headers={"Range": f"bytes={mend}-{mend + 15}"},
                               timeout=timeout).content
            if len(tail) >= 8 and tail[4:8] == b"moov":
                msize = int.from_bytes(tail[:4], "big")
                return (not total or mend + msize <= total), total
            return False, total                      # no moov after mdat → truncated
        if size <= 0:
            break
        off += size
    return True, total                               # ftyp ok but inconclusive → proceed


def ffmpeg_extract(mp4: pathlib.Path, out: pathlib.Path, audio_format: str):
    codec = "flac" if audio_format == "flac" else "pcm_s16le"
    # Force 16-bit: ffmpeg decodes the source AAC to float and FLAC would otherwise
    # store 24-bit (sample_fmt=s32), where bits 17-24 are just lossy-codec
    # reconstruction noise that barely compresses — ~2x larger files for zero usable
    # quality (24 kHz mono 16-bit is acoustic-grade and what OmniVoice consumes).
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(mp4),
           "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16",
           "-c:a", codec, str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


def ffmpeg_concat(parts, out: pathlib.Path):
    """Stream-copy concat of identically-encoded part files via the concat demuxer."""
    listf = out.with_suffix(".list.txt")
    listf.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listf), "-c", "copy", str(out)],
                   check=True, capture_output=True)
    listf.unlink(missing_ok=True)


def ffprobe_audio(path: pathlib.Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    j = json.loads(out)
    st = (j.get("streams") or [{}])[0]
    return int(st["sample_rate"]), int(st["channels"]), float(j["format"]["duration"])


# ------------------------------------------------------------------- per session
def process_session(session, rec, out: pathlib.Path, args):
    pl_id = rec["Pl_Sten_id"]
    sid = f"plsten_{pl_id}"
    ok_marker = out / "state" / f"{sid}.ok"
    if ok_marker.exists():
        return None  # already done

    video = rec.get("video") or {}
    playlist = sorted(video.get("playlist") or [], key=lambda p: p["item"])
    if not playlist:
        raise RuntimeError("novideo: empty playlist")
    if video.get("Vicount") and len(playlist) != video["Vicount"]:
        raise RuntimeError(f"partcount: {len(playlist)} != Vicount {video['Vicount']}")

    ext = "flac" if args.audio_format == "flac" else "wav"
    raw_dir = out / "raw" / sid
    scratch = raw_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    # Download → extract → delete, one part at a time (peak disk ≈ one MP4).
    part_audios = []
    for p in playlist:
        mp4 = raw_dir / f"part_{p['item']}.mp4"
        pa = scratch / f"part_{p['item']}.{ext}"
        if not pa.exists():
            ok, _ = probe_mp4_complete(session, p["file"])
            if not ok:
                raise RuntimeError(
                    f"truncated: incomplete server file (mdat exceeds Content-Length, "
                    f"no moov) for part {p['item']}")
            download_part(session, p["file"], mp4)
            ffmpeg_extract(mp4, pa, args.audio_format)
        if not args.keep_video:
            mp4.unlink(missing_ok=True)
        part_audios.append(pa)

    # Concat (single part: just move).
    final_tmp = scratch / f"session.{ext}"
    if len(part_audios) == 1:
        os.replace(part_audios[0], final_tmp)
    else:
        ffmpeg_concat(part_audios, final_tmp)

    sr, ch, dur = ffprobe_audio(final_tmp)
    if sr != TARGET_SR or ch != 1:
        raise RuntimeError(f"badaudio: sr={sr} ch={ch}")

    audio_final = (out / "audio" / f"{sid}.{ext}").resolve()
    audio_final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(final_tmp, audio_final)

    # Transcript.
    full_text, turns, meta = bg_sten_clean.clean_body(rec.get("Pl_Sten_body") or "")
    tr_path = (out / "transcripts" / f"{sid}.txt").resolve()
    tr_path.parent.mkdir(parents=True, exist_ok=True)
    tr_path.write_text(full_text, encoding="utf-8")
    turns_path = (out / "turns" / f"{sid}.jsonl").resolve()
    turns_path.parent.mkdir(parents=True, exist_ok=True)
    with open(turns_path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    shutil.rmtree(raw_dir, ignore_errors=True)  # drop scratch + any leftover

    return {
        "id": sid, "audio_path": str(audio_final), "text": full_text,
        "language_id": "bg", "session_date": rec.get("Pl_Sten_date"),
        "pl_sten_id": pl_id, "n_parts": len(playlist),
        "duration": round(dur, 2), "turns_path": str(turns_path),
        "headers": meta["headers"],
    }


# ------------------------------------------------------------------- orchestration
def load_done(out: pathlib.Path) -> set:
    state = out / "state"
    return {p.stem for p in state.glob("*.ok")} if state.exists() else set()


def append_jsonl(path: pathlib.Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Bulgarian Parliament ingestion (Phase 1)")
    ap.add_argument("--out", default="data/bg_parliament")
    ap.add_argument("--date-start", default="2010-01-01")
    ap.add_argument("--date-end", default="2018-12-31")
    ap.add_argument("--id-start", type=int, default=None, help="override enumeration low edge")
    ap.add_argument("--id-end", type=int, default=None, help="override enumeration high edge")
    ap.add_argument("--ids", default=None,
                    help="comma-separated explicit Pl_Sten IDs to ingest (bypasses enumeration)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N sessions (0=all)")
    ap.add_argument("--reverse", action="store_true",
                    help="process newest date first (for disk-budgeted newest-first crawls)")
    ap.add_argument("--dry-run", action="store_true", help="enumerate only, no download")
    ap.add_argument("--sleep", type=float, default=1.5, help="seconds between API calls")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--min-free-gb", type=float, default=60.0, help="abort before a session if below")
    ap.add_argument("--audio-format", choices=["flac", "wav"], default="flac")
    ap.add_argument("--keep-video", action="store_true", help="debug: don't delete MP4s")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "state").mkdir(parents=True, exist_ok=True)

    # Single-instance lock: two concurrent crawlers append to the same .part
    # files (size overshoots) and rmtree each other's raw dirs (Errno 2) —
    # observed 2026-07-17. flock is held for the process lifetime and
    # released by the kernel on ANY exit, so a crash never wedges the lock.
    import fcntl
    lock_f = open(out / "crawler.lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"ERROR: another crawler instance is already running "
                 f"({out/'crawler.lock'} is locked) — refusing to start.")
    lock_f.write(f"{os.getpid()}\n")
    lock_f.flush()

    session = make_session(args.max_retries, args.backoff)

    # Enumeration cache (resumable).
    cache_path = out / "enum_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for ln in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(ln); cache[int(o["id"])] = o["date"]
            except Exception:
                pass
    before = set(cache)

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        sessions = [(pid, cache.get(pid)) for pid in ids]  # dates resolved per-session
        print(f"[ids] explicit list ({len(sessions)}): {ids}", flush=True)
    else:
        sessions = enumerate_sessions(session, args.date_start, args.date_end, cache,
                                      sleep=args.sleep, id_lo=args.id_start, id_hi=args.id_end)
        with open(cache_path, "a", encoding="utf-8") as f:  # persist newly-probed dates
            for pid in cache:
                if pid not in before:
                    f.write(json.dumps({"id": pid, "date": cache[pid]}, ensure_ascii=False) + "\n")

    if args.dry_run:
        print(f"[dry-run] {len(sessions)} sessions; cache -> {cache_path}")
        return

    done = load_done(out)
    todo = [(pid, d) for pid, d in sessions if f"plsten_{pid}" not in done]
    if args.reverse:
        todo.sort(key=lambda pd: (pd[1] or ""), reverse=True)  # newest date first
    if args.limit:
        todo = todo[: args.limit]
    print(f"[run] {len(todo)} to process ({len(done)} already done)", flush=True)

    manifest = out / "manifest.jsonl"
    errors = out / "errors.jsonl"
    n_ok = n_err = 0
    for pid, d in todo:
        sid = f"plsten_{pid}"
        if free_gb(out) < args.min_free_gb:
            print(f"[run] STOP: free {free_gb(out):.0f} GB < {args.min_free_gb} GB", flush=True)
            break
        try:
            rec = api_get(session, pid)
            if not isinstance(rec, dict):
                raise RuntimeError("api: no JSON record")
            row = process_session(session, rec, out, args)
            if row is None:
                continue
            append_jsonl(manifest, row)
            (out / "state" / f"{sid}.ok").write_text("", encoding="utf-8")
            n_ok += 1
            print(f"[ok] {sid} {d} parts={row['n_parts']} dur={row['duration']:.0f}s "
                  f"text={len(row['text'])}c free={free_gb(out):.0f}GB", flush=True)
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            stage = str(exc).split(":")[0]
            append_jsonl(errors, {"id": sid, "date": d, "stage": stage,
                                  "error": str(exc), "ts": int(time.time())})
            shutil.rmtree(out / "raw" / sid, ignore_errors=True)
            print(f"[err] {sid}: {exc}", file=sys.stderr, flush=True)
        time.sleep(args.sleep)

    print(f"[done] ok={n_ok} err={n_err} manifest={manifest}")


if __name__ == "__main__":
    main()
