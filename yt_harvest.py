#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Harvest a YouTube channel's audio + auto-captions as a conversational ASR training axis.

Built for @DnevenRed (publicistic talk/interviews — spontaneous political speech, the axis
parliament/audiobooks/FLEURS/CV lack) but generic via --channel. IMPORTANT: the channel's
transcriptions are YouTube AUTO-captions (kind=asr, verified 2026-08-04) — machine ASR, NOT
gold labels. They are harvested as a SECOND TEACHER for two-teacher agreement filtering
(Whisper hypothesis vs YT caption; keep agreeing spans), not as ground truth. Labels come from
the pseudo-label pipeline (bnr_pseudolabel.py stages, later with --agree-subs).

Downloads per video: best audio-only stream (m4a/opus) + the Bulgarian auto-sub (vtt) when
present, into data/ytdr/media/<id>.*, appending one line per video to data/ytdr/yt_manifest.jsonl:
  {video_id, title, duration, upload_date, audio, subs, channel}

RESUMABLE: videos already in the manifest with audio on disk are skipped — re-run the same
line after any interruption. POLITE: --delay between videos + yt-dlp's own backoff.
Note: downloading is against YouTube TOS (research-practice gray zone, same call as BNR —
docs/BG_ASR_BNR_PLAN.md caveats); do not redistribute.

    pip install yt-dlp
    ./venv/bin/python yt_harvest.py --limit 3          # smoke
    ./venv/bin/python yt_harvest.py                    # full channel (resumable)
    ./venv/bin/python yt_harvest.py --channel https://www.youtube.com/@SomeOther/videos
"""
import argparse
import json
import pathlib
import time

try:
    import yt_dlp
except ImportError:
    raise SystemExit("yt-dlp not installed. Run:  ./venv/bin/python -m pip install yt-dlp")


def enumerate_videos(channel_url, playlistend=None):
    """flat playlist -> [{'id','title','duration'}, ...] (no downloads).

    playlistend caps the ENUMERATION itself (newest-first) — essential for huge channels
    (e.g. TV stations with thousands of uploads) where a full listing takes minutes/hangs.
    """
    opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    if playlistend:
        opts["playlistend"] = playlistend
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = info.get("entries") or []
    # channel pages can nest one level (tabs); flatten
    if entries and entries[0].get("entries"):
        entries = [e for tab in entries for e in (tab.get("entries") or [])]
    return [e for e in entries if e and e.get("id")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="https://www.youtube.com/@DnevenRed/videos")
    ap.add_argument("--out-dir", default="data/ytdr")
    ap.add_argument("--lang", default="bg", help="subtitle language to fetch (auto-subs)")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between videos")
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (smoke)")
    ap.add_argument("--min-duration", type=float, default=60.0,
                    help="skip clips shorter than this (shorts/teasers)")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    manifest = out / "yt_manifest.jsonl"

    done = set()
    if manifest.exists():
        for line in manifest.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["video_id"])
            except Exception:
                pass
    print(f"resuming: {len(done):,} videos already in manifest")

    print(f"enumerating {args.channel} …", flush=True)
    # cap the enumeration itself for huge channels: newest limit + headroom for already-done
    # + short-skips. Without --limit we still list everything (fine for normal channels).
    enum_cap = (args.limit * 2 + len(done)) if args.limit else None
    vids = enumerate_videos(args.channel, playlistend=enum_cap)
    print(f"  {len(vids):,} videos enumerated" + (f" (newest {enum_cap} only)" if enum_cap else ""))
    todo = [v for v in vids if v["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    est_h = sum((v.get("duration") or 0) for v in vids) / 3600
    print(f"  to download: {len(todo):,}  (channel total ~{est_h:.0f} h)\n")

    dl_opts = {
        "quiet": True, "no_warnings": True, "noprogress": True,
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(media / "%(id)s.%(ext)s"),
        "writeautomaticsub": True, "writesubtitles": True,
        "subtitleslangs": [args.lang], "subtitlesformat": "vtt",
        "retries": 3, "ignoreerrors": True,
    }
    mf = manifest.open("a", encoding="utf-8")
    n_ok = n_skip = n_fail = 0
    nbytes = 0
    t0 = time.time()
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        for i, v in enumerate(todo, 1):
            vid = v["id"]
            dur = v.get("duration") or 0
            if dur and dur < args.min_duration:
                n_skip += 1
                print(f"[{i}/{len(todo)}] {vid} SKIP short ({dur:.0f}s) {v.get('title','')[:50]}",
                      flush=True)
                continue
            print(f"[{i}/{len(todo)}] {vid} ({dur/60:.0f} min) {v.get('title','')[:60]} …",
                  end="", flush=True)
            try:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
            except Exception as e:
                print(f" FAIL ({e})", flush=True)
                n_fail += 1
                time.sleep(args.delay)
                continue
            if info is None:
                print(" FAIL (no info)", flush=True)
                n_fail += 1
                time.sleep(args.delay)
                continue
            audio = next((p for p in media.glob(f"{vid}.*")
                          if p.suffix in (".m4a", ".webm", ".opus", ".mp3")), None)
            subs = next(iter(media.glob(f"{vid}.{args.lang}.vtt")), None)
            if audio is None:
                print(" FAIL (no audio file)", flush=True)
                n_fail += 1
                time.sleep(args.delay)
                continue
            sz = audio.stat().st_size
            nbytes += sz
            n_ok += 1
            mf.write(json.dumps({
                "video_id": vid, "title": info.get("title") or v.get("title"),
                "duration": info.get("duration") or dur,
                "upload_date": info.get("upload_date"),
                "audio": str(audio), "subs": str(subs) if subs else None,
                "channel": args.channel,
            }, ensure_ascii=False) + "\n")
            mf.flush()
            el = time.time() - t0
            eta_min = (el / i) * (len(todo) - i) / 60
            print(f" {sz/1e6:.0f} MB{' +subs' if subs else ' (NO subs)'} | "
                  f"total {n_ok} dl / {nbytes/1e9:.2f} GB / {n_fail} fail | ETA ~{eta_min:.0f} min",
                  flush=True)
            time.sleep(args.delay)
    mf.close()
    print(f"\nDONE: {n_ok} downloaded ({nbytes/1e9:.2f} GB), {n_skip} short-skipped, {n_fail} failed")
    print(f"  media -> {media}/    manifest -> {manifest}")
    print("Next: speech/music-seg -> Whisper + faithfulness filter (+ two-teacher agreement vs "
          "the .vtt subs) -> cuts. See docs/BG_ASR_BNR_PLAN.md pipeline.")


if __name__ == "__main__":
    main()
