#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Crawl the BNR archive: download every audio segment + its metadata (for ASR pseudo-labeling).

Step 3 of docs/BG_ASR_BNR_PLAN.md. Enumerates posts from the sitemap, parses each page's
`__NEXT_DATA__` for audio `Sections`, downloads `/api/media/<Id>` -> data/bnr/audio/<Id>.mp3,
and appends one metadata line per segment to data/bnr/bnr_manifest.jsonl:
  {audio_id, path, filename, source, description, post_id, post_title, category, post_url}
The description/title/category are kept for later Bulgarian filtering + weak-label context; the
transcript itself comes later from Whisper (see the plan — filter, don't LLM-rewrite).

RESUMABLE: segments already in the manifest with their mp3 on disk are skipped (survives power
cuts — re-run the same line). POLITE: research User-Agent, `--delay` between every request,
honors robots.txt (only /*/search is blocked). Copyright: see the plan — don't redistribute.

    ./venv/bin/python bnr_crawl.py                    # full crawl (resumable; ~300h/~17GB, hours)
    ./venv/bin/python bnr_crawl.py --limit-posts 3    # smoke
    ./venv/bin/python bnr_crawl.py --delay 1.5        # gentler
    ./venv/bin/python bnr_crawl.py --category vremena-i-hora   # one section only
"""
import argparse
import json
import os
import pathlib
import re
import time

from bnr_scale_check import BASE, NEXT_RE, _get, fetch_text, enumerate_posts

PID_RE = re.compile(r"/post/(\d+)/")


def post_data(html):
    m = NEXT_RE.search(html)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except Exception:
        return None
    return ((data.get("props") or {}).get("pageProps") or {}).get("data") or {}


def download(mid, dst, timeout=180):
    """GET /api/media/<mid> -> dst (atomic via .part). Returns bytes written."""
    tmp = pathlib.Path(str(dst) + ".part")
    with _get(f"{BASE}/api/media/{mid}", timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("empty download")
    os.replace(tmp, dst)
    return dst.stat().st_size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/bnr")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--limit-posts", type=int, default=0, help="only first N posts (smoke)")
    ap.add_argument("--category", default=None, help="restrict to one category UrlName")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    adir = out / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    manifest = out / "bnr_manifest.jsonl"

    done = set()
    if manifest.exists():
        for line in manifest.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["audio_id"])
            except Exception:
                pass
    print(f"resuming: {len(done):,} segments already in manifest")

    posts = enumerate_posts()
    if args.limit_posts:
        posts = posts[: args.limit_posts]
    print(f"{len(posts):,} posts to crawl\n")

    mf = manifest.open("a", encoding="utf-8")
    seen = new = fail = 0
    nbytes = 0
    for pi, url in enumerate(posts, 1):
        m = PID_RE.search(url)
        pid = m.group(1) if m else ""
        try:
            d = post_data(fetch_text(url))
        except Exception as e:
            print(f"  [warn] post {pid}: {e}")
            time.sleep(args.delay)
            continue
        time.sleep(args.delay)
        if not d:
            continue
        cat = (((d.get("ProgramOwner") or {}).get("Category") or {}).get("UrlName") or "")
        if args.category and cat != args.category:
            continue
        title = d.get("Title") or ""
        for s in (d.get("Sections") or []):
            a = s.get("Audio")
            if not a or not a.get("Id"):
                continue
            mid = a["Id"]
            seen += 1
            dst = adir / f"{mid}.mp3"
            if mid in done and dst.exists():
                continue
            sz = None
            for attempt in range(args.retries):
                try:
                    sz = download(mid, dst)
                    break
                except Exception as e:
                    if attempt == args.retries - 1:
                        print(f"    [fail] {mid} (post {pid}): {e}")
                    time.sleep(args.delay * (attempt + 1))
            if sz is None:
                fail += 1
                continue
            rec = {"audio_id": mid, "path": str(dst), "filename": a.get("FileName"),
                   "source": a.get("Source"), "description": a.get("Description"),
                   "post_id": pid, "post_title": title, "category": cat, "post_url": url}
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            done.add(mid)
            new += 1
            nbytes += sz
            time.sleep(args.delay)
        if pi % 25 == 0:
            print(f"  {pi}/{len(posts)} posts | {seen} segs seen, {new} new, "
                  f"{nbytes/1e9:.1f} GB, {fail} failed")
    mf.close()
    print(f"\nDONE: {seen} segments seen, {new} downloaded ({nbytes/1e9:.1f} GB), {fail} failed")
    print(f"  audio -> {adir}/    manifest -> {manifest}")
    print("Next (plan step 4): VAD-segment -> Whisper -> faithfulness filter -> cuts.")


if __name__ == "__main__":
    main()
