#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Scale-check the BNR archive harvest — how many posts / audio segments / est. hours?

Step 2 of docs/BG_ASR_BNR_PLAN.md: before building the full crawler, size the opportunity.
Enumerates every post from the sitemap, samples `--sample` of them, parses each page's
`__NEXT_DATA__` for audio `Sections`, and estimates total hours by sizing a subset of the mp3s
via an HTTP `Content-Range` probe (1-byte request → total bytes; 128 kbps ⇒ seconds = bytes/16000).

POLITE + READ-ONLY: research User-Agent, `--delay` between requests, downloads nothing but the
sampled HTML pages and tiny 1-byte range probes. Honors robots.txt (only /*/search is blocked).

    ./venv/bin/python bnr_scale_check.py --sample 40
    ./venv/bin/python bnr_scale_check.py --sample 80 --delay 1.5   # gentler / tighter estimate
"""
import argparse
import json
import re
import statistics as st
import time
import urllib.request

BASE = "https://archives.bnr.bg"
UA = "Mozilla/5.0 (OmniVoice-BG research scale-check; contact: partners@aidentit.com)"
POST_RE = re.compile(r"https://archives\.bnr\.bg/archives/post/\d+/[^< ]+")
NEXT_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
CR_RE = re.compile(r"/(\d+)\s*$")


def _get(url, timeout=30, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if rng:
        req.add_header("Range", rng)
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_text(url):
    with _get(url) as r:
        return r.read().decode("utf-8", "ignore")


def enumerate_posts():
    """sitemap index -> sub-sitemaps -> unique /archives/post/ URLs."""
    idx = fetch_text(f"{BASE}/sitemap.xml")
    subs = re.findall(r"<loc>([^<]+)</loc>", idx)
    urls = set()
    for s in subs:
        try:
            urls.update(POST_RE.findall(fetch_text(s)))
        except Exception as e:
            print(f"  [warn] sub-sitemap {s}: {e}")
    return sorted(urls)


def audio_sections(post_html):
    """-> list of (audio_id, source) for sections that carry audio."""
    m = NEXT_RE.search(post_html)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except Exception:
        return []
    secs = (((data.get("props") or {}).get("pageProps") or {}).get("data") or {}).get("Sections") or []
    out = []
    for s in secs:
        a = s.get("Audio")
        if a and a.get("Id"):
            out.append((a["Id"], (a.get("Source") or "").strip()))
    return out


def media_bytes(mid, timeout=30):
    """total mp3 size via Content-Range (1-byte probe), or None."""
    try:
        with _get(f"{BASE}/api/media/{mid}", rng="bytes=0-0", timeout=timeout) as r:
            m = CR_RE.search(r.headers.get("Content-Range", ""))
            return int(m.group(1)) if m else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=40, help="how many posts to inspect")
    ap.add_argument("--per-post-probe", type=int, default=2, help="mp3s to size per sampled post")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (politeness)")
    ap.add_argument("--bitrate-kbps", type=float, default=128.0, help="mp3 bitrate for byte→sec est.")
    args = ap.parse_args()

    print("enumerating posts from sitemap…")
    posts = enumerate_posts()
    n = len(posts)
    print(f"  total posts: {n:,}")
    if not n:
        return
    step = max(1, n // args.sample)
    sample = posts[::step][: args.sample]          # deterministic even sampling
    print(f"  sampling {len(sample)} posts (every {step}th)\n")

    bps = args.bitrate_kbps * 1000 / 8
    seg_counts, durations, sources = [], [], {}
    for i, u in enumerate(sample, 1):
        try:
            secs = audio_sections(fetch_text(u))
        except Exception as e:
            print(f"  [warn] {u.split('/')[-1]}: {e}")
            continue
        seg_counts.append(len(secs))
        for _id, src in secs:
            sources[src] = sources.get(src, 0) + 1
        for mid, _src in secs[: args.per_post_probe]:
            b = media_bytes(mid)
            if b:
                durations.append(b / bps)
            time.sleep(args.delay)
        if i % 10 == 0:
            print(f"  {i}/{len(sample)} posts inspected…")
        time.sleep(args.delay)

    avg_segs = st.mean(seg_counts) if seg_counts else 0
    avg_dur = st.mean(durations) if durations else 0
    total_segs = n * avg_segs
    total_hours = total_segs * avg_dur / 3600
    print("\n=== SCALE ESTIMATE ===")
    print(f"  posts                 : {n:,}")
    print(f"  avg audio segs / post : {avg_segs:.1f}   (from {len(seg_counts)} sampled posts)")
    print(f"  avg segment duration  : {avg_dur/60:.1f} min   (sized {len(durations)} mp3s @ {args.bitrate_kbps:.0f} kbps)")
    print(f"  --> ~{total_segs:,.0f} audio segments,  ~{total_hours:,.0f} hours total")
    print("\n  top 'Source' tags (for later Bulgarian/provenance filtering):")
    for src, c in sorted(sources.items(), key=lambda x: -x[1])[:8]:
        print(f"    {c:4d}x  {src or '(empty)'}")
    print("\n  NOTE: hours assume a constant bitrate; a few % may be non-Bulgarian (filter by "
          "Source or Whisper language-id at harvest). Next: draft the full crawler if the scale is worth it.")


if __name__ == "__main__":
    main()
