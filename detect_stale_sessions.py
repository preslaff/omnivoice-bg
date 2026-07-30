#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Detect parliament sessions whose Kaldi segments no longer match the on-disk audio.

A `refresh_bg_parliament` pass replaced some parliament FLACs with shorter/different
audio (and updated data/bg_parliament/manifest.jsonl to match) WITHOUT re-deriving the
Kaldi segments. Those stale segments point past the real end of the current audio
(ffmpeg yields an empty WAV there), so any cut built from them would train/score on
silence or misaligned audio. A session is flagged STALE when its max segment end-time
exceeds the true file duration by > --tol seconds.

Writes:
  * data/lhotse/bg/stale_sessions.txt      — the blocklist (one session id per line),
    consumed by make_asr_split.py and prep_lhotse_manifests.py so both the split and
    the manifests exclude the same sessions (keeps the two ASR systems comparable).
  * data/bg_parliament/manifest_stale.jsonl — the stale sessions' manifest rows, ready
    to feed a Kaldi re-segmentation against the current audio (data recovery for v2).

    ./venv/bin/python detect_stale_sessions.py
"""
import argparse
import collections
import json
import os
import pathlib
import re
from concurrent.futures import ProcessPoolExecutor

FF_INPUT = re.compile(r"-i\s+(\S+)")


def parse_dir(d):
    d = pathlib.Path(d)
    path, segend = {}, collections.defaultdict(float)
    for line in (d / "wav.scp").read_text(encoding="utf-8").splitlines():
        rid, rest = line.split(None, 1)
        m = FF_INPUT.search(rest)
        if m:
            path.setdefault(rid, m.group(1))
    for line in (d / "segments").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 4:
            segend[p[1]] = max(segend[p[1]], float(p[3]))
    return path, segend


def probe(rp):
    import soundfile as sf
    rec, p = rp
    try:
        return rec, sf.info(p).duration
    except Exception:
        return rec, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=[
        "exp/kaldi_pilot/data/train_cleaned",
        "exp/kaldi_pilot/data/parl_new_cleaned",
        "exp/kaldi_pilot/data/parl_new2_cleaned",
    ])
    ap.add_argument("--manifest", default="data/bg_parliament/manifest.jsonl")
    ap.add_argument("--out", default="data/lhotse/bg/stale_sessions.txt")
    ap.add_argument("--stale-manifest", default="data/bg_parliament/manifest_stale.jsonl")
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args()

    path, segend = {}, collections.defaultdict(float)
    for d in args.dirs:
        if not (pathlib.Path(d) / "segments").exists():
            print(f"  [skip] {d} (no segments)")
            continue
        pp, se = parse_dir(d)
        path.update({k: v for k, v in pp.items() if k not in path})
        for k, v in se.items():
            segend[k] = max(segend[k], v)

    parl = {r: p for r, p in path.items() if r.startswith("plsten")}
    print(f"probing {len(parl):,} parliament FLAC headers with {args.jobs} workers…")
    fdur = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for rec, d in ex.map(probe, parl.items(), chunksize=8):
            fdur[rec] = d

    stale, missing = [], []
    for rec in parl:
        d = fdur.get(rec)
        if d is None:
            missing.append(rec)
        elif segend[rec] > d + args.tol:
            stale.append(rec)
    stale.sort()

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(stale) + ("\n" if stale else ""))
    print(f"\nparliament sessions: {len(parl)}  |  consistent {len(parl)-len(stale)-len(missing)}"
          f"  stale {len(stale)}  missing {len(missing)}")
    print(f"wrote blocklist -> {out}")

    # stage the stale sessions' manifest rows for later Kaldi re-segmentation
    staleset = set(stale)
    mp = pathlib.Path(args.manifest)
    if mp.exists():
        rows = [l for l in mp.read_text(encoding="utf-8").splitlines()
                if json.loads(l).get("id") in staleset]
        sm = pathlib.Path(args.stale_manifest)
        sm.write_text("\n".join(rows) + ("\n" if rows else ""))
        print(f"wrote {len(rows)} stale manifest rows -> {sm}  (feed to re-segmentation)")
    if missing:
        print(f"  NOTE: {len(missing)} FLAC(s) unreadable/missing: {missing[:5]}")


if __name__ == "__main__":
    main()
