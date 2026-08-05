#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Repair the 110 'stale' parliament FLACs — broken containers, NOT short audio.

ROOT CAUSE (found 2026-08-05): the refresh crawl byte-CONCATENATED multi-part session FLACs.
The first part's STREAMINFO header claims only that part's length (e.g. 3.37 h), while
sequential decoders (ffmpeg/Kaldi pipes) play through all appended streams (15.7 h real).
Header-seeking readers (lhotse/soundfile) can only address part 1 — the actual mechanism
behind the 'stale sessions' quarantine and the vetting desync. Stream-copy remux fails
(non-monotonic DTS across the glued streams), so the fix is a LOSSLESS decode -> re-encode
(FLAC -> PCM -> FLAC) producing one clean stream with a correct header.

Per file: re-encode to <name>.fix.flac -> verify the new header duration matches Kaldi's
full-decode ground truth (exp/kaldi_pilot/data/parl_restale/utt2dur, ±--tol s) -> move the
broken original to data/bg_parliament/audio_brokenheader/ -> atomically install the fix.
RESUMABLE: already-repaired files (header ≈ utt2dur) are skipped. Parallel via --jobs.

    ./venv/bin/python repair_flac_headers.py --limit 1     # smoke on the smallest file
    ./venv/bin/python repair_flac_headers.py --jobs 6      # full repair (~2-4 h CPU)
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import subprocess
import time


def load_truth():
    """session -> Kaldi full-decode duration (authoritative)."""
    truth = {}
    for line in pathlib.Path("exp/kaldi_pilot/data/parl_restale/utt2dur").read_text().splitlines():
        rec, dur = line.split()
        truth[rec] = float(dur)
    return truth


def header_dur(path):
    import soundfile as sf
    return sf.info(str(path)).duration


def repair_one(task):
    sid, path, want, tol = task
    src = pathlib.Path(path)
    t0 = time.time()
    have = header_dur(src)
    if abs(have - want) <= tol:
        return sid, "already-ok", have, 0.0
    fix = src.with_suffix(".fix.flac")
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
                        "-c:a", "flac", "-compression_level", "5", str(fix)],
                       capture_output=True)
    if r.returncode != 0 or not fix.exists() or fix.stat().st_size == 0:
        fix.unlink(missing_ok=True)
        return sid, f"ENCODE-FAIL: {r.stderr.decode()[-120:]}", have, time.time() - t0
    got = header_dur(fix)
    if abs(got - want) > tol:
        fix.unlink(missing_ok=True)
        return sid, f"VERIFY-FAIL: fixed header {got:.1f}s != kaldi {want:.1f}s", have, time.time() - t0
    bdir = src.parent.parent / "audio_brokenheader"
    bdir.mkdir(exist_ok=True)
    src.rename(bdir / src.name)                 # keep the broken original until user deletes
    fix.rename(src)                             # atomic install under the original path
    return sid, f"REPAIRED {have/3600:.2f}h->{got/3600:.2f}h", got, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stale-manifest", default="data/bg_parliament/manifest_stale.jsonl")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="only N files (smallest first; smoke)")
    args = ap.parse_args()

    truth = load_truth()
    tasks = []
    for line in open(args.stale_manifest, encoding="utf-8"):
        r = json.loads(line)
        sid = r["id"]
        if sid in truth and pathlib.Path(r["audio_path"]).exists():
            tasks.append((sid, r["audio_path"], truth[sid], args.tol))
    tasks.sort(key=lambda t: pathlib.Path(t[1]).stat().st_size)     # smallest first
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} files to check/repair (jobs={args.jobs})")

    done = ok = fail = 0
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for sid, status, dur, dt in ex.map(repair_one, tasks):
            done += 1
            if status.startswith(("ENCODE", "VERIFY")):
                fail += 1
            else:
                ok += 1
            el = time.time() - t0
            eta = (el / done) * (len(tasks) - done) / 60
            print(f"[{done}/{len(tasks)}] {sid}: {status}  ({dt:.0f}s)  | ok {ok} fail {fail} "
                  f"| ETA ~{eta:.0f} min", flush=True)
    print(f"\nDONE: {ok} ok, {fail} failed. Broken originals in data/bg_parliament/audio_brokenheader/")
    print("Next: re-run the segments-vs-audio sanity check, then the delta path "
          "(prep_lhotse_manifests --dirs parl_restale_cleaned ... + compute_fbank_bg).")


if __name__ == "__main__":
    main()
