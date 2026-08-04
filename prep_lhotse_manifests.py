#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Convert the Kaldi-aligned BG corpus to lhotse manifests for icefall/Zipformer.

Builds lhotse Recordings/Cuts DIRECTLY from the Kaldi `wav.scp` + `segments` +
`text`. The Kaldi `wav.scp` wraps every source in an `ffmpeg … -ar 16000 … -f wav -`
PIPE — but a pipe is NOT seekable, so fbank would have to decode the WHOLE source
per cut (catastrophic for the 2 h parliament FLACs with thousands of segments each,
and 192 K ffmpeg subprocess launches for the audiobook clips). Instead we parse the
real file path out of each `-i <path>` and attach it as a seekable `type="file"`
source (audiobook `data/bg400/clips/*.wav`, parliament `data/bg_parliament/audio/*.flac`
— all 24 kHz mono PCM). fbank then seeks to each segment's slice and resamples
24k→16k on the fly. File headers are probed in parallel (header-only, fast — NOT the
per-file ffmpeg decode that dragged); durations are therefore exact.

One MonoCut is built per segment (== trim_to_supervisions output, without lhotse's
overlapping/channel machinery which trips on time-overlapping parliament segments).
Applies the frozen whole-session split (make_asr_split.py) and cleans supervision
text (strip U+0301 + drop <unk> OOV tokens).

    ./venv/bin/python prep_lhotse_manifests.py   # -> data/lhotse/bg/cuts_{train,dev,test}.jsonl.gz
"""
import argparse
import os
import pathlib
import re
from concurrent.futures import ProcessPoolExecutor

ACUTE = "́"
FF_INPUT = re.compile(r"-i\s+(\S+)")
TARGET_SR = 16000


def clean_text(s):
    toks = [t for t in s.replace(ACUTE, "").split() if t != "<unk>"]
    return " ".join(toks)


def parse_dir(d):
    """Return path{rec->file}, text{utt->str}, segs[(utt,rec,start,end)]."""
    d = pathlib.Path(d)
    path = {}
    for line in (d / "wav.scp").read_text(encoding="utf-8").splitlines():
        rid, rest = line.split(None, 1)
        m = FF_INPUT.search(rest)
        if m:
            path[rid] = m.group(1)
    text = {}
    for line in (d / "text").read_text(encoding="utf-8").splitlines():
        p = line.split(None, 1)
        if len(p) == 2:
            text[p[0]] = p[1]
    segs = []
    for line in (d / "segments").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 4:
            segs.append((p[0], p[1], float(p[2]), float(p[3])))
    return path, text, segs


def probe(rp):
    """(rec, path) -> (rec, path, sr, channels, frames) via header-only read."""
    import soundfile as sf
    rec, p = rp
    try:
        info = sf.info(p)
        return rec, p, info.samplerate, info.channels, info.frames
    except Exception as e:                       # missing/unreadable file
        return rec, p, None, None, str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=[
        "exp/kaldi_pilot/data/train_cleaned",
        "exp/kaldi_pilot/data/parl_new_cleaned",
        "exp/kaldi_pilot/data/parl_new2_cleaned",
    ])
    ap.add_argument("--split", default="data/lhotse/bg/split")
    ap.add_argument("--out", default="data/lhotse/bg")
    ap.add_argument("--exclude", default="data/lhotse/bg/stale_sessions.txt",
                    help="blocklist of stale sessions (detect_stale_sessions.py) to skip entirely")
    ap.add_argument("--reseg-dir", default=None,
                    help="re-segmented Kaldi data dir (parl_restale_cleaned from "
                         "resegment_stale_sessions.sh) that OVERRIDES the base --dirs for its "
                         "sessions: fresh segments matching the CURRENT audio, NOT subject to the "
                         "stale blocklist. Recovers the quarantined sessions for v2 (train only).")
    ap.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args()

    from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment

    split = {}
    for name in ("train", "dev", "test"):
        for rec in (pathlib.Path(args.split) / f"{name}_recordings.txt").read_text().split():
            split[rec] = name
    print(f"split: {sum(v=='train' for v in split.values())} train / "
          f"{sum(v=='dev' for v in split.values())} dev / "
          f"{sum(v=='test' for v in split.values())} test recordings")

    ex = pathlib.Path(args.exclude)
    stale = set(ex.read_text(encoding="utf-8").split()) if ex.exists() else set()
    if stale:
        print(f"excluding {len(stale)} stale sessions ({args.exclude}) — audio/segment mismatch")

    # gather unique (rec -> file) and all segments across dirs (skip stale sessions)
    rec_path, text, all_segs = {}, {}, []
    for d in args.dirs:
        if not (pathlib.Path(d) / "segments").exists():
            print(f"  [skip] {d} (no segments)")
            continue
        path, txt, segs = parse_dir(d)
        for rec, p in path.items():
            if rec not in stale:
                rec_path.setdefault(rec, p)
        text.update(txt)
        all_segs.extend(s for s in segs if s[1] not in stale)
        print(f"  {d}: {len(path):,} recs, {len(segs):,} segments parsed")

    # --reseg-dir OVERRIDE: recovered sessions come EXCLUSIVELY from the re-segmented dir.
    # Their stale segments were already dropped above (they're in the blocklist); here we add the
    # fresh ones (matching the current audio), NOT subject to the blocklist. They land in TRAIN
    # (split.get -> "train") since none of the quarantined sessions are in the frozen dev/test.
    if args.reseg_dir and (pathlib.Path(args.reseg_dir) / "segments").exists():
        rd = pathlib.Path(args.reseg_dir)
        path, txt, segs = parse_dir(rd)
        reseg_recs = set(path)
        all_segs = [s for s in all_segs if s[1] not in reseg_recs]   # drop any base leftovers
        for rec, p in path.items():
            rec_path[rec] = p                                        # override base path (same audio)
        text.update(txt)
        all_segs.extend(segs)                                        # fresh segments, not stale-filtered
        n_holdout = sum(1 for r in reseg_recs if split.get(r) in ("dev", "test"))
        print(f"  reseg {rd.name}: {len(reseg_recs):,} recovered sessions, {len(segs):,} fresh "
              f"segments (override blocklist)"
              + (f"  !! {n_holdout} are in dev/test — will NOT go to train" if n_holdout else ""))
    elif args.reseg_dir:
        print(f"  [warn] --reseg-dir {args.reseg_dir} has no segments/ — ignored")

    # parallel header-probe -> seekable file-source recordings (resampled to 16k)
    print(f"probing {len(rec_path):,} file headers with {args.jobs} workers…")
    recordings, missing = {}, 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for rec, p, sr, ch, frames in ex.map(probe, rec_path.items(), chunksize=256):
            if sr is None:
                missing += 1
                continue
            r = Recording(
                id=rec, sampling_rate=sr, num_samples=frames, duration=frames / sr,
                sources=[AudioSource(type="file", channels=list(range(ch)), source=p)])
            recordings[rec] = r.resample(TARGET_SR) if sr != TARGET_SR else r
    print(f"built {len(recordings):,} recordings ({missing} files missing/unreadable)")

    # one MonoCut per segment (cut.start offsets into the recording; supervision is
    # cut-relative). Clamp the segment end to the true recording length.
    buckets, empty, clamped = {"train": [], "dev": [], "test": []}, 0, 0
    for utt, rec, st, en in all_segs:
        r = recordings.get(rec)
        if r is None:
            continue
        txt = clean_text(text.get(utt, ""))
        if not txt:
            empty += 1
            continue
        end = min(en, r.duration)
        if end - st < 0.05:                      # empty/degenerate after clamp
            clamped += 1
            continue
        dur = round(end - st, 5)
        buckets[split.get(rec, "train")].append(MonoCut(
            id=utt, start=round(st, 5), duration=dur, channel=0, recording=r,
            supervisions=[SupervisionSegment(
                id=utt, recording_id=rec, start=0.0, duration=dur,
                channel=0, text=txt, language="Bulgarian")]))
    print(f"dropped {empty:,} empty-text + {clamped:,} degenerate segments")

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name, cl in buckets.items():
        CutSet.from_cuts(cl).to_file(out / f"cuts_{name}.jsonl.gz")
        print(f"  wrote {len(cl):,} cuts ({sum(c.duration for c in cl)/3600:.1f} h) "
              f"-> {out}/cuts_{name}.jsonl.gz")
    print("\nNext: compute fbank on these cuts + train BPE, then the icefall zipformer recipe.")


if __name__ == "__main__":
    main()
