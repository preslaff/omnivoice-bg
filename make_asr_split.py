#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Freeze a train/dev/TEST split for the Bulgarian ASR by WHOLE recording.

PARLIAMENT-ONLY holdout (no leakage). The parliament recording IS the whole
session (plsten_<ID>), so held-out sessions are clean. Audiobook "recordings" are
per-CLIP (~7 s), so holding out clips would LEAK the same book/reader into TRAIN —
therefore ALL audiobooks go to TRAIN, and the benchmark is spontaneous parliament
speech (the Finnish-paper framing; the publishable, interesting domain). The
frozen id lists are consumed identically by the Kaldi chain baseline and the
icefall/Zipformer recipe, so the two systems are comparable (docs/BG_ASR_PLAN.md §2).

Reads Kaldi `segments` (`utt rec start end`) from the given cleaned data dirs,
sums per-recording duration, and greedily fills the held-out targets from a
seeded shuffle. Deterministic (seed) → reproducible split.

    ./venv/bin/python make_asr_split.py            # defaults below
    ./venv/bin/python make_asr_split.py --dirs exp/kaldi_pilot/data/train_cleaned \
        exp/kaldi_pilot/data/parl_new_cleaned exp/kaldi_pilot/data/parl_new2_cleaned
"""
import argparse
import pathlib
import random
from collections import defaultdict


def rec_durations(seg_dirs):
    """recording-id -> total seconds, and which dir it came from."""
    dur = defaultdict(float)
    for d in seg_dirs:
        seg = pathlib.Path(d) / "segments"
        if not seg.exists():
            print(f"  [skip] {seg} absent")
            continue
        n = 0
        for line in seg.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 4:
                dur[p[1]] += float(p[3]) - float(p[2])
                n += 1
        print(f"  {d}: {n:,} segments over {sum(1 for r in dur)} recs so far")
    return dur


def is_parliament(rec):
    return rec.startswith("plsten")


def fill(pool, target_h, rng):
    """Greedily take whole recordings (shuffled) until ~target hours."""
    rng.shuffle(pool)
    picked, acc = [], 0.0
    for rec, sec in pool:
        if acc >= target_h * 3600:
            break
        picked.append(rec); acc += sec
    return picked, acc / 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=[
        "exp/kaldi_pilot/data/train_cleaned",
        "exp/kaldi_pilot/data/parl_new_cleaned",
        "exp/kaldi_pilot/data/parl_new2_cleaned",   # folded in if it exists
    ])
    ap.add_argument("--out", default="data/lhotse/bg/split")
    ap.add_argument("--test-parl-h", type=float, default=15.0)
    ap.add_argument("--dev-parl-h", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dur = rec_durations(args.dirs)
    parl = [(r, s) for r, s in dur.items() if is_parliament(r)]
    audio = [(r, s) for r, s in dur.items() if not is_parliament(r)]
    ph = sum(s for _, s in parl) / 3600
    ah = sum(s for _, s in audio) / 3600
    print(f"\ntotal: {len(dur):,} recs, {(ph+ah):.1f} h  "
          f"(parliament {len(parl)} recs/{ph:.1f} h · audiobook {len(audio)} recs/{ah:.1f} h)")

    rng = random.Random(args.seed)
    # PARLIAMENT-ONLY holdout (whole sessions, no leakage); all audiobooks -> TRAIN.
    test_p, tph = fill(list(parl), args.test_parl_h, rng)
    dev_p, dph = fill([x for x in parl if x[0] not in set(test_p)], args.dev_parl_h, rng)

    held = set(test_p) | set(dev_p)
    train = [r for r in dur if r not in held]
    trh = sum(dur[r] for r in train) / 3600
    train_parl_h = sum(dur[r] for r in train if is_parliament(r)) / 3600

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "test_recordings.txt").write_text("\n".join(sorted(test_p)) + "\n")
    (out / "dev_recordings.txt").write_text("\n".join(sorted(dev_p)) + "\n")
    (out / "train_recordings.txt").write_text("\n".join(sorted(train)) + "\n")

    print(f"\nfrozen split (parliament-only holdout) -> {out}/")
    print(f"  TEST : {len(test_p)} parliament sessions ({tph:.1f} h)")
    print(f"  DEV  : {len(dev_p)} parliament sessions ({dph:.1f} h)")
    print(f"  TRAIN: {len(train):,} recs ({trh:.1f} h) — parliament {train_parl_h:.1f} h "
          f"+ all audiobook {ah:.1f} h")
    assert not (held & set(train)), "LEAK: a held-out rec is in train"
    print("  ✓ no train/held-out overlap")


if __name__ == "__main__":
    main()
