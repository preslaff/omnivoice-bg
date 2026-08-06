#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Merge training CutSets into ONE general-purpose BG ASR train manifest, with a leakage guard.

Phase 2 of the general-purpose pivot (docs/BG_ASR_ROADMAP.md): combine the v1 parliament+AB400
train, the recovered parliament, and the FLEURS / Common Voice **train** splits into a single
`data/fbank/bg_cuts_train_gp.jsonl.gz`. All inputs must already be 80-dim fbank @ 16 kHz
(compute_fbank_bg.py / prep_fleurs_bg.py / prep_commonvoice_bg.py) — this only concatenates cut
records (which point at the existing feature files), it does NOT recompute features.

HELD-OUT GUARD (the reason this script exists). Every eval utterance must be absent from train.
We collect recording-ids AND cut-ids from each `--holdout` eval manifest (+ the frozen parliament
`{dev,test}_recordings.txt`), and drop any train cut whose recording-id or cut-id is in that set.
This is self-healing: even if a *superset* split is passed in (Common Voice `validated` ⊇ `test`,
so it contains the CV test clips), the guard removes the offending clips and reports the count.
It streams (lazy read + incremental write) so the 419k-cut parliament set doesn't blow memory.

    ./venv/bin/python merge_train_corpus.py \
        --train data/fbank/bg_cuts_train.jsonl.gz \
        --train data/fbank/bg_cuts_train_reseg.jsonl.gz \
        --train data/fbank/bg_cuts_fleurs_train.jsonl.gz \
        --train data/fbank/bg_cuts_cv_train.jsonl.gz \
        --holdout data/fbank/bg_cuts_dev.jsonl.gz \
        --holdout data/fbank/bg_cuts_test.jsonl.gz \
        --holdout data/fbank/bg_cuts_fleurs.jsonl.gz \
        --holdout data/fbank/bg_cuts_cv.jsonl.gz \
        --out data/fbank/bg_cuts_train_gp.jsonl.gz
    #  add --dry-run to report the merge without writing.
"""
import argparse
import pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="append", default=[],
                    help="a training cut manifest (repeatable); missing files are skipped+warned")
    ap.add_argument("--holdout", action="append", default=[],
                    help="an eval cut manifest whose recording-ids AND cut-ids must NOT appear "
                         "in train (repeatable)")
    ap.add_argument("--split-dir", default="data/lhotse/bg/split",
                    help="also treat {dev,test}_recordings.txt here as held-out session ids")
    ap.add_argument("--out", default="data/fbank/bg_cuts_train_gp.jsonl.gz")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="write in source order (default: GLOBAL seeded shuffle — a source-ordered "
                         "manifest makes the sampler's windowed shuffle sweep registers in sequence "
                         "each epoch, causing dev sawtooth + end-of-epoch recency bias; icefall's "
                         "train-all-shuf convention. Lesson from GP-v2's 2026-08-06 false start.)")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    from lhotse import CutSet

    # --- 1. build the held-out id set (recording-ids + cut-ids) ---
    held = set()
    for h in args.holdout:
        p = pathlib.Path(h)
        if not p.exists():
            print(f"  [warn] holdout {h} missing — skipped")
            continue
        n0 = len(held)
        for c in CutSet.from_file(p):
            held.add(c.id)
            if c.recording_id:
                held.add(c.recording_id)
        print(f"  holdout {p.name}: +{len(held)-n0:,} ids")
    sd = pathlib.Path(args.split_dir)
    for f in ("dev_recordings.txt", "test_recordings.txt"):
        fp = sd / f
        if fp.exists():
            ids = fp.read_text(encoding="utf-8").split()
            held.update(ids)
            print(f"  split {f}: +{len(ids)} session ids")
    print(f"held-out id set: {len(held):,} ids\n")

    # --- 2. stream train sources through the guard into the merged manifest ---
    def kept_cuts():
        seen = set()
        stats = {}
        for t in args.train:
            p = pathlib.Path(t)
            if not p.exists():
                print(f"  [warn] train {t} missing — skipped")
                continue
            n = kept = leak = dup = 0
            dur = 0.0
            first_dim = None
            for c in CutSet.from_file(p):
                n += 1
                if first_dim is None and c.has_features:
                    first_dim = c.features.num_features
                if c.id in held or (c.recording_id and c.recording_id in held):
                    leak += 1
                    continue
                if c.id in seen:                    # de-dup across overlapping sources
                    dup += 1
                    continue
                seen.add(c.id)
                kept += 1
                dur += c.duration
                yield c
            dimflag = "" if first_dim in (None, 80) else f"  !! feat-dim={first_dim} (expected 80)"
            stats[p.name] = (n, kept, leak, dup, dur)
            print(f"  {p.name}: {n:,} in -> {kept:,} kept "
                  f"({leak:,} leaked-removed, {dup:,} dup) {dur/3600:.1f} h{dimflag}")
        main.stats = stats

    out = pathlib.Path(args.out)
    if args.dry_run:
        total = sum(1 for _ in kept_cuts())
        print(f"\n[dry-run] would write {total:,} cuts -> {out} (nothing written)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.no_shuffle:
        with CutSet.open_writer(out) as w:
            total = 0
            for c in kept_cuts():
                w.write(c)
                total += 1
    else:
        import random
        random.seed(args.seed)
        cuts = list(kept_cuts())                    # eager: global shuffle needs all in memory
        random.shuffle(cuts)
        CutSet.from_cuts(cuts).to_file(out)
        total = len(cuts)
        print(f"  globally shuffled (seed {args.seed})")
    hrs = sum(v[4] for v in main.stats.values()) / 3600
    print(f"\nmerged {total:,} cuts ({hrs:.1f} h) -> {out}")
    print("Next: train the Zipformer on this manifest (data.json --> bg_cuts_train_gp.jsonl.gz);")
    print("      eval on frozen parliament dev/test + held-out FLEURS/CV test (now in-distribution).")


if __name__ == "__main__":
    main()
