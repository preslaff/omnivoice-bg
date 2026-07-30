#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Compute 80-dim fbank features for the Bulgarian ASR cuts (icefall/Zipformer prep).

Reads the seekable-file-source cuts from prep_lhotse_manifests.py (each cut seeks to
its segment slice in the source wav/flac and resamples 24k->16k on load), extracts
80-bin log-mel fbank at 16 kHz on CPU (num_jobs workers), and writes feature-bearing
cuts to data/fbank/bg_cuts_{split}.jsonl.gz — the manifests the Zipformer recipe reads.

CPU extraction (torchaudio Fbank) is used rather than GPU kaldifeat so nothing extra
needs building for Blackwell; ~649 h takes a while but is a one-time cost. Optional
3-fold speed perturbation (0.9/1.1) on TRAIN only is the usual icefall WER win — off
by default to get a first baseline fast (it triples feature size + extraction time).

    ./venv/bin/python compute_fbank_bg.py                 # train+dev+test, no perturb
    ./venv/bin/python compute_fbank_bg.py --speed-perturb # 3x train augmentation
"""
import argparse
import os
import pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="data/lhotse/bg")
    ap.add_argument("--out-dir", default="data/fbank")
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "train"])
    ap.add_argument("--num-mel-bins", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=min(24, os.cpu_count() or 8))
    ap.add_argument("--speed-perturb", action="store_true",
                    help="3-fold (0.9/1.1) speed perturbation on the train split")
    args = ap.parse_args()

    from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter

    extractor = Fbank(FbankConfig(num_mel_bins=args.num_mel_bins, sampling_rate=16000))
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        src = pathlib.Path(args.in_dir) / f"cuts_{split}.jsonl.gz"
        dst = out / f"bg_cuts_{split}.jsonl.gz"
        if dst.exists():
            print(f"[skip] {dst} exists")
            continue
        cuts = CutSet.from_file(src)
        n0 = len(cuts)
        if split == "train" and args.speed_perturb:
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
            print(f"{split}: {n0:,} -> {len(cuts):,} cuts after 3-fold speed perturb")
        print(f"{split}: extracting {args.num_mel_bins}-dim fbank for {len(cuts):,} cuts "
              f"with {args.jobs} workers -> {dst}")
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=str(out / f"bg_feats_{split}"),
            num_jobs=args.jobs,
            storage_type=LilcomChunkyWriter,
        )
        cuts.to_file(dst)
        dur = sum(c.duration for c in cuts) / 3600
        print(f"  wrote {len(cuts):,} cuts ({dur:.1f} h) -> {dst}")

    print("\nNext: train_bpe_bg.py (BPE-500), then adapt an icefall zipformer recipe "
          "to read data/fbank/bg_cuts_{train,dev,test}.jsonl.gz + data/lang_bpe_500.")


if __name__ == "__main__":
    main()
