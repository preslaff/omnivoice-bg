#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Plot the milestone-A TDNN training curve from Kaldi's nnet3 diagnostic
logs (Kaldi emits no TensorBoard events; the objective lives in
exp/nnet3/<dir>/log/compute_prob_{train,valid}.N.log). Reads whatever
exists so far, so it works mid-training — re-run anytime to refresh.

Usage:  ./venv/bin/python plot_tdnn_progress.py
        ./venv/bin/python plot_tdnn_progress.py --dir exp/kaldi_pilot/exp/nnet3/tdnn_a
"""
import argparse
import glob
import os
import re

LL = re.compile(r"log-likelihood for 'output' is (-?[0-9.]+) per frame")
ACC = re.compile(r"accuracy for 'output' is (-?[0-9.]+) per frame")


def series(logdir, split, pat):
    pts = []
    for f in glob.glob(os.path.join(logdir, f"compute_prob_{split}.*.log")):
        m = re.search(rf"compute_prob_{split}\.(\d+)\.log$", f)
        if not m:
            continue
        it = int(m.group(1))
        txt = open(f, encoding="utf-8", errors="ignore").read()
        hit = pat.search(txt)
        if hit:
            pts.append((it, float(hit.group(1))))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="exp/kaldi_pilot/exp/nnet3/tdnn_a")
    ap.add_argument("--out", default="tdnn_progress.png")
    args = ap.parse_args()

    logdir = os.path.join(args.dir, "log")
    if not os.path.isdir(logdir):
        raise SystemExit(f"no log dir at {logdir}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for split, color in (("train", "tab:blue"), ("valid", "tab:orange")):
        ll = series(logdir, split, LL)
        ac = series(logdir, split, ACC)
        if ll:
            a1.plot(*zip(*ll), marker=".", ms=3, label=split, color=color)
        if ac:
            a2.plot(*zip(*ac), marker=".", ms=3, label=split, color=color)
        if ll:
            print(f"{split}: {len(ll)} points, "
                  f"latest iter {ll[-1][0]} LL={ll[-1][1]:.3f}"
                  + (f" acc={ac[-1][1]:.3f}" if ac else ""))
    a1.set_title("log-likelihood / frame (↑ toward 0 = better)")
    a1.set_xlabel("iteration"); a1.set_ylabel("log-likelihood"); a1.legend()
    a1.grid(alpha=0.3)
    a2.set_title("frame accuracy (↑ = better)")
    a2.set_xlabel("iteration"); a2.set_ylabel("accuracy"); a2.legend()
    a2.grid(alpha=0.3)
    fig.suptitle(f"milestone-A TDNN training — {args.dir}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
