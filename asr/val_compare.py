#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Live side-by-side validation-loss table across BG ASR training runs.

`grep validation:` on one run tells you the number but not whether it is GOOD. This lines the
runs up by epoch so an in-flight run is read against the finished ones:

    v1     649 h, offline   -> 5.29 % test
    gp-v2  1,745 h, offline -> 4.12 % test
    gp-s   1,745 h, STREAMING (causal) — expected to sit ABOVE gp-v2 for the whole run; that
           gap is the price of giving up right context and is the point of the experiment.

icefall validates several times per epoch, so the BEST (lowest) value per epoch is shown; for
the epoch in flight that is a best-so-far and will only improve.

    ./venv/bin/python asr/val_compare.py                  # one shot
    watch -n 120 '~/WORK/OmniVoice/venv/bin/python ~/WORK/OmniVoice/asr/val_compare.py'
"""
import argparse
import datetime
import glob
import pathlib
import re

REC = pathlib.Path.home() / "icefall/egs/bg_parliament/ASR"
RUNS = [("v1", "zipformer/exp"), ("gp-v2", "zipformer/exp_gp"), ("gp-s", "zipformer/exp_gp_stream")]

VAL = re.compile(r"Epoch (\d+), validation: loss=([0-9.]+)")
BATCH = re.compile(r"Epoch (\d+), batch (\d+),")


def scan(exp_dir):
    """(best val loss per epoch, last (epoch,batch), first/last batch timestamps)."""
    best, last_bt, stamps = {}, None, []
    for f in sorted(glob.glob(str(REC / exp_dir / "log" / "log-train-*"))):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                m = VAL.search(line)
                if m:
                    e, v = int(m.group(1)), float(m.group(2))
                    best[e] = min(best.get(e, 9e9), v)
                    continue
                m = BATCH.search(line)
                if m:
                    last_bt = (int(m.group(1)), int(m.group(2)))
                    try:
                        stamps.append(datetime.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
                    except ValueError:
                        pass
    return best, last_bt, stamps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches-per-epoch", type=int, default=14000)
    ap.add_argument("--num-epochs", type=int, default=30)
    args = ap.parse_args()

    data = {name: scan(d) for name, d in RUNS}
    present = [n for n, _ in RUNS if data[n][0]]
    epochs = sorted({e for n in present for e in data[n][0]})

    print(f"{'ep':>3} " + "".join(f"{n:>9}" for n in present) + "   gp-s−gp-v2")
    print("-" * (4 + 9 * len(present) + 13))
    for e in epochs:
        row = f"{e:3d} "
        for n in present:
            v = data[n][0].get(e)
            row += f"{v:9.4f}" if v is not None else f"{'—':>9}"
        a, b = data.get("gp-s", ({},))[0].get(e), data.get("gp-v2", ({},))[0].get(e)
        row += f"{a - b:+13.4f}" if (a is not None and b is not None) else ""
        print(row)

    for n in present:
        best, last_bt, stamps = data[n]
        if not last_bt or len(stamps) < 3:
            continue
        ep, bt = last_bt
        done = (ep - 1) * args.batches_per_epoch + bt
        total = args.num_epochs * args.batches_per_epoch
        # Rate from the last stretch only, so a resumed/paused run is not averaged in.
        tail = stamps[-min(len(stamps), 60):]
        span = (tail[-1] - tail[0]).total_seconds()
        rate = span / max(len(tail) - 1, 1) / 50 if span > 0 else 0     # logged every 50 batches
        age = (datetime.datetime.now() - stamps[-1]).total_seconds()
        state = "RUNNING" if age < 600 else f"idle {age/3600:.1f} h"
        line = (f"\n{n}: epoch {ep}/{args.num_epochs}, batch {bt:,} "
                f"({100*done/total:.1f}% of run) · {state}")
        if rate > 0 and state == "RUNNING":
            eta_h = (total - done) * rate / 3600
            line += (f" · {rate:.3f} s/batch · {rate*args.batches_per_epoch/3600:.2f} h/epoch"
                     f" · ETA {eta_h:.1f} h "
                     f"({(datetime.datetime.now() + datetime.timedelta(hours=eta_h)):%a %H:%M})")
        print(line)


if __name__ == "__main__":
    main()
