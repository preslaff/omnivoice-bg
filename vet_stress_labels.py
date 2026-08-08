# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Native-vetting sheet for harvested stress labels: sample N tokens per
observed variant of the chosen words and emit ready-to-run ffplay commands
(word +/- context) so a native speaker can judge whether the aligner's stress
choice matches the audio.

    ./venv/bin/python vet_stress_labels.py \
        --labels data/kaldi/stress_labels_parliament2.jsonl \
        --words вълна,чета,говори,всички,може,още --out vetting_sheet.md
"""
import argparse
import json
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/kaldi/stress_labels_parliament2.jsonl")
    ap.add_argument("--words", default=None,
                    help="comma-separated surface forms; default = the 12 "
                         "split words with the most minority evidence")
    ap.add_argument("--per-variant", type=int, default=5)
    ap.add_argument("--pad", type=float, default=0.7,
                    help="context seconds around the word in the clip")
    ap.add_argument("--out", default="vetting_sheet.md")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_wm = defaultdict(list)
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["audio_path"]:
                by_wm[(r["word"], r["marked"])].append(r)

    if args.words:
        words = args.words.split(",")
    else:  # words with both variants, most minority evidence first
        by_word = defaultdict(list)
        for (w, m), rows in by_wm.items():
            by_word[w].append(len(rows))
        words = [w for w, c in sorted(by_word.items(),
                                      key=lambda kv: -min(kv[1]))
                 if len(c) > 1][:12]

    rng = random.Random(args.seed)
    n_clips = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# Stress-label vetting sheet\n\n"
                  "For each clip: run the command, listen, and mark the row "
                  "OK (audio says the marked stress) or BAD.\n")
        for w in words:
            variants = sorted((m for (ww, m) in by_wm if ww == w),
                              key=lambda m: -len(by_wm[(w, m)]))
            if not variants:
                out.write(f"\n## {w} — no labeled tokens\n")
                continue
            counts = "  ".join(f"{m}:{len(by_wm[(w, m)])}" for m in variants)
            out.write(f"\n## {w}   ({counts})\n\n")
            for m in variants:
                rows = rng.sample(by_wm[(w, m)],
                                  min(args.per_variant, len(by_wm[(w, m)])))
                out.write(f"### expected: {m}\n\n")
                for r in rows:
                    t0 = max(0.0, r["t0"] - args.pad)
                    dur = r["dur"] + 2 * args.pad
                    sent = r["sentence"]
                    i = sent.find(w)
                    ctx = sent[max(0, i - 40):i + len(w) + 40] if i >= 0 else sent[:80]
                    out.write(f"- `…{ctx}…`  [ ] OK  [ ] BAD\n"
                              f"  `ffplay -hide_banner -loglevel error "
                              f"-autoexit -nodisp -ss {t0:.2f} -t {dur:.2f} "
                              f"{r['audio_path']}`\n")
                    n_clips += 1
    print(f"{n_clips} clips across {len(words)} words -> {args.out}")


if __name__ == "__main__":
    main()
