# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Interactive CLI vetting of harvested stress labels.

Samples N tokens per observed stress variant of the chosen words, plays each
clip (ffplay, word +/- context), shows the sentence with the word highlighted
and the aligner's marked form, and records a single-keypress judgment:

  o  OK    — the audio says the marked stress
  b  BAD   — the audio says a different stress
  u  UNSURE
  r  replay          m  replay with more context
  s  skip (no record)         q  quit (summary + exit)

Judgments append to --results (JSONL); rerunning skips already-judged clips,
so sessions are resumable. A tally per (word, variant) prints at the end —
that is the measured label accuracy for the fine-tune go/no-go.

    ./venv/bin/python vet_stress_cli.py \
        --labels data/kaldi/stress_labels_parliament2.jsonl \
        --words чета,говори,всички,може,още,колеги,съюз,страна
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import termios
import tty
from collections import Counter, defaultdict

BOLD, DIM, GREEN, RED, YEL, RST = ("\033[1m", "\033[2m", "\033[32m",
                                   "\033[31m", "\033[33m", "\033[0m")


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":  # Ctrl-C in raw mode
        raise KeyboardInterrupt
    return ch.lower()


_TMP_WAV = os.path.join(tempfile.gettempdir(), f"vet_clip_{os.getpid()}.wav")


def play(row, pad):
    """ffplay's own -ss HANGS on the crawl's 2 h session FLACs (ffmpeg writes
    them without a seektable; ffmpeg's input seek is instant, ffplay's seek
    path is not — diagnosed 2026-07-05). Extract the clip with ffmpeg, then
    play the small wav; hard timeout so a wedged player can't freeze the app."""
    t0 = max(0.0, row["t0"] - pad)
    dur = row["dur"] + 2 * pad
    ext = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t0:.2f}",
         "-t", f"{dur:.2f}", "-i", row["audio_path"], _TMP_WAV], check=False)
    if ext.returncode != 0:
        print(f"{RED}[extract failed: {row['audio_path']}]{RST}", end="",
              flush=True)
        return
    # paplay, not ffplay: it targets the PipeWire default sink directly.
    # ffplay's SDL audio layer picked a different (inaudible) device on this
    # box even with the default sink set correctly (2026-07-05).
    try:
        subprocess.run(["paplay", _TMP_WAV], check=False, timeout=dur + 15)
    except FileNotFoundError:
        subprocess.run(
            ["ffplay", "-hide_banner", "-loglevel", "error", "-autoexit",
             "-nodisp", _TMP_WAV], check=False, timeout=dur + 15)
    except subprocess.TimeoutExpired:
        print(f"{RED}[player timed out — check the audio output device: "
              f"pactl get-default-sink]{RST}", end="", flush=True)


def sample_items(labels_path, words, per_variant, seed):
    by_wm = defaultdict(list)
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["audio_path"]:
                by_wm[(r["word"], r["marked"])].append(r)
    if not words:  # both-variant words with the most minority evidence
        by_word = defaultdict(list)
        for (w, _), rows in by_wm.items():
            by_word[w].append(len(rows))
        words = [w for w, c in sorted(by_word.items(),
                                      key=lambda kv: -min(kv[1]))
                 if len(c) > 1][:12]
    rng = random.Random(seed)
    items = []
    for w in words:
        variants = sorted((m for (ww, m) in by_wm if ww == w),
                          key=lambda m: -len(by_wm[(w, m)]))
        for m in variants:
            rows = by_wm[(w, m)]
            for r in rng.sample(rows, min(per_variant, len(rows))):
                items.append({**r, "clip_id": f"{r['utt']}:{r['t0']}:{m}",
                              "n_variant_tokens": len(rows)})
    return items


def show(row, i, n):
    sent = row["sentence"]
    w = row["word"]
    j = sent.find(w)
    if j >= 0:
        sent = sent[:j] + BOLD + YEL + sent[j:j + len(w)] + RST + DIM \
            + sent[j + len(w):]
    print(f"\n{DIM}[{i}/{n}]{RST}  {BOLD}{row['word']}{RST} -> "
          f"{BOLD}{GREEN}{row['marked']}{RST}  "
          f"{DIM}({row['n_variant_tokens']} tokens of this variant, "
          f"{row['utt']} @ {row['t0']:.1f}s){RST}")
    print(f"  {DIM}{sent}{RST}")
    print(f"  {DIM}[o]k [b]ad [u]nsure [r]eplay [m]ore-context [w]ide "
          f"[s]kip [q]uit — judge the highlighted word only{RST} ",
          end="", flush=True)


def summary(results):
    tally = defaultdict(Counter)
    for r in results:
        tally[(r["word"], r["marked"])][r["verdict"]] += 1
    print(f"\n{BOLD}=== vetting summary ==={RST}")
    tot = Counter()
    for (w, m), c in sorted(tally.items()):
        judged = c["ok"] + c["bad"]
        acc = f"{c['ok'] / judged:.0%}" if judged else "-"
        print(f"  {m:<16} ok={c['ok']:<3} bad={c['bad']:<3} "
              f"unsure={c['unsure']:<3} acc={acc}")
        tot.update(c)
    judged = tot["ok"] + tot["bad"]
    if judged:
        print(f"{BOLD}  overall: {tot['ok']}/{judged} OK "
              f"({tot['ok'] / judged:.0%}), {tot['unsure']} unsure{RST}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/kaldi/stress_labels_parliament2.jsonl")
    ap.add_argument("--results", default="data/kaldi/vetting_results.jsonl")
    ap.add_argument("--words", default=None, help="comma-separated; default "
                    "= top-12 split words by minority evidence")
    ap.add_argument("--per-variant", type=int, default=5)
    ap.add_argument("--pad", type=float, default=1.0,
                    help="context seconds around the word; the clip contains "
                         "NEIGHBOURING words too — judge only the highlighted "
                         "one ([w] plays a wide window when unsure)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    words = args.words.split(",") if args.words else None
    items = sample_items(args.labels, words, args.per_variant, args.seed)

    done = {}
    if os.path.exists(args.results):
        with open(args.results, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["clip_id"]] = r
    todo = [it for it in items if it["clip_id"] not in done]
    print(f"{len(items)} sampled clips, {len(done)} already judged, "
          f"{len(todo)} to go  (results -> {args.results})")

    verdicts = {"o": "ok", "k": "ok", "b": "bad", "x": "bad", "u": "unsure"}
    n_new = 0
    try:
        with open(args.results, "a", encoding="utf-8") as out:
            for idx, row in enumerate(todo, 1):
                pad = args.pad
                show(row, idx, len(todo))
                play(row, pad)
                while True:
                    ch = getch()
                    if ch == "r":
                        play(row, pad)
                    elif ch == "m":
                        pad += 0.8
                        play(row, pad)
                    elif ch == "w":  # whole neighbourhood, sentence-scale
                        play(row, pad + 3.0)
                    elif ch == "s":
                        print("skip")
                        break
                    elif ch == "q":
                        raise KeyboardInterrupt
                    elif ch in verdicts:
                        v = verdicts[ch]
                        color = {"ok": GREEN, "bad": RED, "unsure": YEL}[v]
                        print(f"{color}{v}{RST}")
                        rec = {"clip_id": row["clip_id"], "word": row["word"],
                               "marked": row["marked"], "utt": row["utt"],
                               "t0": row["t0"], "dur": row["dur"],
                               "sentence": row["sentence"], "verdict": v}
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                        done[row["clip_id"]] = rec
                        n_new += 1
                        break
    except KeyboardInterrupt:
        print("\nstopped.")
    print(f"{n_new} new judgments this session.")
    summary(list(done.values()))


if __name__ == "__main__":
    main()
