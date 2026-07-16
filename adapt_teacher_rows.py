#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Adapt BgGPT sense-teacher rows to the Accentor training format.

The teacher (bggpt_teacher.py) emits per-word rows:
    {"word": "ходи", "sense": "а", "marked": "хо́ди", "sentence": "...", ...}
but train_accentor.py consumes whole-sentence rows:
    {"id", "text": <plain sentence>, "marked": <sentence w/ U+0301>,
     "mask_words": [[s,e], ...]}   # spans EXCLUDED from the loss
where the loss is taken over every vowel NOT inside a mask_words span.

A teacher row knows exactly ONE word's stress, so this inverts the usual
masking: mark only the target word in the sentence and mask EVERYTHING else
(so only the target word's vowels are supervised, with the vetted stress).

Rows are skipped (counted) when the target word is absent or appears more
than once in the sentence (ambiguous which occurrence BgGPT judged), or when
the marked/text lengths desync (the same guard train_accentor applies).

Usage:
    ./venv/bin/python adapt_teacher_rows.py \
        --in data/bggpt_teacher/sense_rows.jsonl \
        --out data/bggpt_teacher/teacher_marked.jsonl
    # then: build_accentor_finetune.py --extra-marked data/bggpt_teacher/teacher_marked.jsonl
"""
import argparse
import json

from accentor_model import ACUTE, VOWELS
from select_chitanka_sentences import WORD_RE


def stressed_vowel_index(marked):
    """0-based index (among the word's vowels) of the vowel carrying U+0301."""
    vc = 0
    for ch in marked:
        if ch == ACUTE:
            return vc - 1
        if ch in VOWELS:
            vc += 1
    return -1


def adapt(word, marked, sentence):
    """-> training-format dict, or None (skip) with a reason."""
    vi = stressed_vowel_index(marked)
    if vi < 0:
        return None, "no-acute"
    hits = [m for m in WORD_RE.finditer(sentence)
            if m.group().lower().replace("ѝ", "и") == word]
    if len(hits) != 1:
        return None, ("absent" if not hits else "multi")
    m = hits[0]
    s, e = m.start(), m.end()
    orig = sentence[s:e]                       # original casing preserved
    # position of the vi-th vowel within the original-case word
    seen, pos = 0, -1
    for i, ch in enumerate(orig):
        if ch.lower().replace("ѝ", "и") in VOWELS:
            if seen == vi:
                pos = i
                break
            seen += 1
    if pos < 0:
        return None, "vowel-index-oob"
    marked_sent = sentence[:s] + orig[:pos + 1] + ACUTE + orig[pos + 1:] + sentence[e:]
    # desync guard identical to train_accentor.row_arrays
    if sum(1 for c in marked_sent if c != ACUTE) != len(sentence):
        return None, "desync"
    return {
        "id": f"tch_{word}",
        "text": sentence,
        "marked": marked_sent,
        "mask_words": [[0, s], [e, len(sentence)]],   # supervise only the word
        "source": "bggpt-teacher",
    }, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/bggpt_teacher/sense_rows.jsonl")
    ap.add_argument("--out", default="data/bggpt_teacher/teacher_marked.jsonl")
    args = ap.parse_args()

    from collections import Counter
    reasons = Counter()
    n_in = n_out = 0
    with open(args.inp, encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8") as of:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            r = json.loads(line)
            row, why = adapt(r["word"], r["marked"], r["sentence"])
            reasons[why] += 1
            if row is not None:
                of.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"{n_in:,} teacher rows -> {n_out:,} training rows ({args.out})")
    print("skips: " + ", ".join(f"{k}={v}" for k, v in reasons.items()
                                 if k != "ok"))


if __name__ == "__main__":
    main()
