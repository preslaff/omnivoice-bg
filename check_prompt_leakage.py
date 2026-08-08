#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Verify the ASR playground prompt bank does not overlap the training corpus text.

Read mode shows a volunteer a sentence and treats that sentence as the reference. If the
model was TRAINED on the same sentence, read mode measures memorization rather than
recognition, and the resulting eval rows are worthless. This checks every prompt against
data/lm/lm_text_gp.txt — the 14.4M-word text of the GP-v2 training corpus, produced by
prep_lm_text.py --cuts from the same manifest the acoustic model saw.

Comparison is on the shared scoring normalization (bg_text_norm), not raw strings, so
"127 лева" and "сто двадесет и седем лева" are correctly seen as the same sentence.

Also flags prompts that are too long to read inside the recorder's cap.

    ./venv/bin/python check_prompt_leakage.py                 # exits non-zero on a hit
    ./venv/bin/python check_prompt_leakage.py --max-words 25
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bg_text_norm import normalize

REPO = pathlib.Path(__file__).resolve().parent


def read_prompts(path):
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append((n, s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(REPO / "config/bg/asr_prompts.txt"))
    ap.add_argument("--corpus", default=str(REPO / "data/lm/lm_text_gp.txt"))
    ap.add_argument("--max-words", type=int, default=25,
                    help="warn above this (the recorder hard-stops at 15 s)")
    args = ap.parse_args()

    prompts = read_prompts(pathlib.Path(args.prompts))
    print(f"prompts: {len(prompts)}  <- {args.prompts}")

    dupes = {}
    for ln, s in prompts:
        dupes.setdefault(normalize(s), []).append(ln)
    dup = {k: v for k, v in dupes.items() if len(v) > 1}
    if dup:
        print(f"\n{len(dup)} DUPLICATE prompt(s):")
        for k, lns in dup.items():
            print(f"  lines {lns}: {k[:70]}")

    long = [(ln, s, len(s.split())) for ln, s in prompts if len(s.split()) > args.max_words]
    if long:
        print(f"\n{len(long)} prompt(s) over {args.max_words} words (may not fit 15 s):")
        for ln, s, w in long:
            print(f"  line {ln} ({w}w): {s[:70]}…")

    corpus = pathlib.Path(args.corpus)
    if not corpus.exists():
        print(f"\nSKIPPED leakage check — {corpus} absent "
              f"(build it with prep_lm_text.py --cuts <gp manifest>)")
        return 1 if (dup or long) else 0

    print(f"\nloading corpus text: {args.corpus}")
    seen = set()
    n_lines = 0
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            seen.add(normalize(line))
    print(f"  {n_lines:,} lines / {len(seen):,} distinct normalized sentences")

    hits = [(ln, s) for ln, s in prompts if normalize(s) in seen]
    if hits:
        print(f"\nLEAK: {len(hits)} prompt(s) appear verbatim in the training corpus:")
        for ln, s in hits:
            print(f"  line {ln}: {s}")
        print("\nReplace them — read mode on these would measure memorization.")
        return 2

    print(f"\nOK: 0 of {len(prompts)} prompts appear in the training corpus.")
    return 1 if (dup or long) else 0


if __name__ == "__main__":
    sys.exit(main())
