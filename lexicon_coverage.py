#!/usr/bin/env python3
# Copyright 2024 Xiaomi Corp. (authors: OmniVoice-BG)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""L3: measure token coverage of the stress lexicon over our corpus, and characterise OOV.

Coverage target (Accentor §3.2): >=99% of word tokens. OOV falls into actionable buckets:
numbers/dates (-> verbalizer), Latin/foreign (-> G2P fallback, ablation A4), and genuine
Cyrillic gaps (-> investigate: inflections rechko missed, names, loanwords).

    ./venv/bin/python lexicon_coverage.py \
        --lexicon data/bg_lexicon/bg_stress_candidates.sqlite \
        --jsonl data/bg/train.jsonl data/bg/dev.jsonl data/bg400/train.jsonl
"""
import argparse
import json
import re
import sqlite3
from collections import Counter

TOKEN = re.compile(r"[0-9]+|[a-zа-я]+")


def load_names(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    names = {r[0] for r in con.execute("SELECT DISTINCT name FROM candidates")}
    con.close()
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/bg_lexicon/bg_stress_candidates.sqlite")
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--field", default="text")
    ap.add_argument("--top", type=int, default=40, help="top Cyrillic OOV to print")
    args = ap.parse_args()

    print("loading lexicon names ...", flush=True)
    names = load_names(args.lexicon)
    print(f"  {len(names)} surface forms")

    tot = covered = n_num = n_lat = n_cyr_oov = 0
    cyr_oov = Counter()
    lat = Counter()
    for path in args.jsonl:
        rows = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                txt = json.loads(line).get(args.field, "")
                for t in TOKEN.findall(txt.lower()):
                    tot += 1
                    if t.isdigit():
                        n_num += 1
                    elif t in names:
                        covered += 1
                    elif t.isascii():
                        n_lat += 1; lat[t] += 1
                    else:
                        n_cyr_oov += 1; cyr_oov[t] += 1
                rows += 1
        print(f"  {path}: {rows} lines", flush=True)

    word_tok = tot - n_num  # word tokens (exclude bare numbers)
    print(f"\n=== coverage ===")
    print(f"all tokens           : {tot}")
    print(f"  numbers            : {n_num} ({100*n_num/tot:.2f}%)")
    print(f"  covered (in lex)   : {covered} ({100*covered/tot:.2f}% of all)")
    print(f"  Latin/foreign      : {n_lat} ({100*n_lat/tot:.2f}%)")
    print(f"  Cyrillic OOV       : {n_cyr_oov} ({100*n_cyr_oov/tot:.2f}%)")
    if word_tok:
        print(f"word-token coverage (excl. bare numbers): "
              f"{100*covered/word_tok:.2f}%  (target >=99%)")
    cyr_word = covered + n_cyr_oov
    if cyr_word:
        print(f"Cyrillic-word coverage (excl. numbers+Latin): "
              f"{100*covered/cyr_word:.2f}%")

    print(f"\n=== top {args.top} Cyrillic OOV (the real gaps) ===")
    for w, c in cyr_oov.most_common(args.top):
        print(f"  {c:>6}  {w}")
    print(f"\ndistinct Cyrillic OOV types: {len(cyr_oov)}")
    print(f"\n=== top 15 Latin tokens ===")
    for w, c in lat.most_common(15):
        print(f"  {c:>6}  {w}")


if __name__ == "__main__":
    main()
