#!/usr/bin/env python3
# Copyright 2024 Xiaomi Corp. (authors: OmniVoice-BG)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""L3 coverage completion: extend the stress lexicon into a full ALIGNMENT lexicon.

The L1/L2 lexicon holds only STRESS-MARKED forms (what the OmniVoice front-end marks),
so it covers ~57% of corpus tokens — the rest are unmarked function words / monosyllables
(in rechko but NULL-stressed) and irregulars (съм paradigm, absent from rechko). The
DNN-HMM aligner needs a pronunciation for EVERY word, so here we add:
  - all rechko surface forms not already present (word + derivative_form names), with
    stress assigned by rule: 1 vowel -> stressed; 2 vowels -> both (over-gen); >=3 (rare,
    unmarked) -> all positions;
  - a curated supplement of irregulars absent from rechko (config/bg/lexicon_supplement.json).

OOV beyond this (proper names / loanwords) is handled at runtime by the rule-based G2P
(build_lexicon_phones.g2p phonetizes any Cyrillic string).

    ./venv/bin/python extend_lexicon_coverage.py \
        --dump ../rechko/db.sql --lexicon data/bg_lexicon/lexicon.txt \
        --supplement config/bg/lexicon_supplement.json \
        --out data/bg_lexicon/lexicon_full.txt --measure data/bg/train.jsonl ...
"""
import argparse
import json
import mmap
import re
import shutil

from build_lexicon_phones import g2p
from build_stress_candidates import vowel_indices, mark_at, all_positions

NAME = re.compile(r"\(\d+,'((?:[^'\\]|\\.)*)'")
TOKEN = re.compile(r"[0-9]+|[a-zа-я]+")


def rechko_forms(dump):
    f = open(dump, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def regions(name):
        head = b"INSERT INTO `" + name + b"` VALUES"; pos = 0
        while True:
            s = mm.find(head, pos)
            if s < 0:
                break
            nxt = mm.find(b"INSERT INTO `", s + len(head))
            e = nxt if nxt > 0 else len(mm)
            yield s, e; pos = e

    forms = set()
    for tbl in (b"word", b"derivative_form"):
        for s, e in regions(tbl):
            reg = mm[s:e].decode("utf-8", "replace")
            for m in NAME.finditer(reg):
                nm = m.group(1).lower()
                if nm and re.fullmatch(r"[а-я-]+", nm):
                    forms.add(nm)
    return forms


def marks_for_unmarked(form):
    """Rule-assigned stress candidates for a form rechko left unmarked."""
    nv = len(vowel_indices(form))
    if nv == 0:
        return []
    if nv == 1:
        return [mark_at(form, 1)]          # monosyllable -> its vowel
    return [m for m in all_positions(form) if m]  # 2 (homograph hotspot) or rare >=3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="../rechko/db.sql")
    ap.add_argument("--lexicon", default="data/bg_lexicon/lexicon.txt")
    ap.add_argument("--supplement", default="config/bg/lexicon_supplement.json")
    ap.add_argument("--out", default="data/bg_lexicon/lexicon_full.txt")
    ap.add_argument("--measure", nargs="*", default=[], help="jsonl files to re-measure coverage")
    args = ap.parse_args()

    # names already in the L2 lexicon
    have = set()
    with open(args.lexicon, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                have.add(line.split("\t", 1)[0])
    print(f"L2 lexicon: {len(have)} surface forms")

    shutil.copyfile(args.lexicon, args.out)
    out = open(args.out, "a", encoding="utf-8")

    added_forms = added_prons = 0
    # curated supplement first (irregulars rechko lacks)
    sup = json.load(open(args.supplement, encoding="utf-8")).get("forms", {})
    for surface, marks in sup.items():
        surface = surface.lower()
        if surface in have:
            continue
        for mk in marks:
            ph = g2p(mk)
            if ph:
                out.write(f"{surface}\t{' '.join(ph)}\n"); added_prons += 1
        have.add(surface); added_forms += 1

    # full rechko vocabulary (unmarked forms get rule-assigned stress)
    print("parsing rechko full vocab ...", flush=True)
    for form in rechko_forms(args.dump):
        if form in have:
            continue
        prons = {tuple(g2p(m)) for m in marks_for_unmarked(form) if m}
        prons = {p for p in prons if p}
        if not prons:
            continue
        for p in prons:
            out.write(f"{form}\t{' '.join(p)}\n"); added_prons += 1
        have.add(form); added_forms += 1
    out.close()
    print(f"added: {added_forms} surface forms, {added_prons} pronunciations")
    print(f"full alignment lexicon: {args.out} ({len(have)} surface forms)")

    if args.measure:
        tot = cov = num = lat = oov = 0
        for path in args.measure:
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                for t in TOKEN.findall(json.loads(line).get("text", "").lower()):
                    tot += 1
                    if t.isdigit():
                        num += 1
                    elif t in have:
                        cov += 1
                    elif t.isascii():
                        lat += 1
                    else:
                        oov += 1
        wt = tot - num
        print(f"\n=== coverage after extension ===")
        print(f"tokens={tot}  covered={100*cov/wt:.2f}%  "
              f"Cyrillic-OOV={100*oov/wt:.2f}%  Latin={lat}  (target >=99%)")
        print("(remaining Cyrillic OOV = proper names/loanwords -> runtime G2P fallback)")


if __name__ == "__main__":
    main()
