#!/usr/bin/env python3
# Copyright 2024 Xiaomi Corp. (authors: OmniVoice-BG)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""L2 of the Accentor lexicon build: grapheme->phoneme over the multi-candidate stress
map, producing a Kaldi pronunciation lexicon with SEPARATE stressed/unstressed vowel
phonemes (what the annotation acoustic model needs, per Accentor §3.3).

Bulgarian orthography is near-phonemic, so G2P is rule-based:
  - vowels а е и о у ъ -> a e i o u y, each as a STRESSED (…1) or UNSTRESSED (…0) phone
    decided by the U+0301 mark from L1 (build_stress_candidates.py).
  - iotation/softness as a /j/ glide: я->j+a, ю->j+u, йо/ьо->j+o, й->j.
  - digraphs: дж->dzh, дз->dz, щ->sh t; affricates ц->c, ч->ch.
  - regressive voicing assimilation + word-final devoicing over obstruent clusters.

Each stress candidate of a surface form becomes one pronunciation line, so the lexicon
is the multi-pronunciation input for both the DNN-HMM aligner and the per-sentence
stress-decoding graph.

    ./venv/bin/python build_lexicon_phones.py --self-test
    ./venv/bin/python build_lexicon_phones.py \
        --cand data/bg_lexicon/bg_stress_candidates.sqlite \
        --out-lexicon data/bg_lexicon/lexicon.txt \
        --out-phones  data/bg_lexicon/phones.txt
"""
import argparse
import pathlib
import sqlite3

ACUTE = "́"  # U+0301

VMAP = {"а": "a", "е": "e", "и": "i", "о": "o", "у": "u", "ъ": "y"}
CMAP = {
    "б": "b", "в": "v", "г": "g", "д": "d", "ж": "zh", "з": "z", "к": "k",
    "л": "l", "м": "m", "н": "n", "п": "p", "р": "r", "с": "s", "т": "t",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh",
}
VOWEL_PHONES = ["a", "e", "i", "o", "u", "y"]
SONORANTS = {"l", "m", "n", "r", "j"}
# obstruent voiced<->voiceless pairs (for assimilation); h has no voiced pair
DEVOICE = {"b": "p", "v": "f", "g": "k", "d": "t", "z": "s", "zh": "sh",
           "dzh": "ch", "dz": "c"}
VOICE = {v: k for k, v in DEVOICE.items()}
VOICED = set(DEVOICE)          # voiced obstruents
VOICELESS = set(VOICE) | {"h"}  # voiceless obstruents
OBSTRUENTS = VOICED | VOICELESS


def g2p(marked: str):
    """Stress-marked Cyrillic spelling -> list of ASCII phones (vowels carry 0/1)."""
    s = marked.lower()
    phones = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == ACUTE:           # consumed with its vowel
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < n else ""
        # digraphs
        if c == "д" and nxt == "ж":
            phones.append("dzh"); i += 2; continue
        if c == "д" and nxt == "з":
            phones.append("dz"); i += 2; continue
        if c == "щ":
            phones += ["sh", "t"]; i += 1; continue
        # iotation / softness as a j-glide
        if c in ("я", "ю"):
            stressed = nxt == ACUTE
            v = "a" if c == "я" else "u"
            phones += ["j", v + ("1" if stressed else "0")]
            i += 1; continue
        if c == "й":
            phones.append("j"); i += 1; continue
        if c == "ь":             # soft sign: ьо -> j o; else drop
            if nxt == "о":
                st = (i + 2 < n and s[i + 2] == ACUTE)
                phones += ["j", "o" + ("1" if st else "0")]
                i += 2; continue
            i += 1; continue
        # plain vowel
        if c in VMAP:
            stressed = nxt == ACUTE
            phones.append(VMAP[c] + ("1" if stressed else "0"))
            i += 1; continue
        # plain consonant
        if c in CMAP:
            phones.append(CMAP[c]); i += 1; continue
        # anything else (hyphen, stray) -> skip
        i += 1
    return _assimilate(phones)


def _is_obstruent(p):
    return p in OBSTRUENTS


def _voicing_of(p):
    """'voiced' / 'voiceless' / None for a phone."""
    if p in VOICED:
        return "voiced"
    if p in VOICELESS:
        return "voiceless"
    return None


def _assimilate(phones):
    """Word-final devoicing + regressive voicing assimilation over obstruent clusters."""
    if not phones:
        return phones
    p = list(phones)
    # final devoicing: last phone if voiced obstruent
    if p[-1] in VOICED:
        p[-1] = DEVOICE[p[-1]]
    # regressive: an obstruent takes the voicing of the next obstruent
    for i in range(len(p) - 2, -1, -1):
        if not _is_obstruent(p[i]):
            continue
        nv = _voicing_of(p[i + 1]) if _is_obstruent(p[i + 1]) else None
        if p[i + 1] == "v":
            nv = None  # Bulgarian: в does NOT trigger regressive voicing (свобода→/svoboda/)
        if nv == "voiced" and p[i] in VOICELESS and p[i] in VOICE:
            p[i] = VOICE[p[i]]
        elif nv == "voiceless" and p[i] in VOICED:
            p[i] = DEVOICE[p[i]]
    return p


def write_phones(path):
    vowels = [v + s for v in VOWEL_PHONES for s in ("0", "1")]
    cons = ["b", "p", "v", "f", "g", "k", "d", "t", "z", "s", "zh", "sh",
            "c", "dz", "ch", "dzh", "h", "l", "m", "n", "r", "j"]
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("# vowels: <quality><0=unstressed,1=stressed>\n")
        for v in vowels:
            f.write(v + "\n")
        f.write("# consonants\n")
        for c in cons:
            f.write(c + "\n")
    return len(vowels) + len(cons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="data/bg_lexicon/bg_stress_candidates.sqlite")
    ap.add_argument("--out-lexicon", default="data/bg_lexicon/lexicon.txt")
    ap.add_argument("--out-phones", default="data/bg_lexicon/phones.txt")
    ap.add_argument("--self-test", action="store_true", help="g2p a few words and exit")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (debug)")
    args = ap.parse_args()

    if args.self_test:
        for w in ["за́мък", "замъ́к", "въ́лна", "вълна́", "те́ла", "тела́",
                  "гово́ри", "говори́", "бълга́рия", "държа́ва", "рабо́та",
                  "град", "хля́б", "ня́ма", "ю́ли", "ма́йка", "сва́тба",
                  "ща́стие", "о́бувка", "дъжд"]:
            print(f"  {w:<12} -> {' '.join(g2p(w))}")
        return

    nph = write_phones(args.out_phones)
    con = sqlite3.connect(f"file:{args.cand}?mode=ro", uri=True)
    q = "SELECT name, mark FROM candidates ORDER BY name"
    if args.limit:
        q += f" LIMIT {args.limit}"
    out = pathlib.Path(args.out_lexicon)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = empties = 0
    with open(out, "w", encoding="utf-8") as f:
        for name, mark in con.execute(q):
            ph = g2p(mark)
            if not ph:
                empties += 1
                continue
            f.write(f"{name}\t{' '.join(ph)}\n")
            rows += 1
    con.close()
    print(f"phones.txt : {args.out_phones} ({nph} phones)")
    print(f"lexicon.txt: {out} ({rows} pronunciations, {empties} empty skipped, "
          f"{out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
