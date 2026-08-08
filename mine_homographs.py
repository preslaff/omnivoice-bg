# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mine BOTH readings of verb-imperative homographs from chitanka text.

For homograph verbs (гово́ри 3sg vs говори́ imperative), a context-aware model
learns the boundary only from CORRECT labels of BOTH readings. Rather than
purify the hard, rare, ambiguous imperative side, we label each reading via
its own HIGH-PRECISION cue and SKIP the genuinely-ambiguous middle (a wrong
label inverts a class — fine-tune #2 lesson):

  3SG  (root, e.g. но́си)  <- subject/subordinate immediately before
                             (той/тя/то/да/че/ще/ако/който…), OR a PROCLITIC
                             object clitic (го носи, му прави). High-yield and
                             clean; works in narration too, not just dialogue.
  IMP  (final, e.g. носи́) <- imperative particle before (хайде/недей/моля…);
                             PERFECTIVE verbs also: clause-initial in a short
                             '!' dialogue clause (clean immediate command).
  SKIP (ambiguous)         <- 'не'/'я' before (negation / particle-pronoun
                             collision); a bare clause-initial '!' on an
                             IMPERFECTIVE verb (indignant 3sg 'Хо́ди по жени!',
                             surface-identical to the durative imperative).

Marks come from the curated lexicon (config/bg/homograph_verbs.json,
DB-verified). Only the target token is supervised; every other word is masked,
so each row is a pure one-token teaching signal in train_accentor's
marked.jsonl format.

Usage:
    ./venv/bin/python mine_homographs.py --corpus data/chitanka \
        --lexicon config/bg/homograph_verbs.json \
        --out data/chitanka/homograph_contexts.jsonl \
        --max-per-class 2000 --show 30
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

from bg_chitanka_clean import split_sentences
from select_chitanka_sentences import WORD_RE

ACUTE = "́"

# Imperative particles immediately before the verb (both aspects).
PARTICLES_BEFORE = {"недей", "недейте", "хайде", "моля", "стига", "ела", "дай"}
# Subject / subordinator before the verb → 3sg (root). "те" excluded (3pl
# takes -ят, not -и) as is "аз/ние/вие" (wrong agreement for a 3sg -и form).
SUBJ_3SG = {"той", "тя", "то", "кой", "който", "която", "което", "които",
            "никой", "всеки", "да", "че", "ще", "ако", "когато", "докато",
            "защото", "щом", "като", "дето", "нали", "който", "понеже"}
# Proclitic object clitics (clitic BEFORE the verb) → 3sg/subjunctive, root.
# "я" excluded (collides with the imperative particle "я"); "те" excluded
# (subject/object ambiguity).
PROCLITIC_3SG = {"го", "ги", "ме", "ни", "ви", "ми", "ти", "му", "й", "ѝ", "им"}
# Skip entirely — genuinely ambiguous before the verb.
AMBIG_PREV = {"не", "я"}


def load_lexicon(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["homographs"], set(d.get("perfective", []))


def apply_mark(orig, marked_lc):
    i = marked_lc.find(ACUTE)
    if i <= 0 or i - 1 > len(orig):
        return orig
    return orig[:i] + ACUTE + orig[i:]


def sentence_initial(sent, start):
    """First word of the sentence — only opening dashes/quotes/space before."""
    return sent[:start].lstrip("—–- «„\"“'\t") == ""


def classify(prev, is_perfective, sinit, medium_ok):
    """-> 'imp' | 'sg3' | None (skip). PRECISION-FIRST."""
    if prev in AMBIG_PREV:
        return None
    if prev in SUBJ_3SG or prev in PROCLITIC_3SG:
        return "sg3"
    if prev in PARTICLES_BEFORE:
        return "imp"
    if is_perfective and medium_ok and sinit:
        return "imp"
    return None


def mine_sentence(sent, hom, perf, dialogue, medium_max_words):
    toks = list(WORD_RE.finditer(sent.lower()))
    spans = [t.span() for t in toks]
    ends_bang = sent.rstrip()[-1:] == "!"
    medium_ok = (dialogue and ends_bang and len(toks) <= medium_max_words
                 and "…" not in sent and "..." not in sent)
    hits = {}          # idx -> mark string
    tags = []          # (verb, reading) for reporting
    for i, t in enumerate(toks):
        w = t.group(0)
        if w not in hom:
            continue
        prev = toks[i - 1].group(0) if i > 0 else None
        r = classify(prev, w in perf, sentence_initial(sent, t.start()),
                     medium_ok)
        if r:
            hits[i] = hom[w][r]
            tags.append((w, r))
    if not hits:
        return None
    pieces, mask, pos = [], [], 0
    for i, (s, e) in enumerate(spans):
        pieces.append(sent[pos:s])
        if i in hits:
            pieces.append(apply_mark(sent[s:e], hits[i]))
        else:
            pieces.append(sent[s:e])
            mask.append([s, e])
        pos = e
    pieces.append(sent[pos:])
    return "".join(pieces), mask, tags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/chitanka")
    ap.add_argument("--lexicon", default="config/bg/homograph_verbs.json")
    ap.add_argument("--out", default="data/chitanka/homograph_contexts.jsonl")
    ap.add_argument("--max-per-class", type=int, default=2000,
                    help="hard cap on rows per (verb, reading)")
    ap.add_argument("--balance-ratio", type=int, default=0,
                    help="cap each verb's 3sg rows at RATIO x its imperative "
                         "count (0 = off). Curbs the root-majority bias that "
                         "kept imperfective imperatives failing.")
    ap.add_argument("--balance-floor", type=int, default=300,
                    help="with --balance-ratio, keep at least this many 3sg "
                         "rows per verb so the root reading stays supervised")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--medium-max-words", type=int, default=8,
                    help="perfective clause-initial-'!' cue only on sentences "
                         "this short (longer '!' = 3sg narration)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    hom, perf = load_lexicon(args.lexicon)
    print(f"{len(hom)} homograph verbs ({len(perf)} perfective) "
          f"from {args.lexicon}")

    books = [json.loads(l) for l in
             open(os.path.join(args.corpus, "books.jsonl"), encoding="utf-8")]
    rng = random.Random(args.seed)

    per_class = defaultdict(list)   # (verb, reading) -> rows
    scanned = 0
    for book in books:
        with open(book["path"], encoding="utf-8") as f:
            for pi, par in enumerate(f.read().splitlines()):
                dialogue = par.lstrip()[:1] in "—–-«„"
                for si, sent in enumerate(split_sentences(par)):
                    if not (args.min_chars <= len(sent) <= args.max_chars):
                        continue
                    if len(WORD_RE.findall(sent)) < args.min_words:
                        continue
                    scanned += 1
                    res = mine_sentence(sent, hom, perf, dialogue,
                                        args.medium_max_words)
                    if res is None:
                        continue
                    marked, mask, tags = res
                    row = {"id": f"hom_{book['id']}_p{pi}s{si}",
                           "text": sent, "marked": marked, "mask_words": mask,
                           "n_words": len(WORD_RE.findall(sent)),
                           "n_marked": len(tags), "n_func": 0,
                           "n_ambig": 0, "n_oov": 0, "hits": tags}
                    for key in set(tags):
                        per_class[key].append(row)

    # Balance: cap each verb's abundant 3sg to a multiple of its rare
    # imperative count, so the model stops seeing a 40:1 root majority that
    # biases every prediction toward root (fine-tune #3: имперфектив
    # imperatives прави/носи/ходи stayed failed under 2000:50). The floor
    # keeps enough clean 3sg to hold the (currently solid) root reading.
    imp_count = {v: len(per_class.get((v, "imp"), [])) for v in hom}
    seen, out_rows = set(), []
    for key in sorted(per_class, key=lambda k: -len(per_class[k])):
        verb, reading = key
        rows = per_class[key]
        rng.shuffle(rows)
        cap = args.max_per_class
        if reading == "sg3" and args.balance_ratio:
            cap = min(cap, max(args.balance_ratio * imp_count[verb],
                               args.balance_floor))
        for r in rows[:cap]:
            if r["id"] not in seen:
                seen.add(r["id"])
                out_rows.append(r)
    rng.shuffle(out_rows)

    with open(args.out + ".tmp", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(args.out + ".tmp", args.out)

    final = Counter()
    for r in out_rows:
        for key in set(map(tuple, r["hits"])):
            final[key] += 1
    print(f"scanned {scanned:,} sentences, {len(out_rows):,} rows written")
    print(f"{'verb':<9} {'imp(final)':>11} {'sg3(root)':>11}")
    for w in sorted(hom):
        print(f"  {w:<9} {final.get((w,'imp'),0):>9} {final.get((w,'sg3'),0):>11}")
    print(f"-> {args.out}")

    if args.show:
        print(f"\n{args.show} random labeled rows (imp=final, sg3=root):")
        for r in rng.sample(out_rows, min(args.show, len(out_rows))):
            tag = ",".join(f"{w}:{rd}" for w, rd in r["hits"])
            print(f"  [{tag}] {r['marked']}")


if __name__ == "__main__":
    main()
