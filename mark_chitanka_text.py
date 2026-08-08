# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Text-only Accentor training rows from the cleaned chitanka corpus.

Replaces the synthetic-audio `ambig` labels after the BgTTS oracle failed the
stability test (2026-07-04: homograph stress errors are systematic, e.g. че́та
always synthesized as чета́ — frequency prior, not context). Label sources:

  - UNAMBIGUOUS lexicon words (exactly one stress candidate): marked with
    U+0301 — no audio needed, the dictionary IS the ground truth. This is the
    sentence-scale analog of the paper's §3.4 dictionary extension.
  - FUNCTION WORDS (config/bg/function_words_unstressed.json + the bare-clitic
    entries of config/bg/lexicon_supplement.json): labeled ALL-UNSTRESSED —
    a real zero label, not a mask (they are ~30% of running tokens; masking
    them would starve the loss). Marked supplement entries (бе́ше) label like
    unambiguous lexicon words.
  - AMBIGUOUS words (>=2 candidates) and OOV words: left unmarked and listed
    in `mask_words` — the Accentor trainer must exclude their vowels from the
    per-vowel BCE loss. Their contextual labels come from HUMAN speech only
    (parliament crawl + the real audiobooks in data/bg, data/bg400).

Output row (marked.jsonl):
  {"id", "book", "text",            # original unmarked sentence (casing kept)
   "marked",                        # text with U+0301 on resolved words
   "mask_words": [[start,end], …],  # char spans in "text" to EXCLUDE from loss
   "n_words", "n_marked", "n_ambig", "n_oov"}

Usage:
    ./venv/bin/python mark_chitanka_text.py \
        --corpus data/chitanka --sqlite data/bg_lexicon/bg_stress_candidates.sqlite \
        --out data/chitanka/marked.jsonl --max-tokens 80000000
"""
import argparse
import json
import os
import random
import re
import sqlite3
from collections import Counter

from tqdm import tqdm

from bg_chitanka_clean import split_sentences
from select_chitanka_sentences import WORD_RE, sentence_ok

VOWELS = set("аъоуеиюяѝ")


def load_lexicon(path):
    """candidates DB -> {surface: mark} for unambiguous forms, set(ambiguous)."""
    db = sqlite3.connect(path)
    uniq, ambig = {}, set()
    for name, mark in db.execute("SELECT name, mark FROM candidates"):
        if name in ambig:
            continue
        prev = uniq.get(name)
        if prev is None:
            uniq[name] = mark
        elif prev != mark:
            del uniq[name]
            ambig.add(name)
    return uniq, ambig


def load_function_words(func_path, supplement_path, uniq, ambig):
    """Zero-label (all-unstressed) closed class; marked supplement entries
    (бе́ше) join `uniq` instead. Ambiguity in the main lexicon wins."""
    func = set()
    if func_path and os.path.exists(func_path):
        with open(func_path, encoding="utf-8") as f:
            func.update(w for w in json.load(f)["words"] if w not in ambig)
    if supplement_path and os.path.exists(supplement_path):
        with open(supplement_path, encoding="utf-8") as f:
            for surface, readings in json.load(f)["forms"].items():
                if len(readings) != 1 or surface in ambig:
                    continue
                if "́" in readings[0]:
                    uniq.setdefault(surface, readings[0])
                else:
                    func.add(surface)
    # The curated function list outranks the lexicon: rechko carries marked
    # entries for some clitics (и -> и́, му -> му́) which would otherwise label
    # every running token stressed (caught by eval_accentor confidences).
    for w in func:
        uniq.pop(w, None)
    return func


def mark_sentence(sent, uniq, ambig, func=frozenset()):
    """Apply unambiguous marks; return (marked, mask_spans, counters)."""
    mask = []
    n_marked = n_func = n_ambig = n_oov = n_words = 0
    pos = 0
    pieces = []
    for m in WORD_RE.finditer(sent.lower()):
        n_words += 1
        s, e = m.span()
        word = m.group(0)
        orig = sent[s:e]
        pieces.append(sent[pos:s])
        if word in ambig:
            n_ambig += 1
            mask.append([s, e])
            pieces.append(orig)
        elif word in uniq:
            n_marked += 1
            pieces.append(apply_mark(orig, uniq[word]))
        elif word in func:
            # Explicit all-unstressed label: unmarked AND unmasked.
            n_func += 1
            pieces.append(orig)
        else:
            # OOV — no label. Single-vowel words could arguably default to
            # "stressed" but clitics break that; stay conservative and mask.
            n_oov += 1
            mask.append([s, e])
            pieces.append(orig)
        pos = e
    pieces.append(sent[pos:])
    return "".join(pieces), mask, (n_words, n_marked, n_func, n_ambig, n_oov)


def apply_mark(orig, marked_lc):
    """Transfer the U+0301 position from the lowercase lexicon form onto the
    original-cased word (casing is an Accentor input feature — keep it)."""
    i = marked_lc.find("́")
    if i <= 0 or i - 1 > len(orig):
        return orig
    return orig[: i] + "́" + orig[i:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/chitanka")
    ap.add_argument("--sqlite", default="data/bg_lexicon/bg_stress_candidates.sqlite")
    ap.add_argument("--function-words", default="config/bg/function_words_unstressed.json")
    ap.add_argument("--supplement", default="config/bg/lexicon_supplement.json")
    ap.add_argument("--out", default="data/chitanka/marked.jsonl")
    ap.add_argument("--max-tokens", type=int, default=80_000_000,
                    help="stop after this many word tokens (paper: 70M)")
    ap.add_argument("--max-book-tokens", type=int, default=50_000,
                    help="per-book token cap (0 = unlimited) — corpus diversity")
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--max-chars", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("loading lexicon ...")
    uniq, ambig = load_lexicon(args.sqlite)
    func = load_function_words(args.function_words, args.supplement, uniq, ambig)
    print(f"  {len(uniq):,} unambiguous forms, {len(ambig):,} ambiguous, "
          f"{len(func)} zero-label function words")

    books = []
    with open(os.path.join(args.corpus, "books.jsonl"), encoding="utf-8") as f:
        for line in f:
            books.append(json.loads(line))
    random.Random(args.seed).shuffle(books)

    tot = Counter()
    out_tmp = args.out + ".tmp"
    with open(out_tmp, "w", encoding="utf-8") as of:
        pbar = tqdm(books, desc="mark")
        for book in pbar:
            if tot["words"] >= args.max_tokens:
                break
            book_words = 0
            with open(book["path"], encoding="utf-8") as f:
                pars = f.read().splitlines()
            for pi, par in enumerate(pars):
                if tot["words"] >= args.max_tokens:
                    break
                if args.max_book_tokens and book_words >= args.max_book_tokens:
                    break
                for si, sent in enumerate(split_sentences(par)):
                    if not sentence_ok(sent, args.min_chars, args.max_chars):
                        continue
                    marked, mask, (nw, nm, nf, na, no) = mark_sentence(
                        sent, uniq, ambig, func)
                    # Gate on the masked fraction (what the loss can't use) and
                    # require real stress supervision, not only zero-labels.
                    if nw == 0 or (na + no) / nw > 0.30 or nm / nw < 0.35:
                        continue
                    of.write(json.dumps({
                        "id": f"{book['id']}_p{pi}s{si}", "book": book["id"],
                        "text": sent, "marked": marked, "mask_words": mask,
                        "n_words": nw, "n_marked": nm, "n_func": nf,
                        "n_ambig": na, "n_oov": no,
                    }, ensure_ascii=False) + "\n")
                    tot.update(words=nw, marked=nm, func=nf, ambig=na, oov=no,
                               sents=1)
                    book_words += nw
            pbar.set_postfix(
                Mtok=f"{tot['words']/1e6:.1f}",
                lbl=f"{(tot['marked']+tot['func'])/max(tot['words'],1):.0%}")
    os.replace(out_tmp, args.out)

    w = max(tot["words"], 1)
    print(f"sentences={tot['sents']:,}  tokens={tot['words']/1e6:.1f}M")
    print(f"labeled: {(tot['marked']+tot['func'])/w:.1%} "
          f"(stress-marked {tot['marked']/w:.1%} + zero-label function "
          f"{tot['func']/w:.1%})   masked-ambiguous: {tot['ambig']/w:.1%}   "
          f"masked-OOV: {tot['oov']/w:.1%}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
