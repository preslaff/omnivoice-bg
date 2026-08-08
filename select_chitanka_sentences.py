# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Phase C2 — coverage-greedy sentence selection from the cleaned chitanka corpus.

The full archive is ~600M words (~75,000 h if narrated) — far beyond any
synthesis budget — so we select a lexically-targeted subset for the BgTTS
synthetic-audiobook branch:

  tier "ambig"    sentences giving ambiguous (multi-candidate) lexicon forms
                  contextual coverage, capped at --per-ambig contexts per form;
  tier "newtype"  sentences contributing word types not yet selected
                  (greedy type coverage — the paper's audiobooks contributed
                  254K types; this tier replicates that lexical diversity);
  tier "fill"     optional random fill to the --hours budget.

Quality gate (exact text<->audio correspondence is the whole point — anything
the TTS would verbalize breaks the label):
  - 25..250 chars (BgTTS chunks at 250 — one sentence must stay one chunk),
  - >= 4 words, ends with terminal punctuation, not ALL-CAPS,
  - Bulgarian Cyrillic + punctuation only: no digits, no Latin.

Usage:
    ./venv/bin/python select_chitanka_sentences.py \
        --corpus data/chitanka --sqlite data/bg_lexicon/bg_stress_candidates.sqlite \
        --out data/chitanka/selected.jsonl --hours 100
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

CHARS_PER_SEC = 15.0  # rough narrated-Bulgarian rate for budgeting

WORD_RE = re.compile(r"[а-яьѝ]+(?:-[а-яьѝ]+)*")
ALLOWED_RE = re.compile(r'^[а-яьѝА-ЯЍ\s,;:.!?…«»„“”"\'()\—–-]+$')
TERMINAL_RE = re.compile(r"[.!?…]$")


def sentence_ok(s: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(s) <= max_chars):
        return False
    if not TERMINAL_RE.search(s):
        return False
    if not ALLOWED_RE.match(s):
        return False
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return False  # ALL-CAPS headings
    if len(WORD_RE.findall(s.lower())) < 4:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/chitanka")
    ap.add_argument("--sqlite", default="data/bg_lexicon/bg_stress_candidates.sqlite")
    ap.add_argument("--out", default="data/chitanka/selected.jsonl")
    ap.add_argument("--hours", type=float, default=100.0, help="synthesis budget")
    ap.add_argument("--per-ambig", type=int, default=25,
                    help="max selected contexts per ambiguous form")
    ap.add_argument("--min-new-types", type=int, default=1,
                    help="unseen types needed for a 'newtype' selection")
    ap.add_argument("--max-book-sec", type=float, default=0.0,
                    help="max estimated audio per book (0 = unlimited); forces "
                         "context diversity across books instead of swallowing "
                         "the first few novels whole")
    ap.add_argument("--fill", action="store_true",
                    help="random-fill remaining budget after coverage saturates")
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--max-chars", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    ambig = {r[0] for r in db.execute(
        "SELECT name FROM candidates GROUP BY name HAVING count(DISTINCT mark) > 1")}
    print(f"ambiguous forms: {len(ambig):,}")

    books = []
    with open(os.path.join(args.corpus, "books.jsonl"), encoding="utf-8") as f:
        for line in f:
            books.append(json.loads(line))
    random.Random(args.seed).shuffle(books)  # genre diversity before budget cutoff

    budget_sec = args.hours * 3600
    total_sec = 0.0
    seen_types = set()
    ambig_counts = Counter()
    reasons = Counter()
    n_seen = n_pass = 0

    out_tmp = args.out + ".tmp"
    with open(out_tmp, "w", encoding="utf-8") as of:
        pbar = tqdm(books, desc="select")
        for book in pbar:
            if total_sec >= budget_sec:
                break
            book_sec = 0.0
            with open(book["path"], encoding="utf-8") as f:
                pars = f.read().splitlines()
            for pi, par in enumerate(pars):
                if total_sec >= budget_sec:
                    break
                if args.max_book_sec and book_sec >= args.max_book_sec:
                    break
                for si, sent in enumerate(split_sentences(par)):
                    n_seen += 1
                    if not sentence_ok(sent, args.min_chars, args.max_chars):
                        continue
                    n_pass += 1
                    words = set(WORD_RE.findall(sent.lower()))
                    hot = [w for w in words & ambig
                           if ambig_counts[w] < args.per_ambig]
                    new_types = words - seen_types
                    if hot:
                        reason = "ambig"
                    elif len(new_types) >= args.min_new_types:
                        reason = "newtype"
                    elif args.fill:
                        reason = "fill"
                    else:
                        continue
                    est_sec = len(sent) / CHARS_PER_SEC
                    total_sec += est_sec
                    seen_types |= words
                    for w in words & ambig:
                        ambig_counts[w] += 1
                    reasons[reason] += 1
                    of.write(json.dumps({
                        "id": f"{book['id']}_p{pi}s{si}",
                        "book": book["id"],
                        "text": sent,
                        "n_chars": len(sent),
                        "est_sec": round(est_sec, 2),
                        "reason": reason,
                    }, ensure_ascii=False) + "\n")
                    if total_sec >= budget_sec:
                        break
            pbar.set_postfix(h=f"{total_sec/3600:.1f}", types=len(seen_types))
    os.replace(out_tmp, args.out)

    n_sel = sum(reasons.values())
    cov1 = sum(1 for v in ambig_counts.values() if v >= 1)
    covk = sum(1 for v in ambig_counts.values() if v >= args.per_ambig)
    print(f"selected {n_sel:,} / passed {n_pass:,} / scanned {n_seen:,} sentences")
    print(f"est audio: {total_sec/3600:.1f} h   reasons: {dict(reasons)}")
    print(f"type coverage: {len(seen_types):,} distinct types")
    print(f"ambig coverage: {cov1:,}/{len(ambig):,} forms with >=1 context, "
          f"{covk:,} saturated at {args.per_ambig}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
