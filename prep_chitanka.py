# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Clean the chitanka book archive into per-book prose + a corpus catalog.

Phase C1 of the synthetic-audiobook branch (see docs/BG_PARLIAMENT_INGESTION.md,
"Missing dependencies"): chitanka txt exports -> cleaned prose the sentence
selector (select_chitanka_sentences.py) and the BgTTS synthesis pipeline
consume. Mirrors prep_bg400.py conventions (argparse, tqdm, absolute paths,
JSONL catalogs).

Input   data/chitanka_archive/{romans_txt,novels_txt}/<id>.txt
Output  <out>/clean/<src>_<id>.txt    one paragraph per line
        <out>/books.jsonl             {id, src, path, title, ...stats}
        <out>/skipped.jsonl           books that failed structural cleaning

Usage:
    ./venv/bin/python prep_chitanka.py --out data/chitanka           # full run
    ./venv/bin/python prep_chitanka.py --out data/chitanka --limit 300 --seed 42
"""
import argparse
import json
import os
import random

from tqdm import tqdm

from bg_chitanka_clean import clean_book, split_sentences

SRC_DIRS = {"romans": "romans_txt", "novels": "novels_txt"}

# Guardrails against a mis-cleaned book poisoning the corpus.
MIN_PARAGRAPHS = 20
MIN_CYRILLIC_RATIO = 0.60


def cyrillic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum("Ѐ" <= c <= "ӿ" for c in letters) / len(letters)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default="data/chitanka_archive")
    ap.add_argument("--out", default="data/chitanka")
    ap.add_argument("--limit", type=int, default=0, help="process only N books (pilot)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed for --limit")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    clean_dir = os.path.join(out_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)

    books = []
    for src, sub in SRC_DIRS.items():
        d = os.path.join(os.path.abspath(args.archive), sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".txt"):
                books.append((src, os.path.join(d, fn)))
    if args.limit:
        random.Random(args.seed).shuffle(books)
        books = books[: args.limit]

    books_path = os.path.join(out_dir, "books.jsonl")
    skipped_path = os.path.join(out_dir, "skipped.jsonl")
    n_ok = n_skip = 0
    tot_chars = tot_sents = 0

    with open(books_path, "w", encoding="utf-8") as bf, open(
        skipped_path, "w", encoding="utf-8"
    ) as sf:
        for src, path in tqdm(books, desc="clean"):
            book_id = f"{src}_{os.path.splitext(os.path.basename(path))[0]}"
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    res = clean_book(f.read())
            except Exception as e:  # noqa: BLE001 - log-and-continue crawl style
                res, err = None, f"exception: {e}"
            else:
                err = None
                if res is None:
                    err = "no section markers (nonstandard export)"
                elif res.stats["n_paragraphs"] < MIN_PARAGRAPHS:
                    err = f"too few paragraphs ({res.stats['n_paragraphs']})"
                else:
                    sample = " ".join(res.paragraphs[:50])
                    cr = cyrillic_ratio(sample)
                    if cr < MIN_CYRILLIC_RATIO:
                        err = f"low cyrillic ratio ({cr:.2f})"
            if err:
                n_skip += 1
                sf.write(json.dumps({"id": book_id, "path": path, "error": err},
                                    ensure_ascii=False) + "\n")
                continue

            clean_path = os.path.join(clean_dir, f"{book_id}.txt")
            tmp = clean_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as cf:
                cf.write("\n".join(res.paragraphs) + "\n")
            os.replace(tmp, clean_path)

            n_sents = sum(len(split_sentences(p)) for p in res.paragraphs)
            row = {"id": book_id, "src": src, "path": clean_path,
                   "title": res.title, "n_sents": n_sents, **res.stats}
            bf.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
            tot_chars += res.stats["n_chars"]
            tot_sents += n_sents

    print(f"ok={n_ok} skipped={n_skip}  ->  {books_path}")
    print(f"total: {tot_chars/1e6:.1f}M chars, {tot_sents/1e3:.0f}K sentences")
    # ~15 chars/sec of narrated Bulgarian -> rough audio-hours equivalent.
    print(f"rough full-narration equivalent: ~{tot_chars/15/3600:,.0f} h")


if __name__ == "__main__":
    main()
