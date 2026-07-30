#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Fetch genuinely-UNSEEN Bulgarian text for out-of-register Accentor evaluation.

The Accentor trains on chitanka (fiction) + parliament (harvest). Both eval
scripts historically sample parliament, which is *in* the training register, so
they don't test generalization to the text a TTS front-end actually meets (ads,
narration, encyclopedic, news). This pulls plain-text Bulgarian Wikipedia
articles across diverse topics — a register the model provably never trained on
— into data/eval_wild/, which the automatic SER then scores WITHOUT labels:

    ./venv/bin/python fetch_eval_wild.py                 # -> data/eval_wild/wiki.txt
    ./venv/bin/python eval_accentor_ood.py --ckpt <ckpt> --transcripts data/eval_wild --n 4000

On 2026-07-30 this gave ft7c SER 0.089% vs ft7b 0.119% on ~33k unseen words —
confirming ft7c generalizes better on an unseen register, not just the battery.

Wikipedia rate-limits (HTTP 429); the delay between requests keeps it happy.
Add topics for more coverage.
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request

TOPICS = [
    "Изкуствен интелект", "Квантова механика", "Рила", "Черно море",
    "Христо Ботев", "Българска кухня", "Слънчева система", "ДНК",
    "Футбол", "Фотосинтеза", "Компютър", "Космос", "Витоша", "Дунав",
    "Иван Вазов", "Възраждане (България)", "Електричество", "Вулкан",
]


def fetch(title):
    url = "https://bg.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "omnivoice-eval/1.0"})
    d = json.load(urllib.request.urlopen(req, timeout=20))
    return next(iter(d["query"]["pages"].values())).get("extract", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/eval_wild/wiki.txt")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="seconds between requests (avoid HTTP 429)")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=220)
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sents, seen = [], set()
    for t in TOPICS:
        try:
            txt = re.sub(r"==+[^=]+==+", "", fetch(t))   # strip section headers
            for para in txt.split("\n"):
                for s in re.split(r"(?<=[.!?])\s+", para.strip()):
                    s = s.strip()
                    if (args.min_chars <= len(s) <= args.max_chars
                            and len(s.split()) >= 5 and not s.startswith("=")
                            and s not in seen):
                        seen.add(s)
                        sents.append(s)
            print(f"  {t}: ok")
        except Exception as e:
            print(f"  {t}: FAIL {e}")
        time.sleep(args.delay)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sents) + "\n")
    words = sum(len(s.split()) for s in sents)
    print(f"\nwrote {len(sents)} sentences, ~{words:,} words -> {args.out}")


if __name__ == "__main__":
    main()
