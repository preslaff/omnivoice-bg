#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""BgGPT sense-teacher: text-derived homograph training rows (plan §P2.5).

Two-step design validated 2026-07-15 (bggpt_validate2.py, 12/12): BgGPT
decides the SENSE of a homograph in context — a text-native task — and
config/bg/sense_inventory.json maps the sense to the U+0301 mark. The
model never sees stress notation (L17a: LLMs know the sense, not the
sound). This is how chitanka's unlimited TEXT substitutes for the
reference paper's 11,000 h of audio: contexts from text, ground truth
anchored by the Kaldi audio harvest.

Provenance discipline: rows carry source="bggpt-teacher" and live in
their own manifest so they can be spot-checked, ablated, or withdrawn
wholesale without touching the audio-derived labels. Before trusting a
full run, native-review a sample:
    shuf -n 50 data/bggpt_teacher/sense_rows.jsonl

Resume-safe: already-answered (word, sentence) pairs are skipped, output
is appended row-by-row — a power surge costs nothing.

Usage (defaults are sane; ~13 words x 600 sentences ~= 4-6 h on the
GPU-hosted 27B, runs fine alongside CPU-bound Kaldi stages):
    ./venv/bin/python bggpt_teacher.py                # full run
    ./venv/bin/python bggpt_teacher.py --dry-run      # mine + count only
    ./venv/bin/python bggpt_teacher.py --per-word 100 --limit 500  # pilot
"""
import argparse
import json
import os
import pathlib
import random
import urllib.request

from mine_homographs import WORD_RE, split_sentences

SYSTEM = (
    "Ти си експерт по българска граматика и лексикология. Дадено е "
    "изречение и дума от него с няколко възможни тълкувания. Определи кое "
    "тълкуване е вярно В ТОВА изречение. Отговори САМО с буквата на "
    "правилното тълкуване."
)


def ask(url, sentence, word, senses, timeout=300):
    opts = "\n".join(f"{s['id']}) {s['gloss']}" for s in senses)
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Изречение: {sentence}\nДума: „{word}“\n{opts}"},
        ],
        "temperature": 0, "max_tokens": 6,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"].strip().lower()


def ask_checked(url, sentence, word, senses, double_check):
    """Resolve to a sense id, or None. With double_check, the options are
    presented a second time in reverse order under swapped letters; only
    agreement on the SAME underlying sense survives (cancels position bias
    and coin-flip answers at 2x query cost)."""
    ids = [s["id"] for s in senses]
    a1 = ask(url, sentence, word, senses)[:1]
    if a1 not in ids:
        return None
    if not double_check:
        return a1
    relabeled = [dict(s, id=ids[i]) for i, s in enumerate(reversed(senses))]
    a2 = ask(url, sentence, word, relabeled)[:1]
    if a2 not in ids:
        return None
    # map the second answer's letter back to the underlying sense id
    a2_true = list(reversed(senses))[ids.index(a2)]["id"]
    return a1 if a1 == a2_true else None


def mine(corpus, words, min_chars, max_chars, min_words, oversample, rng):
    """word -> list of candidate sentences containing it (deduped)."""
    books = [json.loads(l) for l in
             open(os.path.join(corpus, "books.jsonl"), encoding="utf-8")]
    rng.shuffle(books)
    cands = {w: [] for w in words}
    seen = set()
    need = {w: oversample for w in words}
    for book in books:
        if not any(need.values()):
            break
        try:
            text = open(book["path"], encoding="utf-8").read()
        except OSError:
            continue
        for par in text.splitlines():
            for sent in split_sentences(par):
                if not (min_chars <= len(sent) <= max_chars):
                    continue
                toks = WORD_RE.findall(sent.lower())
                if len(toks) < min_words:
                    continue
                # Proper-name guard: WORD_RE is lowercase-only, so a raw-text
                # match means the word appears uncapitalized. Mid-sentence
                # capitalized-only occurrences (Моли the name) are skipped;
                # sentence-initial capitals (imperatives!) are kept.
                raw_toks = set(WORD_RE.findall(sent))
                first = sent.lstrip('—–-«„"\' ').split(' ', 1)[0]
                first = first.strip('.,!?…:;').lower()
                for w in words:
                    if need[w] and w in toks and (w in raw_toks or w == first):
                        key = (w, sent)
                        if key not in seen:
                            seen.add(key)
                            cands[w].append(sent)
                            need[w] -= 1
    return cands


def mine_from_file(path, words, min_chars, max_chars, min_words, per_word, rng):
    """Targeted source: read pre-mined sentences from a JSONL (`text` field,
    e.g. data/chitanka/imperatives.jsonl) instead of scanning books. Same
    proper-name guard. Use to top up senses that random prose starves
    (imperatives, rare nouns) — mine_homographs finds candidates, BgGPT
    confirms the sense here, so only agreed rows survive."""
    cands = {w: [] for w in words}
    seen = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        sent = json.loads(line).get("text", "").strip()
        if not (min_chars <= len(sent) <= max_chars):
            continue
        toks = WORD_RE.findall(sent.lower())
        if len(toks) < min_words:
            continue
        raw_toks = set(WORD_RE.findall(sent))
        first = sent.lstrip('—–-«„"\' ').split(' ', 1)[0].strip('.,!?…:;').lower()
        for w in words:
            if w in toks and (w in raw_toks or w == first):
                key = (w, sent)
                if key not in seen:
                    seen.add(key)
                    cands[w].append(sent)
    for w in cands:
        rng.shuffle(cands[w])
        cands[w] = cands[w][:per_word]
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/chitanka")
    ap.add_argument("--sentences", default=None,
                    help="targeted round: read pre-mined sentences from this "
                         "JSONL (`text` field, e.g. "
                         "data/chitanka/imperatives.jsonl) instead of scanning "
                         "the book corpus. Resumes into --out, so it tops up "
                         "starved senses without redoing the random sweep.")
    ap.add_argument("--inventory", default="config/bg/sense_inventory.json")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--out", default="data/bggpt_teacher/sense_rows.jsonl")
    ap.add_argument("--per-word", type=int, default=600,
                    help="sentences to label per homograph")
    ap.add_argument("--limit", type=int, default=0,
                    help="total query cap this run (0 = no cap)")
    ap.add_argument("--include-review", action="store_true",
                    help="also teach words marked needs_native_review "
                         "(против, коса) — default skips them")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="mine and report candidate counts, no queries")
    ap.add_argument("--double-check", action="store_true",
                    help="ask twice with reversed option order; keep only "
                         "agreements (2x cost, kills position bias — "
                         "recommended for the full run)")
    args = ap.parse_args()

    inv = json.load(open(args.inventory, encoding="utf-8"))["inventory"]
    words = {w: e for w, e in inv.items()
             if args.include_review or e["status"] == "validated"}
    skipped_words = sorted(set(inv) - set(words))
    print(f"{len(words)} homographs to teach"
          + (f" (skipping review-flagged: {', '.join(skipped_words)})"
             if skipped_words else ""))

    rng = random.Random(args.seed)
    if args.sentences:
        print(f"targeted round: mining from {args.sentences}")
        cands = mine_from_file(args.sentences, list(words), args.min_chars,
                               args.max_chars, args.min_words,
                               args.per_word, rng)
    else:
        cands = mine(args.corpus, list(words), args.min_chars, args.max_chars,
                     args.min_words, oversample=args.per_word * 3, rng=rng)
        for w in cands:
            rng.shuffle(cands[w])
            cands[w] = cands[w][:args.per_word]
    print("mined candidates: " +
          " ".join(f"{w}:{len(c)}" for w, c in sorted(cands.items())))
    if args.dry_run:
        return

    # Resume: skip pairs already answered.
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["word"], r["sentence"]))
    if done:
        print(f"resume: {len(done)} rows already answered")

    queue = [(w, s) for w in sorted(cands) for s in cands[w]
             if (w, s) not in done]
    rng.shuffle(queue)  # interleave words so partial runs stay balanced
    if args.limit:
        queue = queue[:args.limit]
    print(f"{len(queue)} queries to run")

    n_ok = n_bad = 0
    from collections import Counter
    hist = Counter()
    with open(out, "a", encoding="utf-8") as f, \
         open(str(out) + ".rejects", "a", encoding="utf-8") as rej:
        for i, (w, sent) in enumerate(queue):
            senses = words[w]["senses"]
            by_id = {s["id"]: s for s in senses}
            try:
                ans = ask_checked(args.url, sent, w, senses,
                                  args.double_check)
            except Exception as e:  # noqa: BLE001 — keep the run alive
                rej.write(json.dumps({"word": w, "sentence": sent,
                                      "error": str(e)},
                                     ensure_ascii=False) + "\n")
                n_bad += 1
                continue
            if ans is None:
                rej.write(json.dumps({"word": w, "sentence": sent,
                                      "answer": "disagreement/invalid"},
                                     ensure_ascii=False) + "\n")
                n_bad += 1
                continue
            s = by_id[ans]
            f.write(json.dumps({
                "word": w, "sense": s["id"], "marked": s["marked"],
                "sentence": sent, "source": "bggpt-teacher",
            }, ensure_ascii=False) + "\n")
            f.flush()
            hist[(w, s["marked"])] += 1
            n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(queue)}  ok={n_ok} bad={n_bad}",
                      flush=True)

    print(f"\ndone: {n_ok} rows, {n_bad} rejects -> {out}")
    print("per-sense counts (minority senses are the point — check them):")
    for (w, m), c in sorted(hist.items()):
        print(f"  {m}: {c}")
    print(f"\nnative spot-check before use:  shuf -n 50 {out}")


if __name__ == "__main__":
    main()
