#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Pull the *suspicious* both-variant stress tokens from a harvested labels file
into a ranked vetting shortlist.

A "both-variant" word is one the aligner labeled with >1 stress placement across
the corpus (e.g. ве́че vs вече́). Two kinds hide in there:
  * TRUE homographs   — both readings are real (ве́че/вече́, отгово́ри/о́тговори,
    imperative купи́ vs aorist ку́пи). Keep both; they are the whole point.
  * ALIGNMENT NOISE   — the word has one correct stress but the aligner put the
    accent elsewhere on some tokens (или́ is always или́; и́ли:472 is likely error).

Both look identical in the counts, so a human ear (the native reviewer) has to
decide. This tool ranks the words so that time is spent where it matters most:
by **minority-token impact** — how many labels are at risk if the minority
reading is wrong — and annotates each with the minority *share* and a real
example sentence per variant (context alone often settles it).

Output: a markdown sheet + a ready-to-run `vet_stress_cli.py --words ...` line for
audio spot-checks of the top suspects.

    ./venv/bin/python build_vetting_shortlist.py \
        --labels data/kaldi/stress_labels_incr_20260728.jsonl \
        --out vetting_shortlist_incr_20260728.md
"""
import argparse
import json
import unicodedata
from collections import defaultdict

ACUTE = "́"


def stressed_syllable(marked):
    """Return the 0-based index (among vowels) carrying the acute, or -1."""
    vowels = "аъоуеияюѝАЪОУЕИЯЮ"
    vi = -1
    idx = -1
    for ch in unicodedata.normalize("NFC", marked):
        # NFC keeps combining accent separate for Cyrillic (no precomposed form)
        pass
    # walk the raw string, counting vowels, noting which precedes the accent
    vcount = -1
    for ch in marked:
        if ch in vowels:
            vcount += 1
        elif ch == ACUTE:
            idx = vcount
    return idx


def load(labels_path):
    by_wm = defaultdict(list)       # (word, marked) -> [rows]
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("audio_path"):
                by_wm[(r["word"], r["marked"])].append(r)
    return by_wm


def example(rows):
    """Pick a clear example sentence (prefer one without <unk>, medium length)."""
    cand = sorted(
        rows,
        key=lambda r: (r["sentence"].count("<unk>"), abs(len(r["sentence"].split()) - 10)),
    )
    return cand[0] if cand else None


def bold_word(sentence, word):
    return " ".join(f"**{w}**" if w == word else w for w in sentence.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/kaldi/stress_labels_incr_20260728.jsonl")
    ap.add_argument("--out", default="vetting_shortlist_incr_20260728.md")
    ap.add_argument("--min-minority", type=int, default=20,
                    help="skip words whose 2nd-most-common variant has fewer "
                         "tokens than this (below = rare noise, low impact)")
    ap.add_argument("--top", type=int, default=40, help="rows in the sheet")
    ap.add_argument("--vet-top", type=int, default=20,
                    help="how many words to put in the printed vet_stress_cli command")
    args = ap.parse_args()

    by_wm = load(args.labels)

    # Aggregate per word.
    by_word = defaultdict(dict)     # word -> {marked: count}
    for (w, m), rows in by_wm.items():
        by_word[w][m] = len(rows)

    rows_out = []
    total_minority_tokens = 0
    both_variant_words = 0
    for w, variants in by_word.items():
        if len(variants) < 2:
            continue
        both_variant_words += 1
        ordered = sorted(variants.items(), key=lambda kv: -kv[1])
        (m0, c0), (m1, c1) = ordered[0], ordered[1]
        minority = sum(c for _, c in ordered[1:])         # everything but dominant
        total = sum(variants.values())
        total_minority_tokens += minority
        share = minority / total
        if c1 < args.min_minority:
            continue
        rows_out.append({
            "word": w,
            "ordered": ordered,
            "dominant": (m0, c0),
            "runnerup": (m1, c1),
            "minority": minority,
            "total": total,
            "share": share,
            "n_variants": len(variants),
            "pos0": stressed_syllable(m0),
            "pos1": stressed_syllable(m1),
        })

    # Rank by impact (minority tokens at risk) — that's the label-noise surface.
    rows_out.sort(key=lambda d: -d["minority"])
    shortlist = rows_out[: args.top]

    # A secondary lens: near-balanced splits (genuine ambiguity / real homographs).
    balanced = sorted((d for d in rows_out if 0.35 <= d["share"] <= 0.65),
                      key=lambda d: -d["total"])[:20]

    with open(args.out, "w", encoding="utf-8") as out:
        w = out.write
        w(f"# Stress vetting shortlist — `{args.labels}`\n\n")
        w(f"- Both-variant word types: **{both_variant_words}**\n")
        w(f"- Total minority (at-risk) tokens: **{total_minority_tokens:,}**\n")
        w(f"- Words with ≥{args.min_minority} minority tokens (shown): "
          f"**{len(rows_out)}** (top {len(shortlist)} below)\n\n")
        w("`share` = minority ÷ total. High share + common function word ⇒ likely "
          "*alignment noise* (drop the minority). Balanced share on a content word ⇒ "
          "likely a *true homograph* (keep both). Example sentences are the fastest "
          "tell; play audio only when context is ambiguous.\n\n")

        w("## Ranked by minority-token impact (most labels at risk first)\n\n")
        w("| # | word | variants (count) | minority | share | example (minority variant) |\n")
        w("|--:|------|------------------|---------:|------:|----------------------------|\n")
        for i, d in enumerate(shortlist, 1):
            vv = "  ".join(f"{m}:{c}" for m, c in d["ordered"])
            ex = example(by_wm[(d["word"], d["runnerup"][0])])
            exs = bold_word(ex["sentence"], d["word"]) if ex else "—"
            if len(exs) > 90:
                exs = exs[:88] + "…"
            w(f"| {i} | {d['word']} | {vv} | {d['minority']} | {d['share']:.0%} | {exs} |\n")

        w("\n## Near-balanced splits (35–65% — genuine ambiguity / true homographs)\n\n")
        w("| word | variants (count) | share | example A | example B |\n")
        w("|------|------------------|------:|-----------|-----------|\n")
        for d in balanced:
            (mA, cA), (mB, cB) = d["ordered"][0], d["ordered"][1]
            exA = example(by_wm[(d["word"], mA)])
            exB = example(by_wm[(d["word"], mB)])
            sA = bold_word(exA["sentence"], d["word"])[:60] if exA else "—"
            sB = bold_word(exB["sentence"], d["word"])[:60] if exB else "—"
            w(f"| {d['word']} | {mA}:{cA}  {mB}:{cB} | {d['share']:.0%} | {sA} | {sB} |\n")

    vet_words = ",".join(d["word"] for d in shortlist[: args.vet_top])
    print(f"wrote {args.out}")
    print(f"  both-variant word types : {both_variant_words}")
    print(f"  at-risk minority tokens : {total_minority_tokens:,}")
    print(f"  shortlisted (≥{args.min_minority} minority): {len(rows_out)}  "
          f"(sheet shows top {len(shortlist)})")
    print()
    print("Audio spot-check the top suspects with:")
    print(f"  ./venv/bin/python vet_stress_cli.py --labels {args.labels} \\")
    print(f"      --words \"{vet_words}\"")


if __name__ == "__main__":
    main()
