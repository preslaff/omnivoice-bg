# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Accentor fine-tune rows from harvested speech stress labels, under the
three-tier policy calibrated by native vetting (2026-07-05, see
docs/BG_PIPELINE_GUIDE.md stage 5b):

  TIER 3 (contextual gold — curated homograph classes, vetting confirmed both
         readings are real): keep the aligner's variant as the label.
  TIER 2 (prosodically weak words where the GMM's final-stress bias can even
         invert corpus majorities — може́ 1125:517 yet 0/5 vetted): NO stress
         supervision, mask.
  TIER 1 (everything else): trust the token only if it matches the canonical
         rechko stress AND the word's corpus-wide canonical share >= --min-share
         (default 0.90); other tokens masked.

Each utterance becomes one training row in the marked.jsonl format
(mark_chitanka_text.py): non-harvest words are labeled from the lexicon
exactly like the chitanka pretrain (unambiguous marked / function zero-label /
ambiguous+OOV masked); harvest tokens overlay per the tiers. Only utterances
with >= 1 surviving speech label are kept — the rest add nothing over
chitanka. Rows are lowercase (Kaldi text); the caps feature sees only the
chitanka mix rows.

A random sample of the (re-marked) chitanka corpus is mixed in at
--mix-ratio rows per speech row to prevent forgetting the general lexicon.

Usage:
    ./venv/bin/python build_accentor_finetune.py \
        --labels data/kaldi/stress_labels.jsonl \
        --labels data/kaldi/stress_labels_parliament2.jsonl \
        --mix-chitanka data/chitanka/marked.jsonl --mix-ratio 5 \
        --out data/accentor_finetune.jsonl
"""
import argparse
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict

from mark_chitanka_text import apply_mark, load_function_words, load_lexicon
from select_chitanka_sentences import WORD_RE

# Vetting 2026-07-05: GMM final-stress bias class — corpus counts untrustworthy.
# Milestone-A re-vetting (2026-07-16, TDNN labels): the split resolved by
# WORD CLASS, not by aligner majority —
#   * PROMOTED (removed): може→мо́же, всички→вси́чки were content words with
#     real lexical stress; the TDNN + veto now give them clean single-stress
#     (мо́же/вси́чки 100% vetted). други→дру́ги likewise clean (never listed;
#     its harvest majority други́ 59% was inverted noise — F3 — the veto drops it).
#   * KEPT/ADDED: като is a proclitic function word whose "stress" is a phrasal
#     artifact (като́ 100%/ка́то 14% is prominence leaking to the following
#     stressed word, NOT lexical) — same trap as и́ли/ме́жду; suppress.
TIER2_EXCLUDE = {"съюз", "вече", "още", "един", "една",
                 "или", "колеги", "между", "като"}
# Battery-confirmed + vetting-confirmed contextual homographs (both readings
# real). Grows as more classes are vetted. The -и verbs are the
# imperative(final-stress) vs 3sg-present(root-stress) class: говори́!/гово́ри,
# носи́!/но́си — a REGULAR morphophonological alternation, but declarative
# speech (parliament/narration) starves the imperative reading (2026-07-05:
# говори́ 17 vs гово́ри 250, моли́ 0). Kept in tier-3 so the few real minority
# tokens are not gated out; true fix is minority-reading DATA (see guide 6e).
TIER3_CONTEXTUAL = {"говори", "чета", "против", "вълна", "пара", "коса",
                    "само", "моли", "носи", "прави", "купи", "учи", "ходи"}


def load_canonical(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return dict(db.execute("SELECT name, stressed FROM stress"))


def mark_with_speech(sent, uniq, ambig, func, speech_q, keep):
    """Like mark_chitanka_text.mark_sentence, plus a speech-label overlay:
    ambiguous tokens consume from speech_q (word -> deque of harvest rows in
    time order) and are labeled iff keep(word, marked) says so."""
    mask, pieces = [], []
    n_words = n_marked = n_speech = n_masked = 0
    pos = 0
    for m in WORD_RE.finditer(sent.lower()):
        n_words += 1
        s, e = m.span()
        word = m.group(0)
        orig = sent[s:e]
        pieces.append(sent[pos:s])
        if word in ambig and speech_q.get(word):
            r = speech_q[word].pop(0)
            if keep(word, r["marked"]):
                n_speech += 1
                pieces.append(apply_mark(orig, r["marked"]))
            else:
                n_masked += 1
                mask.append([s, e])
                pieces.append(orig)
        elif word in uniq:
            n_marked += 1
            pieces.append(apply_mark(orig, uniq[word]))
        elif word in func:
            pieces.append(orig)
        else:
            n_masked += 1
            mask.append([s, e])
            pieces.append(orig)
        pos = e
    pieces.append(sent[pos:])
    return "".join(pieces), mask, (n_words, n_marked, n_speech, n_masked)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", action="append", default=[],
                    help="harvest jsonl (repeatable: audiobooks + parliament)")
    ap.add_argument("--sqlite", default="data/bg_lexicon/bg_stress_candidates.sqlite")
    ap.add_argument("--canonical", default="data/bg_lexicon/bg_stress.sqlite")
    ap.add_argument("--function-words", default="config/bg/function_words_unstressed.json")
    ap.add_argument("--supplement", default="config/bg/lexicon_supplement.json")
    ap.add_argument("--min-share", type=float, default=0.90,
                    help="tier-1: min corpus share of the canonical variant")
    ap.add_argument("--tier3", default=None, help="extra tier-3 words, comma-sep")
    ap.add_argument("--exclude", default=None, help="extra tier-2 words, comma-sep")
    ap.add_argument("--oversample-minority", type=int, default=1,
                    help="repeat speech rows carrying a tier-3 MINORITY reading "
                         "N times, to counter register imbalance (говори́ 17 : "
                         "гово́ри 250). N=1 disables. NOTE: with single-digit "
                         "minority counts this boosts calibration, not "
                         "generalization — the real fix is more data (guide 6e-bis).")
    ap.add_argument("--extra-marked", action="append", default=[],
                    help="extra marked.jsonl file(s) to append verbatim, e.g. "
                         "data/chitanka/imperatives.jsonl from mine_imperatives.py "
                         "(rule-labeled imperative contexts). Repeatable.")
    ap.add_argument("--vet-results", default="data/kaldi/vetting_results.jsonl",
                    help="vet_stress_cli.py results jsonl: harvest tokens the "
                         "native judged 'bad' or 'unsure' are DROPPED before "
                         "any tier logic (per-token precision gate — the ft5 "
                         "чета inversion fix). '' disables.")
    ap.add_argument("--mix-chitanka", default="data/chitanka/marked.jsonl")
    ap.add_argument("--mix-ratio", type=int, default=5,
                    help="chitanka rows per speech row (0 = no mix)")
    ap.add_argument("--out", default="data/accentor_finetune.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tier3 = TIER3_CONTEXTUAL | set((args.tier3 or "").split(",")) - {""}
    tier2 = TIER2_EXCLUDE | set((args.exclude or "").split(",")) - {""}

    print("loading lexicons ...")
    uniq, ambig = load_lexicon(args.sqlite)
    func = load_function_words(args.function_words, args.supplement, uniq, ambig)
    canonical = load_canonical(args.canonical)

    # Native per-token veto: harvest tokens vetted bad/unsure are removed
    # before ANY counting or tier logic, so noisy minority readings can't
    # poison a class the way the 19%-precise че́та tokens inverted чета (ft5).
    vetoed = set()
    if args.vet_results and os.path.exists(args.vet_results):
        n_ok = 0
        with open(args.vet_results, encoding="utf-8") as f:
            for line in f:
                v = json.loads(line)
                key = (v["utt"], v["word"], v["marked"],
                       round(float(v["t0"]), 2))
                if v["verdict"] in ("bad", "unsure"):
                    vetoed.add(key)
                else:
                    n_ok += 1
        print(f"vet veto: {len(vetoed)} tokens dropped by native verdicts "
              f"({n_ok} vetted ok, kept)")

    # Pass 1: corpus-wide canonical share per word type (tier-1 threshold).
    by_utt = defaultdict(list)
    variant_counts = Counter()
    n_vetoed_seen = 0
    for path in args.labels:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if (r["utt"], r["word"], r["marked"],
                        round(float(r["t0"]), 2)) in vetoed:
                    n_vetoed_seen += 1
                    continue
                by_utt[r["utt"]].append(r)
                variant_counts[(r["word"], r["marked"])] += 1
    if vetoed:
        print(f"vet veto: {n_vetoed_seen} label rows actually removed")
    word_tot = Counter()
    for (w, _), c in variant_counts.items():
        word_tot[w] += c
    share = {w: variant_counts[(w, canonical.get(w))] / word_tot[w]
             for w in word_tot}

    def keep(word, marked):
        if word in tier2:
            return False
        if word in tier3:
            return True
        return (share.get(word, 0) >= args.min_share
                and marked == canonical.get(word))

    # Majority variant per tier-3 word — a labeled token of any OTHER variant
    # is a "minority reading" whose row we oversample (--oversample-minority).
    tier3_majority = {}
    for w in tier3:
        variants = {m: variant_counts[(w, m)]
                    for (ww, m) in variant_counts if ww == w}
        if variants:
            tier3_majority[w] = max(variants, key=variants.get)

    # Pass 2: one training row per utterance with >= 1 surviving speech label.
    rng = random.Random(args.seed)
    stats = Counter()
    tier3_kept = Counter()
    n_speech_rows = 0
    with open(args.out + ".tmp", "w", encoding="utf-8") as out:
        for utt, rows in by_utt.items():
            sent = " ".join(w for w in rows[0]["sentence"].split()
                            if w != "<unk>")
            if len(sent.split()) < 4 or len(sent) > 300:
                stats["skip_short"] += 1
                continue
            speech_q = defaultdict(list)
            for r in rows:  # file order == time order == text order
                speech_q[r["word"]].append(r)
            marked, mask, (nw, nm, ns, nk) = mark_with_speech(
                sent, uniq, ambig, func, speech_q, keep)
            if ns == 0:
                stats["skip_no_speech_label"] += 1
                continue
            if nk / max(nw, 1) > 0.5:
                stats["skip_too_masked"] += 1
                continue
            for r in rows:
                if r["word"] in tier3:
                    tier3_kept[r["marked"]] += 1
            # Oversample if this row carries a tier-3 minority reading.
            repeat = 1
            if args.oversample_minority > 1:
                if any(r["word"] in tier3_majority
                       and r["marked"] != tier3_majority[r["word"]]
                       for r in rows):
                    repeat = args.oversample_minority
                    stats["rows_oversampled"] += 1
            line = json.dumps({
                "id": f"sp_{utt}", "text": sent, "marked": marked,
                "mask_words": mask, "n_words": nw, "n_marked": nm,
                "n_speech": ns,
            }, ensure_ascii=False) + "\n"
            # Same id on every copy: train_accentor splits val by crc32(id), so
            # identical ids keep all copies in the SAME split (no train/val leak).
            for _ in range(repeat):
                out.write(line)
            n_speech_rows += repeat
            stats["speech_tokens_kept"] += ns * repeat

        n_extra = 0
        for xf in args.extra_marked:
            with open(xf, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n_extra += 1

        n_mix = 0
        if args.mix_ratio and args.mix_chitanka:
            want = n_speech_rows * args.mix_ratio
            reservoir = []
            with open(args.mix_chitanka, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if len(reservoir) < want:
                        reservoir.append(line)
                    else:
                        j = rng.randrange(i + 1)
                        if j < want:
                            reservoir[j] = line
            for line in reservoir:
                out.write(line)
            n_mix = len(reservoir)

    os.replace(args.out + ".tmp", args.out)
    print(f"speech rows: {n_speech_rows:,} ({stats['speech_tokens_kept']:,} "
          f"speech-labeled tokens)  chitanka mix: {n_mix:,}"
          + (f"  extra-marked: {n_extra:,}" if args.extra_marked else ""))
    if args.oversample_minority > 1:
        print(f"oversampled x{args.oversample_minority}: "
              f"{stats['rows_oversampled']:,} rows carrying a tier-3 minority "
              f"reading (calibration boost, not new contexts — guide 6e-bis)")
    print(f"skipped: {stats['skip_no_speech_label']:,} no surviving label, "
          f"{stats['skip_too_masked']:,} too masked, "
          f"{stats['skip_short']:,} short/long")
    print("tier-3 label counts:",
          "  ".join(f"{m}:{c}" for m, c in tier3_kept.most_common(12)))
    n_t1 = sum(1 for w in word_tot
               if w not in tier2 and w not in tier3
               and share.get(w, 0) >= args.min_share)
    print(f"tier-1 word types passing --min-share {args.min_share}: "
          f"{n_t1:,}/{len(word_tot):,}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
