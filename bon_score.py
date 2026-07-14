#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Stage 8 (RLAIF ladder), step 3: score the judged best-of-N candidates.

Joins the aligner's per-candidate pronunciation choices (which stress
variant the TTS actually rendered) against the Accentor's marks (which
variant it was TOLD to render), then reports the three numbers the ladder
decision hangs on:

  single-sample  — accuracy of candidate 0 alone (today's serving behavior)
  best-of-N      — accuracy when the Kaldi judge picks the best candidate
  oracle         — sentences where at least one candidate rendered ALL
                   targets correctly (ceiling for any selection scheme)

Scope note: the judge can only adjudicate words with >1 stress variant in
the alignment lexicon (elsewhere forced alignment has no choice). That is
exactly the homograph boundary set the ladder targets.

Also copies, per sentence, candidate 0 and the judge's pick into
<out>/picked/ as NN__c00.wav / NN__bon.wav for by-ear A/B.

Usage (defaults match bon_generate.py/bon_judge.sh):
  ./venv/bin/python bon_score.py
"""
import argparse
import glob
import gzip
import json
import os
import re
import shutil
from collections import defaultdict

import harvest_stress_labels as H

MARKED_WORD_RE = re.compile(r"[а-яьѝ́]+(?:-[а-яьѝ́]+)*")


def load_variants(dict_lexicon):
    """word -> set of U+0301-marked renderings derivable from its prons."""
    variants = defaultdict(set)
    with open(dict_lexicon, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) != 2:
                continue
            word, pron = parts
            m = H.marked_from_phones(word, pron.split())
            if m:
                variants[word].add(m)
    return variants


def targets_from_marked(marked_sentence, variants):
    """(word, wanted_marked, judgeable) for every Accentor-marked word whose
    surface form has >1 stress variant in the alignment lexicon."""
    out = []
    for tok in MARKED_WORD_RE.findall(marked_sentence.lower()):
        if "́" not in tok:
            continue
        plain = tok.replace("́", "")
        vs = variants.get(plain, set())
        if len(vs) > 1:
            out.append((plain, tok, tok in vs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", default="exp/kaldi_pilot")
    ap.add_argument("--ali", default="exp/tri3b_comb_bon_ali")
    ap.add_argument("--manifest", default="bon_eval/manifest.jsonl")
    ap.add_argument("--out", default="bon_eval")
    args = ap.parse_args()

    lang = os.path.join(args.pilot, "data", "lang")
    words = H.read_sym_table(os.path.join(lang, "words.txt"))
    phones = H.read_sym_table(os.path.join(lang, "phones.txt"))
    variants = load_variants(
        os.path.join(args.pilot, "data", "local", "dict", "lexicon.txt"))

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    by_utt = {r["utt"]: r for r in rows}

    # utt -> word -> [rendered marked, in spoken order]
    rendered = defaultdict(lambda: defaultdict(list))
    n_badmap = 0
    for pron_gz in sorted(glob.glob(
            os.path.join(args.pilot, args.ali, "prons.*.gz"))):
        with gzip.open(pron_gz, "rt") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5 or parts[0] not in by_utt:
                    continue
                word = words.get(int(parts[3]), "")
                if len(variants.get(word, ())) < 2:
                    continue
                pron = [phones[int(p)].split("_")[0] for p in parts[4:]]
                m = H.marked_from_phones(word, pron)
                if m is None:
                    n_badmap += 1
                    continue
                rendered[parts[0]][word].append(m)

    # Score every candidate.
    n_unjudgeable = 0
    for r in rows:
        tgt = targets_from_marked(r["marked"], variants)
        r["targets"], r["n_match"], r["n_miss"] = [], 0, 0
        for word, want, judgeable in tgt:
            if not judgeable:  # Accentor mark not among dict variants
                n_unjudgeable += 1
                continue
            got_list = rendered.get(r["utt"], {}).get(word, [])
            got = got_list.pop(0) if got_list else None  # positional for dups
            ok = got == want
            r["targets"].append(
                {"word": word, "want": want, "got": got, "ok": ok})
            r["n_match"] += ok
            r["n_miss"] += got is None
        r["n_targets"] = len(r["targets"])

    by_sent = defaultdict(list)
    for r in rows:
        by_sent[r["sent_idx"]].append(r)
    for cands in by_sent.values():
        cands.sort(key=lambda r: r["cand_idx"])

    n_cands = max(len(c) for c in by_sent.values())
    scored = {si: c for si, c in by_sent.items() if c[0]["n_targets"]}
    print(f"{len(by_sent)} sentences, {len(scored)} with judgeable targets "
          f"({n_badmap} phone-map mismatches, "
          f"{n_unjudgeable // max(n_cands, 1)} unjudgeable marks)\n")

    # Per-word table: how many of N candidates rendered each target correctly.
    print(f"{'sent':>4}  {'target':<14} {'wanted':<14} correct/N   renders")
    tot_single = tot_best = tot_oracle_t = tot_t = 0
    picked = {}
    for si in sorted(scored):
        cands = scored[si]
        n = len(cands)
        for k in range(cands[0]["n_targets"]):
            word = cands[0]["targets"][k]["word"]
            want = cands[0]["targets"][k]["want"]
            oks = [c["targets"][k]["ok"] for c in cands]
            gots = [c["targets"][k]["got"] or "-" for c in cands]
            counts = defaultdict(int)
            for g in gots:
                counts[g] += 1
            dist = " ".join(f"{g}:{c}" for g, c in
                            sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"{si:>4}  {word:<14} {want:<14} {sum(oks)}/{n}         {dist}")
            tot_t += 1
            tot_oracle_t += any(oks)
        best = max(cands, key=lambda c: (c["n_match"], -c["n_miss"],
                                         -c["cand_idx"]))
        picked[si] = best
        tot_single += cands[0]["n_match"] == cands[0]["n_targets"]
        tot_best += best["n_match"] == best["n_targets"]

    n_s = len(scored)
    oracle = sum(any(c["n_match"] == c["n_targets"] for c in cands)
                 for cands in scored.values())
    print(f"\nsentence level (all targets correct), {n_s} scored sentences:")
    print(f"  single-sample (c00) : {tot_single}/{n_s}")
    print(f"  judge best-of-N     : {tot_best}/{n_s}")
    print(f"  oracle (any cand)   : {oracle}/{n_s}")
    print(f"target level: {tot_oracle_t}/{tot_t} targets rendered correctly "
          f"by >=1 candidate")

    # By-ear A/B: candidate 0 vs the judge's pick.
    pickdir = os.path.join(args.out, "picked")
    os.makedirs(pickdir, exist_ok=True)
    for si, best in picked.items():
        shutil.copy(by_sent[si][0]["wav"],
                    os.path.join(pickdir, f"{si:02d}__c00.wav"))
        shutil.copy(best["wav"],
                    os.path.join(pickdir, f"{si:02d}__bon.wav"))
    with open(os.path.join(args.out, "bon_scores.jsonl"), "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nlisten: {pickdir}/  (NN__c00 = today's behavior, "
          f"NN__bon = judge's pick)")
    print(f"full per-candidate detail: {args.out}/bon_scores.jsonl")


if __name__ == "__main__":
    main()
