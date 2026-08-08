# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Out-of-domain evaluation of an Accentor checkpoint on parliament stenograms.

The Accentor trains on chitanka FICTION; parliament stenograms are formal
POLITICAL speech — a different register, vocabulary, and syntax. This is the
generalization test the training-time val split (also chitanka) cannot give.

Two metrics, both on never-trained text:
  1. AUTOMATIC SER (no human) — for every UNAMBIGUOUS lexicon word (single
     stress candidate), compare the model's argmax-stressed vowel against the
     dictionary's. This measures whether lexical stress generalizes to a new
     domain. Reported with denominator + a function-word zero-error rate.
  2. HOMOGRAPH-IN-THE-WILD (native review) — sentences containing the
     homograph verbs, marked by the model, for a native to eyeball the
     contextual choice (no automatic ground truth without the audio).

Also emits a random sample of fully-marked sentences for a native read.

    ./venv/bin/python eval_accentor_ood.py --ckpt exp/accentor_ft4/best.pt \
        --transcripts data/bg_parliament/transcripts --n 4000 --show 25
"""
import argparse
import glob
import json
import os
import random

import torch

from accentor_model import ACUTE, VOWELS, Accentor, encode_text, mark_text
from bg_chitanka_clean import split_sentences
from mark_chitanka_text import load_function_words, load_lexicon
from select_chitanka_sentences import WORD_RE, sentence_ok

HOMOGRAPHS = ["говори", "носи", "прави", "учи", "ходи", "моли", "купи",
              "скочи", "чета", "вълна", "пара", "коса", "само"]


def lex_ordinal(marked):
    """0-based index of the stressed vowel AMONG the word's vowels, from a
    lexicon mark (or None)."""
    i = marked.find(ACUTE)
    if i <= 0:
        return None
    return sum(1 for c in marked[:i] if c in VOWELS) - 1


def load_model(ckpt, device):
    c = torch.load(ckpt, map_location=device)
    cfg = dict(c["config"]); cfg["kernel"] = tuple(cfg["kernel"])
    m = Accentor(**cfg).to(device); m.load_state_dict(c["model"]); m.eval()
    print(f"loaded {ckpt} (step {c.get('step')}, "
          f"train-SER {c.get('best_ser', float('nan')):.3%})")
    return m


@torch.no_grad()
def predict_ordinals(model, text, device):
    """-> list of (word_lc, span, argmax_vowel_ordinal, prob)."""
    ids, caps = encode_text(text)
    probs = torch.sigmoid(model(torch.tensor([ids], device=device),
                                torch.tensor([caps], device=device),
                                torch.ones(1, len(ids), dtype=torch.bool,
                                           device=device)))[0].tolist()
    low = text.lower().replace("ѝ", "и")
    out = []
    for m in WORD_RE.finditer(low):
        s, e = m.span()
        vpos = [i for i in range(s, e) if low[i] in VOWELS]
        if not vpos:
            continue
        k = max(range(len(vpos)), key=lambda j: probs[vpos[j]])
        out.append((m.group(0), (s, e), k, probs[vpos[k]]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="exp/accentor_ft3/best.pt")
    ap.add_argument("--transcripts", default="data/bg_parliament/transcripts")
    ap.add_argument("--sqlite", default="data/bg_lexicon/bg_stress_candidates.sqlite")
    ap.add_argument("--function-words", default="config/bg/function_words_unstressed.json")
    ap.add_argument("--supplement", default="config/bg/lexicon_supplement.json")
    ap.add_argument("--n", type=int, default=4000, help="sentences to sample")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--per-homograph", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)
    uniq, ambig = load_lexicon(args.sqlite)
    func = load_function_words(args.function_words, args.supplement, uniq, ambig)
    print(f"lexicon: {len(uniq):,} unambiguous, {len(ambig):,} ambiguous")

    # sample quality-gated sentences from the stenograms
    rng = random.Random(args.seed)
    files = sorted(glob.glob(os.path.join(args.transcripts, "*.txt")))
    sents = []
    for fp in files:
        for par in open(fp, encoding="utf-8").read().splitlines():
            for s in split_sentences(par):
                if sentence_ok(s, 25, 250):
                    sents.append(s)
    rng.shuffle(sents)
    sents = sents[:args.n]
    print(f"sampled {len(sents):,} parliament sentences from {len(files)} files")

    n_correct = n_eval = n_oov = n_ambig = n_words = 0
    zero_err = zero_n = 0
    conf_sum = 0.0
    homo_hits = {w: [] for w in HOMOGRAPHS}
    for s in sents:
        for word, span, k, p in predict_ordinals(model, s, args.device):
            n_words += 1
            if word in uniq:
                exp = lex_ordinal(uniq[word])
                if exp is None:
                    continue
                n_eval += 1
                conf_sum += p
                if k == exp and p >= args.threshold:
                    n_correct += 1
            elif word in func:
                zero_n += 1
                if p >= args.threshold:
                    zero_err += 1
            elif word in ambig:
                n_ambig += 1
                if word in homo_hits and len(homo_hits[word]) < 200:
                    homo_hits[word].append(s)
            else:
                n_oov += 1

    ser = 1 - n_correct / max(n_eval, 1)
    print("\n=== automatic SER on parliament (out-of-domain) ===")
    print(f"  unambiguous words evaluated : {n_eval:,}")
    print(f"  stress error rate (SER)     : {ser:.3%}")
    print(f"  mean confidence             : {conf_sum/max(n_eval,1):.3f}")
    print(f"  function-word zero-error    : {zero_err/max(zero_n,1):.3%} "
          f"(n={zero_n:,})")
    print(f"  coverage: {n_eval/max(n_words,1):.0%} unambiguous, "
          f"{n_ambig/max(n_words,1):.0%} ambiguous, {n_oov/max(n_words,1):.0%} OOV")

    print(f"\n=== homograph-in-the-wild (native review — no auto ground truth) ===")
    for w in HOMOGRAPHS:
        picks = homo_hits.get(w, [])
        if not picks:
            continue
        for s in rng.sample(picks, min(args.per_homograph, len(picks))):
            print(f"  [{w}] {mark_text(model, s, device=args.device)}")

    if args.show:
        print(f"\n=== {args.show} random marked sentences (native read) ===")
        for s in rng.sample(sents, min(args.show, len(sents))):
            print(f"  {mark_text(model, s, device=args.device)}")


if __name__ == "__main__":
    main()
