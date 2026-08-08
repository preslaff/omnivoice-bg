# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Qualitative evaluation of an Accentor checkpoint.

Three test sets, in increasing difficulty:
  1. HARD WORDS (built-in): the TrainGen failure set — unambiguous dictionary
     words incl. UI imperatives absent from all training audio. The text-only
     pretrain is expected to get ALL of these right.
  2. --sentences FILE: free-form lines (e.g. battery_sentences.txt) — prints
     marked output + per-word confidence for eyeballing.
  3. --battery JSON (homograph_battery_sentences.json): homograph contexts
     with expected readings — scored PASS/FAIL. NOTE: homographs are MASKED in
     text-only training, so failures here are the EXPECTED BASELINE that the
     speech-labeled fine-tune must beat; record, don't despair.

Usage:
    ./venv/bin/python eval_accentor.py --ckpt exp/accentor_text/best.pt \
        --sentences battery_sentences.txt \
        --battery config/bg/homograph_battery_sentences.json
"""
import argparse
import json
import re

import torch

from accentor_model import (VOWELS, Accentor, encode_text, mark_text,
                            _WORD_SPAN_RE)

HARD_SENTENCES = [
    "Налягането в системата расте бързо.",
    "Натиснете бутона и въведете паролата.",
    "Изберете файл от менюто и отворете прозореца.",
    "Запишете промените, преди да затворите приложението.",
    "Свалете архива и разархивирайте съдържанието.",
]


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = dict(ck["config"])
    cfg["kernel"] = tuple(cfg["kernel"])
    model = Accentor(**cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_path} (step {ck.get('step')}, "
          f"best SER {ck.get('best_ser', float('nan')):.3%})")
    return model


@torch.no_grad()
def word_confidences(model, text, device):
    """-> list of (word, argmax_vowel_char_idx, prob) per word."""
    ids, caps = encode_text(text)
    ids_t = torch.tensor([ids], device=device)
    caps_t = torch.tensor([caps], device=device)
    mask = torch.ones_like(ids_t, dtype=torch.bool)
    probs = torch.sigmoid(model(ids_t, caps_t, mask))[0].tolist()
    low = text.lower()
    out = []
    for m in _WORD_SPAN_RE.finditer(text.replace("ѝ", "и")):
        vpos = [i for i in range(*m.span()) if low[i] in VOWELS]
        if vpos:
            best = max(vpos, key=lambda i: probs[i])
            out.append((text[m.start():m.end()], best, probs[best]))
    return out


def show(model, text, device):
    marked = mark_text(model, text, device=device)
    confs = " ".join(f"{w}:{p:.2f}" for w, _, p in
                     word_confidences(model, text, device))
    print(f"  {marked}\n    conf: {confs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="exp/accentor_text/best.pt")
    ap.add_argument("--sentences", default=None)
    ap.add_argument("--battery", default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine (19M params); cuda if free")
    args = ap.parse_args()
    device = args.device
    model = load(args.ckpt, device)

    print("\n=== 1. Hard words (unambiguous — expect ALL correct) ===")
    for s in HARD_SENTENCES:
        show(model, s, device)

    if args.sentences:
        print(f"\n=== 2. {args.sentences} ===")
        with open(args.sentences, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    show(model, line, device)

    if args.battery:
        print(f"\n=== 3. Homograph battery (masked in training — baseline, "
              f"failures expected) ===")
        with open(args.battery, encoding="utf-8") as f:
            entries = json.load(f)["sentences"]
        n_pass = 0
        for e in entries:
            marked = mark_text(model, e["sentence"], device=device).lower()
            ok = e["expected"] in marked
            n_pass += ok
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {e['surface']} -> expect {e['expected']} "
                  f"({e['gloss']})")
            print(f"         {marked}")
        print(f"\n  battery: {n_pass}/{len(entries)} "
              f"(text-only baseline; fine-tune target is ~all)")


if __name__ == "__main__":
    main()
