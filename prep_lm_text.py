#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Prepare the LM training text for the Bulgarian ASR n-gram (KenLM).

Concatenates the Kaldi `text` transcripts of the TRAIN-split recordings — already
normalized to match the ASR targets (lowercase, no punctuation) — EXCLUDING the
held-out dev/test parliament sessions (leakage guard, `make_asr_split.py`). This
covers both in-domain parliament and literary audiobook speech, so no separate
chitanka corpus is needed for a first LM (add one later for extra OOV coverage).

Stress marks (U+0301) are stripped defensively so the LM vocabulary is plain
orthography, matching the ASR/BPE text.

    ./venv/bin/python prep_lm_text.py            # -> data/lm/lm_text.txt
"""
import argparse
import pathlib
import re

ACUTE = "́"


def norm(s):
    # strip stress marks + drop <unk> OOV tokens (kenlm reserves <unk>; the Kaldi
    # `text` marks out-of-vocab words as <unk> inline).
    toks = [t for t in s.replace(ACUTE, "").split() if t != "<unk>"]
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=[
        "exp/kaldi_pilot/data/train_cleaned",
        "exp/kaldi_pilot/data/parl_new_cleaned",
        "exp/kaldi_pilot/data/parl_new2_cleaned",
    ])
    ap.add_argument("--split", default="data/lhotse/bg/split")
    ap.add_argument("--out", default="data/lm/lm_text.txt")
    args = ap.parse_args()

    held = set()
    for f in ("test_recordings.txt", "dev_recordings.txt"):
        p = pathlib.Path(args.split) / f
        if p.exists():
            held |= set(p.read_text(encoding="utf-8").split())
    print(f"held-out recordings excluded from LM: {len(held)}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_skip = n_words = 0
    seen_utt = set()
    with open(out, "w", encoding="utf-8") as w:
        for d in args.dirs:
            t = pathlib.Path(d) / "text"
            if not t.exists():
                print(f"  [skip] {t} absent")
                continue
            for line in t.read_text(encoding="utf-8").splitlines():
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                utt, txt = parts
                if utt in seen_utt:              # dirs can overlap; dedup by utt
                    continue
                seen_utt.add(utt)
                rec = utt.split("-", 1)[0]        # parliament session id; audiobook = whole
                if rec in held:
                    n_skip += 1
                    continue
                txt = norm(txt)
                if txt and txt != "<unk>":
                    w.write(txt + "\n")
                    n_in += 1
                    n_words += len(txt.split())
    print(f"wrote {n_in:,} lines / {n_words:,} words (skipped {n_skip:,} held-out) -> {out}")


if __name__ == "__main__":
    main()
