#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Prepare the LM training text for the Bulgarian ASR n-gram (KenLM).

Two sources, combinable:

* ``--dirs``  Kaldi `text` transcripts (v1 default) — parliament + AB400 audiobook.
* ``--cuts``  lhotse cut manifests (``.jsonl.gz``) — takes the supervision text of
  every cut. This is the GP path: pointing it at the GP **train** manifest gives the
  LM exactly the text the acoustic model was trained on, all axes included (BNR,
  YouTube, FLEURS/CV train), so the LM's coverage matches the AM's instead of
  lagging a corpus behind.

Both are EXCLUDING the held-out dev/test parliament sessions (leakage guard,
`make_asr_split.py`); ``--exclude-cuts`` additionally drops any recording that
appears in an eval manifest (FLEURS/CV test, whose sessions aren't in the
parliament split lists).

Stress marks (U+0301) are stripped defensively so the LM vocabulary is plain
orthography, matching the ASR/BPE text.

    ./venv/bin/python prep_lm_text.py            # v1 -> data/lm/lm_text.txt
    ./venv/bin/python prep_lm_text.py --dirs --cuts <gp_train.jsonl.gz> \
        --exclude-cuts <dev> <test> <fleurs> <cv> --out data/lm/lm_text_gp.txt
"""
import argparse
import gzip
import json
import pathlib
import re

ACUTE = "́"


def norm(s):
    # strip stress marks + drop <unk> OOV tokens (kenlm reserves <unk>; the Kaldi
    # `text` marks out-of-vocab words as <unk> inline).
    toks = [t for t in s.replace(ACUTE, "").split() if t != "<unk>"]
    return " ".join(toks)


def iter_cuts(path):
    """Yield (cut_id, recording_id, text) for every supervision in a lhotse manifest."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            cut = json.loads(line)
            for sup in cut.get("supervisions", []):
                txt = sup.get("text")
                if txt:
                    yield cut["id"], sup.get("recording_id", ""), txt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="Kaldi data dirs with a `text` file; defaults to the v1 trio "
                         "unless --cuts is given (pass an explicit list to combine both)")
    ap.add_argument("--cuts", nargs="*", default=[],
                    help="lhotse cut manifests (.jsonl.gz) to take supervision text from")
    ap.add_argument("--exclude-cuts", nargs="*", default=[],
                    help="eval manifests whose recordings must never enter the LM")
    ap.add_argument("--split", default="data/lhotse/bg/split")
    ap.add_argument("--out", default="data/lm/lm_text.txt")
    args = ap.parse_args()
    if args.dirs is None:                             # v1 default only when no cuts given
        args.dirs = [] if args.cuts else [
            "exp/kaldi_pilot/data/train_cleaned",
            "exp/kaldi_pilot/data/parl_new_cleaned",
            "exp/kaldi_pilot/data/parl_new2_cleaned",
        ]

    held = set()
    for f in ("test_recordings.txt", "dev_recordings.txt"):
        p = pathlib.Path(args.split) / f
        if p.exists():
            held |= set(p.read_text(encoding="utf-8").split())
    n_split_held = len(held)
    held_ids = set()                                  # cut ids, for the eval manifests
    for p in args.exclude_cuts:
        for cut_id, rec, _ in iter_cuts(p):
            held.add(rec)
            held_ids.add(cut_id)
    print(f"held-out recordings excluded from LM: {len(held):,} "
          f"({n_split_held} from the split lists, rest from {len(args.exclude_cuts)} eval manifests)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_skip = n_words = 0
    seen_utt = set()
    with open(out, "w", encoding="utf-8") as w:
        for c in args.cuts:
            n_axis = 0
            for cut_id, rec, txt in iter_cuts(c):
                if cut_id in seen_utt:
                    continue
                seen_utt.add(cut_id)
                if rec in held or cut_id in held_ids:
                    n_skip += 1
                    continue
                txt = norm(txt)
                if txt and txt != "<unk>":
                    w.write(txt + "\n")
                    n_in += 1
                    n_axis += 1
                    n_words += len(txt.split())
            print(f"  [cuts] {c}: +{n_axis:,} lines")
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
