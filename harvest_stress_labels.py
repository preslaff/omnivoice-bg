# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Harvest per-token stress labels from Kaldi alignment pronunciations.

The GMM aligner (kaldi_pilot.sh stage 3-4) chose ONE stress variant per
spoken token of every multi-pronunciation word — the first HUMAN-audio
contextual labels for ambiguous words (homographs), which text-only marking
had to mask. This script joins exp/tri3b_ali/prons.*.gz back to sentences
and audio, keeping only tokens of ambiguous words (>1 pron in the dict).

Output JSONL, one row per ambiguous-word token:
  {"utt", "word",                  # surface form (lowercase)
   "marked",                       # word with U+0301 as the audio said it
   "sentence",                     # normalized transcript of the clip
   "audio_path", "t0", "dur",      # word position inside the clip (sec)
   "n_variants"}                   # how many stress variants were possible

Uses: (a) native vetting of aligner quality (listen at t0), (b) Accentor
fine-tune rows once vetted, (c) homograph battery ground truth.

Usage:
    ./venv/bin/python harvest_stress_labels.py \
        --pilot exp/kaldi_pilot --ali exp/tri3b_ali \
        --out data/kaldi/stress_labels.jsonl
"""
import argparse
import glob
import gzip
import json
import os
import re
from collections import Counter, defaultdict

VOWEL_PHONE_RE = re.compile(r"^[a-z]+[01]$")
VOWELS = set("аъоуеиюяѝ")
FRAME_SEC = 0.01  # Kaldi default frame shift


def read_sym_table(path):
    syms = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            sym, idx = line.split()
            syms[int(idx)] = sym
    return syms


def read_kv(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            k, v = line.rstrip("\n").split(" ", 1)
            out[k] = v
    return out


def load_ambiguous(dict_lexicon):
    """words with >1 pronunciation in the ALIGNMENT dict = words the aligner
    actually had a choice about."""
    prons = defaultdict(set)
    with open(dict_lexicon, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) == 2:
                prons[parts[0]].add(parts[1])
    return {w: len(ps) for w, ps in prons.items() if len(ps) > 1}


def marked_from_phones(word, phones):
    """k-th vowel phone carries the stress bit for the k-th vowel letter
    (our G2P is letter-aligned for vowels)."""
    vowel_flags = [p.endswith("1") for p in phones if VOWEL_PHONE_RE.match(p)]
    vowel_idx = [i for i, ch in enumerate(word) if ch in VOWELS]
    if len(vowel_flags) != len(vowel_idx) or not any(vowel_flags):
        return None
    k = vowel_flags.index(True)
    i = vowel_idx[k]
    return word[: i + 1] + "́" + word[i + 1:]


def audio_path_from_wavscp(cmd):
    """wav.scp lines are ffmpeg pipes: '... -i <path> ...'."""
    m = re.search(r"-i (\S+)", cmd)
    return m.group(1) if m else cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", default="exp/kaldi_pilot")
    ap.add_argument("--ali", default="exp/tri3b_ali",
                    help="alignment dir (relative to --pilot)")
    ap.add_argument("--data", default="data/train",
                    help="data dir the alignment used (relative to --pilot), "
                         "e.g. data/parl_seg for the parliament pilot")
    ap.add_argument("--out", default="data/kaldi/stress_labels.jsonl")
    args = ap.parse_args()

    lang = os.path.join(args.pilot, "data", "lang")
    train = os.path.join(args.pilot, args.data)
    ali = os.path.join(args.pilot, args.ali)

    words = read_sym_table(os.path.join(lang, "words.txt"))
    phones = read_sym_table(os.path.join(lang, "phones.txt"))
    ambig = load_ambiguous(
        os.path.join(args.pilot, "data", "local", "dict", "lexicon.txt"))
    text = read_kv(os.path.join(train, "text"))
    wav = read_kv(os.path.join(train, "wav.scp"))
    # Segmented dirs (parliament): wav.scp is keyed by RECORDING; utterances
    # live in `segments` (utt -> reco start end). Map token times into the
    # recording so audio_path/t0 are directly listenable.
    seg = {}
    seg_path = os.path.join(train, "segments")
    if os.path.exists(seg_path):
        with open(seg_path) as f:
            for line in f:
                utt, reco, s, _ = line.split()
                seg[utt] = (reco, float(s))
    print(f"{len(ambig):,} ambiguous dict words; "
          f"{len(text):,} aligned utterances"
          + (f" ({len(seg):,} from segmented recordings)" if seg else ""))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_rows = n_badmap = 0
    per_word = Counter()
    with open(args.out + ".tmp", "w", encoding="utf-8") as of:
        for pron_gz in sorted(glob.glob(os.path.join(ali, "prons.*.gz"))):
            with gzip.open(pron_gz, "rt") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    utt, t0f, durf, wid = parts[0], parts[1], parts[2], parts[3]
                    word = words.get(int(wid), "")
                    if word not in ambig:
                        continue
                    # strip word-position suffixes (a1_B -> a1)
                    pron = [phones[int(p)].split("_")[0] for p in parts[4:]]
                    marked = marked_from_phones(word, pron)
                    if marked is None:
                        n_badmap += 1
                        continue
                    reco, offset = seg.get(utt, (utt, 0.0))
                    of.write(json.dumps({
                        "utt": utt, "word": word, "marked": marked,
                        "sentence": text.get(utt, ""),
                        "audio_path": audio_path_from_wavscp(wav.get(reco, "")),
                        "t0": round(offset + int(t0f) * FRAME_SEC, 2),
                        "dur": round(int(durf) * FRAME_SEC, 2),
                        "n_variants": ambig[word],
                    }, ensure_ascii=False) + "\n")
                    per_word[(word, marked)] += 1
                    n_rows += 1
    os.replace(args.out + ".tmp", args.out)

    n_types = len({w for w, _ in per_word})
    print(f"{n_rows:,} ambiguous-word tokens labeled "
          f"({n_types:,} word types, {n_badmap} phone-map mismatches skipped)")
    badrate = n_badmap / max(n_rows + n_badmap, 1)
    if badrate > 0.10:
        print(f"\n  *** WARNING: {badrate:.0%} phone-map mismatch — far above "
              f"the ~0.01% of a healthy harvest. The prons.*.gz in {ali} were "
              f"almost certainly written against a DIFFERENT words.txt/phones.txt "
              f"than {lang} (e.g. data/lang was rebuilt after this alignment). "
              f"Re-run steps/get_prons.sh against the CURRENT lang before "
              f"trusting these labels. ***")
    print(f"-> {args.out}")
    # Words where the audio actually split between variants are the gold:
    by_word = defaultdict(list)
    for (w, m), c in per_word.items():
        by_word[w].append((c, m))
    split = {w: sorted(v, reverse=True) for w, v in by_word.items()
             if len(v) > 1}
    print(f"\n{len(split):,} word types with BOTH variants observed; "
          f"top 20 by evidence:")
    for w, v in sorted(split.items(),
                       key=lambda kv: -min(c for c, _ in kv[1]))[:20]:
        print("  " + "  ".join(f"{m}:{c}" for c, m in v))


if __name__ == "__main__":
    main()
