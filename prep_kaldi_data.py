# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Format OmniVoice-BG assets into Kaldi data/dict directories.

Phase-2 groundwork for the Accentor reproduction (annotation AM, gap item 2):
builds the pronunciation dictionary from the L2/L3 lexicon (stressed/
unstressed vowel phones -> the AM learns to hear stress) and utterance data
dirs from the real-audiobook manifests (sentence clips, exact text — the
easiest alignment food; parliament joins later via the same dict).

The dict is restricted to the transcript vocabulary of the given --set
manifests (all stress variants of those words kept). The full 3.35M-entry
lexicon compiles to a 1.2 GB L.fst and OOMs compile-train-graphs at nj=24
(2026-07-04: killed the whole desktop session twice — looked like a machine
restart). Opt out with --full-lexicon only on a much bigger box.

Outputs (under --out, default data/kaldi):
  local/dict/{lexicon.txt,nonsilence_phones.txt,silence_phones.txt,
              optional_silence.txt,extra_questions.txt}
  <setname>/{wav.scp,text,utt2spk,spk2utt}     per --set name=manifest.jsonl

wav.scp uses ffmpeg pipes to feed 16 kHz mono WAV (ASR-standard rate; the
24 kHz FLACs on disk stay untouched).

Usage:
    ./venv/bin/python prep_kaldi_data.py --out data/kaldi \
        --set sivov=data/bg/train.jsonl --set bg400=data/bg400/train.jsonl
"""
import argparse
import json
import os
import re

WORD_RE = re.compile(r"[а-яьѝ]+(?:-[а-яьѝ]+)*")
PHONE_RE = re.compile(r"^[a-z]+[01]?$")


def write_dict(lexicon_path, phones_path, dict_dir, vocab=None):
    """vocab: restrict the lexicon to these surface forms (ALL their stress
    variants kept — that is what alignment chooses between). The full 3.35M
    lexicon compiles into a 1.2 GB L.fst; compile-train-graphs then holds
    ~2.6 GB per job x nj=24 = OOM on a 60 GB box (killed the user session
    twice on 2026-07-04). Alignment only needs transcript words; the full
    lexicon returns later via tiny per-sentence stress-candidate graphs."""
    os.makedirs(dict_dir, exist_ok=True)
    phones = []
    with open(phones_path, encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok.startswith("#") or not tok:
                continue
            if PHONE_RE.match(tok) and tok not in phones:
                phones.append(tok)
    stressed = [p for p in phones if p.endswith("1")]
    unstressed = [p for p in phones if p.endswith("0")]
    consonants = [p for p in phones if not p[-1].isdigit()]
    with open(os.path.join(dict_dir, "nonsilence_phones.txt"), "w") as f:
        f.write("\n".join(phones) + "\n")
    with open(os.path.join(dict_dir, "silence_phones.txt"), "w") as f:
        f.write("SIL\nSPN\n")
    with open(os.path.join(dict_dir, "optional_silence.txt"), "w") as f:
        f.write("SIL\n")
    # Stress-aware tree questions: let the AM split on stressed vs unstressed.
    with open(os.path.join(dict_dir, "extra_questions.txt"), "w") as f:
        f.write(" ".join(stressed) + "\n")
        f.write(" ".join(unstressed) + "\n")
        f.write(" ".join(consonants) + "\n")
        f.write("SIL SPN\n")
    n = n_multi = n_dup = n_oov = 0
    seen = set()
    covered = set()
    with open(os.path.join(dict_dir, "lexicon.txt"), "w",
              encoding="utf-8") as out, open(lexicon_path,
                                             encoding="utf-8") as f:
        out.write("!SIL SIL\n<unk> SPN\n")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            # validate_dict_dir.pl rejects exact duplicate lines (a few exist
            # in lexicon_full.txt from the L3 merge).
            if line in seen:
                n_dup += 1
                continue
            seen.add(line)
            # rechko carries multi-word entries (да-forms, reflexives:
            # "да феминизирате", "чепкам се") — an alignment lexicon is
            # single-word (transcripts are tokenized), and Kaldi would parse
            # the 2nd word as a phone. Drop them.
            if " " in parts[0]:
                n_multi += 1
                continue
            if vocab is not None and parts[0] not in vocab:
                n_oov += 1
                continue
            covered.add(parts[0])
            out.write(f"{parts[0]} {parts[1]}\n")
            n += 1
    print(f"dict: {n:,} pronunciations ({n_multi:,} multi-word entries, "
          f"{n_dup:,} duplicate lines dropped), {len(phones)} phones "
          f"({len(stressed)} stressed / {len(unstressed)} unstressed vowels) "
          f"-> {dict_dir}")
    if vocab is not None:
        missing = len(vocab) - len(covered)
        print(f"  vocab-limited: {len(covered):,}/{len(vocab):,} transcript "
              f"types in lexicon ({missing:,} map to <unk>); "
              f"{n_oov:,} out-of-transcript pronunciations dropped")


def write_set(name, manifest, out_dir, sr, spk_chunk=200):
    """Single-narrator corpora get PSEUDO-SPEAKERS of --spk-chunk utterances:
    Kaldi parallelizes by speaker (2 real speakers < nj), and SAT/fMLLR wants
    per-speaker transforms estimated on minutes, not 200 h."""
    os.makedirs(out_dir, exist_ok=True)
    raw = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            words = WORD_RE.findall(r["text"].lower())
            if not words or not os.path.exists(r["audio_path"]):
                continue
            raw.append((r["id"], r["audio_path"], " ".join(words)))
    raw.sort()
    rows = []
    for i, (rid, path, text) in enumerate(raw):
        spk = f"{name}-s{i // spk_chunk:04d}"
        rows.append((f"{spk}-{rid}", spk, path, text))
    rows.sort()
    spk2utt = {}
    with open(os.path.join(out_dir, "wav.scp"), "w") as wf, open(
        os.path.join(out_dir, "text"), "w", encoding="utf-8"
    ) as tf, open(os.path.join(out_dir, "utt2spk"), "w") as uf:
        for utt, spk, path, text in rows:
            wf.write(f"{utt} ffmpeg -nostdin -v error -i {path} "
                     f"-ac 1 -ar {sr} -f wav - |\n")
            tf.write(f"{utt} {text}\n")
            uf.write(f"{utt} {spk}\n")
            spk2utt.setdefault(spk, []).append(utt)
    with open(os.path.join(out_dir, "spk2utt"), "w") as f:
        for spk in sorted(spk2utt):
            f.write(spk + " " + " ".join(spk2utt[spk]) + "\n")
    print(f"set {name}: {len(rows):,} utts, {len(spk2utt)} pseudo-speakers "
          f"-> {out_dir}")
    return {w for _, _, _, text in rows for w in text.split()}


def write_long_set(name, manifest, out_dir, sr, min_words=100):
    """LONG-RECORDING sets (parliament sessions): ONE utterance per manifest
    row, speaker = utterance (per-session channel adaptation; real per-turn
    speakers come later via turn matching). Feed the result to
    steps/cleanup/segment_long_utterances.sh — do NOT train on it directly.
    Rows with < min_words transcript words are skipped (unpublished
    stenograms)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    n_skip = 0
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            words = WORD_RE.findall((r.get("text") or "").lower())
            if len(words) < min_words or not os.path.exists(r["audio_path"]):
                n_skip += 1
                continue
            rows.append((r["id"], r["audio_path"], " ".join(words)))
    rows.sort()
    with open(os.path.join(out_dir, "wav.scp"), "w") as wf, open(
        os.path.join(out_dir, "text"), "w", encoding="utf-8"
    ) as tf, open(os.path.join(out_dir, "utt2spk"), "w") as uf, open(
        os.path.join(out_dir, "spk2utt"), "w"
    ) as sf:
        for utt, path, text in rows:
            wf.write(f"{utt} ffmpeg -nostdin -v error -i {path} "
                     f"-ac 1 -ar {sr} -f wav - |\n")
            tf.write(f"{utt} {text}\n")
            uf.write(f"{utt} {utt}\n")
            sf.write(f"{utt} {utt}\n")
    print(f"long-set {name}: {len(rows):,} recordings "
          f"({n_skip} skipped: short/missing) -> {out_dir}")
    return {w for _, _, text in rows for w in text.split()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/kaldi")
    ap.add_argument("--lexicon", default="data/bg_lexicon/lexicon_full.txt")
    ap.add_argument("--phones", default="data/bg_lexicon/phones.txt")
    ap.add_argument("--set", action="append", default=[],
                    metavar="NAME=MANIFEST.jsonl",
                    help="speaker-set name + manifest (repeatable)")
    ap.add_argument("--long-set", action="append", default=[],
                    metavar="NAME=MANIFEST.jsonl",
                    help="long-recording set (1 utt/row, spk=utt) for "
                         "segment_long_utterances.sh (repeatable)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--spk-chunk", type=int, default=200,
                    help="utterances per pseudo-speaker")
    ap.add_argument("--full-lexicon", action="store_true",
                    help="do NOT restrict the dict to transcript vocabulary "
                         "(WARNING: 1.2 GB L.fst -> compile-train-graphs "
                         "OOMs at nj=24 on 60 GB RAM)")
    args = ap.parse_args()

    vocab = set()
    for spec in args.set:
        name, _, manifest = spec.partition("=")
        vocab |= write_set(name, manifest, os.path.join(args.out, name),
                           args.sample_rate, args.spk_chunk)
    for spec in args.long_set:
        name, _, manifest = spec.partition("=")
        vocab |= write_long_set(name, manifest, os.path.join(args.out, name),
                                args.sample_rate)
    write_dict(args.lexicon, args.phones,
               os.path.join(args.out, "local", "dict"),
               vocab=None if (args.full_lexicon or not vocab) else vocab)


if __name__ == "__main__":
    main()
