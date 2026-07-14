#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Stage 8 (RLAIF ladder), step 1: best-of-N ORACLE experiment — generation.

Context: the 63h->413h acoustic scale-up moved eval loss (3.88->3.43) but
not stress-rendering fidelity, and marking consistency was already saturated
at ft5b. The residual is architectural: the diffusion LM *samples* renderings
that don't always obey the U+0301 mark. Before building selection or
preference training, measure whether correct renderings EXIST in the
sampling distribution at all:

  1. bon_generate.py (this): for each border sentence, synthesize N
     candidates from one checkpoint using the exact serve-time decoding
     settings — only the seed varies. position_temperature=5.0 (default)
     makes the token-unmasking ORDER stochastic, so seeds give genuinely
     different renderings without leaving the production distribution.
  2. bon_judge.sh: stress-aware forced alignment of every candidate
     (tri3b_comb + the multi-pronunciation lexicon): the aligner decides
     which stress variant each candidate actually rendered — the same
     mechanism that harvested the 1.22M training labels, now as a judge.
  3. bon_score.py: single-sample vs judge-picked best-of-N vs oracle.

Reading the result: oracle >> single-sample -> reranking (and later DPO on
judge-labeled pairs) has headroom. oracle ~= single-sample -> the model
cannot render the contrast; the fix is conditioning, not selection.

Outputs:
  <out>/wavs/sXXcYY.wav        24 kHz candidates
  <out>/manifest.jsonl         one row per candidate (utt, texts, wav)
  <out>/sentences.txt          index
  <kaldi-data>/                ready Kaldi data dir (16 kHz ffmpeg pipes)

Usage:
  ./venv/bin/python bon_generate.py \
      --model exp/omnivoice_bg_full/checkpoint-6500 \
      --sentences eval_ab_border.txt \
      --accentor exp/accentor_ft5b/best.pt \
      --num-candidates 8 --out bon_eval
"""
import argparse
import json
import os
import pathlib
import re

from eval_bg_ab import pick_ref_from_dataset

# Must match prep_kaldi_data.WORD_RE so the judge lexicon lookups agree.
WORD_RE = re.compile(r"[а-яьѝ]+(?:-[а-яьѝ]+)*")
SPK_CHUNK = 56  # utts per pseudo-speaker (fMLLR wants minutes, nj wants >=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sentences", default="eval_ab_border.txt")
    ap.add_argument("--accentor", default="exp/accentor_ft5b/best.pt")
    ap.add_argument("--overrides", default="config/bg/bg_stress_overrides.json")
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--out", default="bon_eval")
    ap.add_argument("--kaldi-data", default="exp/kaldi_pilot/data/bon")
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--train-jsonl", default="data/bg/train.jsonl")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from accentor_model import AccentorAccenter
    from omnivoice import OmniVoice

    sentences = [l.strip() for l in open(args.sentences, encoding="utf-8")
                 if l.strip()]
    acc = AccentorAccenter(args.accentor, args.overrides)
    marked = [acc.accent_text(s) for s in sentences]

    if args.ref_audio:
        ref_audio, ref_text = args.ref_audio, args.ref_text
    else:
        ref_audio, ref_text = pick_ref_from_dataset(args.train_jsonl)

    out = pathlib.Path(args.out)
    wavdir = out / "wavs"
    wavdir.mkdir(parents=True, exist_ok=True)
    with open(out / "sentences.txt", "w", encoding="utf-8") as f:
        for i, (s, m) in enumerate(zip(sentences, marked)):
            f.write(f"{i:02d}\t{s}\t{m}\n")
    print(f"Reference: {ref_audio}")
    print(f"{len(sentences)} sentences x {args.num_candidates} candidates "
          f"-> {wavdir}/")

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32
    model = OmniVoice.from_pretrained(args.model, device_map=dev, dtype=dtype)
    prompt = model.create_voice_clone_prompt(ref_audio, ref_text=ref_text)

    rows = []
    for si, (sent, m) in enumerate(zip(sentences, marked)):
        for ci in range(args.num_candidates):
            torch.manual_seed(args.seed + 1000 * ci + si)
            if dev.startswith("cuda"):
                torch.cuda.manual_seed_all(args.seed + 1000 * ci + si)
            # Serve-time decoding: default temperatures, seed is the only knob.
            audio = model.generate(
                text=m,
                voice_clone_prompt=prompt,
                num_step=args.num_step,
            )
            wav = wavdir / f"s{si:02d}c{ci:02d}.wav"
            sf.write(str(wav), audio[0], 24000)
            rows.append({
                "sent_idx": si, "cand_idx": ci,
                "wav": str(wav.resolve()),
                "sentence": sent, "marked": m,
                "text_norm": " ".join(WORD_RE.findall(sent.lower())),
            })
        print(f"  [{si:02d}] x{args.num_candidates}  {m[:60]}")

    # Kaldi data dir — pseudo-speakers like prep_kaldi_data.write_set.
    kdir = args.kaldi_data
    os.makedirs(kdir, exist_ok=True)
    for i, r in enumerate(rows):
        spk = f"bon-s{i // SPK_CHUNK:04d}"
        r["utt"] = f"{spk}-s{r['sent_idx']:02d}c{r['cand_idx']:02d}"
        r["spk"] = spk
    rows.sort(key=lambda r: r["utt"])
    spk2utt = {}
    with open(os.path.join(kdir, "wav.scp"), "w") as wf, open(
        os.path.join(kdir, "text"), "w", encoding="utf-8"
    ) as tf, open(os.path.join(kdir, "utt2spk"), "w") as uf:
        for r in rows:
            wf.write(f"{r['utt']} ffmpeg -nostdin -v error -i {r['wav']} "
                     f"-ac 1 -ar 16000 -f wav - |\n")
            tf.write(f"{r['utt']} {r['text_norm']}\n")
            uf.write(f"{r['utt']} {r['spk']}\n")
            spk2utt.setdefault(r["spk"], []).append(r["utt"])
    with open(os.path.join(kdir, "spk2utt"), "w") as f:
        for spk in sorted(spk2utt):
            f.write(spk + " " + " ".join(spk2utt[spk]) + "\n")

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} candidates -> {out}/manifest.jsonl ; "
          f"Kaldi dir {kdir} ({len(spk2utt)} pseudo-speakers)")
    print("next: bash bon_judge.sh")


if __name__ == "__main__":
    main()
