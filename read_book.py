#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Read a whole book with the shipped stack: ft6b Accentor marks the text,
the F9-refreshed OmniVoice-BG renders it — an end-to-end audiobook dogfood.

- Skips the chitanka header (everything up to the "Основен текст" line).
- Paragraph-aware chunking (~220 chars at sentence boundaries) keeps both
  the Accentor and the TTS in-distribution; longer pause between paragraphs.
- Deterministic decoding (seeded, temperatures 0) like the production server.
- RESUME-SAFE: chunk WAVs land in <out>/chunks/; rerunning skips existing.
- Writes <out>/book_marked.txt — the full marked text, for reading along
  (every stress decision the Accentor made, reviewable).
- Finalize: concatenates everything to <out>/book.wav (0.35 s between
  chunks, 0.8 s between paragraphs).

Usage:
  # smoke test (first ~2 minutes of audio) — fine alongside other GPU work:
  ./venv/bin/python read_book.py --book data/chitanka_archive/novels_txt/05249.txt --max-chunks 20
  # full book (run when the GPU is free; hours):
  ./venv/bin/python read_book.py --book data/chitanka_archive/novels_txt/05249.txt
"""
import argparse
import json
import pathlib
import re

from eval_bg_ab import pick_ref_from_dataset
from mine_homographs import split_sentences

HEADER_END = re.compile(r"Основен текст")


def load_paragraphs(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    start = 0
    for i, l in enumerate(lines):
        if HEADER_END.search(l):
            start = i + 1
            break
    paras = [l.strip() for l in lines[start:] if l.strip()]
    return paras


def chunk_paragraph(par, max_chars=220):
    """Greedy-pack sentences into chunks <= max_chars (long sentences pass
    through whole — the Accentor and TTS both self-chunk if needed)."""
    out, cur = [], ""
    for s in split_sentences(par):
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--engine", default="omnivoice",
                    choices=["omnivoice", "bgtts"],
                    help="bgtts = the 38M end-to-end engine: PLAIN text (its "
                         "own implicit stress), speaker-embedding cloning")
    ap.add_argument("--model", default="exp/omnivoice_bg_ft6b/checkpoint-3000")
    ap.add_argument("--bgtts-dir", default="models/bgtts-38m")
    ap.add_argument("--accentor", default="exp/accentor_ft6b/last.pt")
    ap.add_argument("--marking", default="dense",
                    choices=["dense", "sparse", "none"],
                    help="dense = mark every word (77%%); sparse = mark ONLY "
                         "homographs + hard words + pins (E3 dialect, the "
                         "over-marking fix); none = plain text. The 4:1 "
                         "marked:unmarked train mix makes sparse/plain "
                         "in-distribution on the existing model.")
    ap.add_argument("--sparse-keep", default="config/bg/sparse_keep.json")
    ap.add_argument("--no-verbalize", action="store_true",
                    help="skip number/date/currency expansion (bg_verbalize)")
    ap.add_argument("--no-final-stress", action="store_true",
                    help="sparse: do NOT keep final-stress polysyllable marks "
                         "(the imperative-rescue rule; on by default)")
    ap.add_argument("--overrides", default="config/bg/bg_stress_overrides.json")
    ap.add_argument("--out", default=None,
                    help="output dir (default reading/<book-stem>)")
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--train-jsonl", default="data/bg/train.jsonl")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = whole book")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--finalize-only", action="store_true",
                    help="skip synthesis; just concatenate existing chunks")
    args = ap.parse_args()

    book = pathlib.Path(args.book)
    out = pathlib.Path(args.out or f"reading/{book.stem}")
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    # --- text -> chunks (CPU); marked for omnivoice, PLAIN for bgtts ---
    paras = load_paragraphs(book)
    if args.engine == "omnivoice" and args.marking in ("dense", "sparse"):
        from accentor_model import AccentorAccenter
        acc = AccentorAccenter(
            args.accentor, args.overrides,
            sparse_keep=args.sparse_keep if args.marking == "sparse" else None,
            sparse_final_stress=not args.no_final_stress)
        mark = acc.accent_text
    else:
        mark = lambda t: t          # plain text (bgtts, or --marking none)
    if not args.no_verbalize and args.engine == "omnivoice":
        from bg_verbalize import normalize as _verb
        paras = [_verb(p) for p in paras]   # numbers -> words BEFORE marking
    chunks = []                     # (text_for_tts, paragraph_index)
    for pi, par in enumerate(paras):
        for piece in chunk_paragraph(par):
            chunks.append((mark(piece), pi))
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    with open(out / "book_marked.txt", "w", encoding="utf-8") as f:
        last_pi = -1
        for m, pi in chunks:
            f.write(("\n\n" if pi != last_pi and last_pi >= 0 else "\n") + m)
            last_pi = pi
    print(f"{len(paras)} paragraphs -> {len(chunks)} chunks "
          f"(marked text: {out/'book_marked.txt'})")

    import numpy as np
    import soundfile as sf

    if not args.finalize_only:
        import torch
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        if args.engine == "bgtts":
            from omnivoice.cli.bgtts_engine import BgTTSEngine
            eng = BgTTSEngine(args.bgtts_dir, args.ref_audio,
                              device="cuda" if dev.startswith("cuda") else "cpu")
            synth = lambda m: eng.synth(m)
        else:
            from omnivoice import OmniVoice
            if args.ref_audio:
                ref_audio, ref_text = args.ref_audio, args.ref_text
            else:
                ref_audio, ref_text = pick_ref_from_dataset(args.train_jsonl)
            dtype = torch.float16 if dev.startswith("cuda") else torch.float32
            model = OmniVoice.from_pretrained(args.model, device_map=dev, dtype=dtype)
            prompt = model.create_voice_clone_prompt(ref_audio, ref_text=ref_text)
            synth = lambda m: model.generate(text=m, voice_clone_prompt=prompt,
                                             num_step=args.num_step,
                                             position_temperature=0.0,
                                             class_temperature=0.0)[0]
        done = skip = 0
        for i, (m, _) in enumerate(chunks):
            wav = out / "chunks" / f"{i:05d}.wav"
            if wav.exists():
                skip += 1
                continue
            torch.manual_seed(args.seed + i)
            if dev.startswith("cuda"):
                torch.cuda.manual_seed_all(args.seed + i)
            sf.write(str(wav), synth(m), 24000)
            done += 1
            if done % 10 == 0:
                print(f"  {done + skip}/{len(chunks)} chunks", flush=True)
        print(f"synthesized {done}, skipped {skip} existing")

    # --- concatenate ---
    sr = 24000
    gap_chunk = np.zeros(int(0.35 * sr), dtype=np.float32)
    gap_para = np.zeros(int(0.80 * sr), dtype=np.float32)
    pieces, last_pi = [], -1
    missing = 0
    for i, (_, pi) in enumerate(chunks):
        wav = out / "chunks" / f"{i:05d}.wav"
        if not wav.exists():
            missing += 1
            continue
        if pieces:
            pieces.append(gap_para if pi != last_pi else gap_chunk)
        data, _ = sf.read(str(wav), dtype="float32")
        pieces.append(data)
        last_pi = pi
    if pieces:
        full = np.concatenate(pieces)
        sf.write(str(out / "book.wav"), full, sr)
        print(f"-> {out/'book.wav'}  ({len(full)/sr/60:.1f} min"
              + (f", {missing} chunks missing — rerun to fill)" if missing else ")"))
    else:
        print("no chunks synthesized yet")


if __name__ == "__main__":
    main()
