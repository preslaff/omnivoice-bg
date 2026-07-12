#!/usr/bin/env python3
"""Ingest a SUBSET of `beleata74/bulgarian-audiobooks-tts-400h` into OmniVoice prep format.

Streams the parquet dataset with `Audio(decode=False)` (raw 24 kHz/16-bit WAV
bytes — no torchcodec, no re-encode), writes each clip as a WAV, and builds
train/dev JSONL manifests from the `transcription` column. Output is then fed to
`omnivoice.scripts.extract_audio_tokens` exactly like the existing bg data.

    ./venv/bin/python prep_bg400.py --out data/bg400 --max-hours 50

Schema of this dataset: {file_name, transcription, audio:{bytes,path}} @ 24 kHz,
single female narrator. Transcriptions are ASR (F5-TTS) — light filtering only.
"""
import argparse
import io
import json
import pathlib
import re
import sys

import soundfile as sf
from tqdm import tqdm

DATASET = "beleata74/bulgarian-audiobooks-tts-400h"
TARGET_SR = 24000
# A clip is dropped if its text is not predominantly Cyrillic (ASR junk) or too short.
CYR = re.compile(r"[А-Яа-яЁё]")


def is_clean_text(t: str) -> bool:
    t = t.strip()
    if len(t) < 8:
        return False
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if CYR.match(c))
    return cyr / len(letters) >= 0.95  # >=95% Cyrillic letters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bg400")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev-every", type=int, default=100)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=18.0)
    ap.add_argument("--max-hours", type=float, default=50.0, help="stop after this many audio-hours kept")
    ap.add_argument("--resume", action="store_true",
                    help="continue a partial ingest: skip the stream past the "
                         "highest already-written clip and APPEND to the "
                         "manifests (--max-hours then counts ADDITIONAL hours)")
    ap.add_argument("--shards-dir", default=None,
                    help="process LOCAL parquet shards (downloaded via "
                         "`hf download`) instead of streaming — immune to "
                         "network flaps. Requires --first-idx.")
    ap.add_argument("--first-idx", type=int, default=None,
                    help="global stream index of the FIRST row of the first "
                         "local shard (e.g. 174847 for train-00116, verified "
                         "from parquet footers)")
    args = ap.parse_args()

    from datasets import Audio, load_dataset

    out = pathlib.Path(args.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    # Resume: continue past the clips already on disk. The id is bg4_<idx>, so
    # the highest existing index tells us where to restart the stream. (HF
    # streaming .skip() still reads — not decodes — the skipped rows, so the
    # network is re-touched but no clip is re-written or re-listed.)
    start_idx, mode = 0, "w"
    if args.resume:
        existing = sorted(int(p.stem.split("_")[1])
                          for p in clips.glob("bg4_*.wav"))
        if existing:
            start_idx, mode = existing[-1] + 1, "a"
            print(f"resume: {len(existing):,} clips on disk; skipping stream "
                  f"to index {start_idx} and appending manifests", flush=True)

    print(f"Streaming {args.dataset}:{args.split} -> {out} "
          f"(target +{args.max_hours}h from idx {start_idx})", flush=True)
    # Shard-aware fast-forward: HF streaming .skip() re-READS every skipped row
    # over the network (~60 GB for a deep resume). Parquet footers give
    # per-shard row counts for a few KB each, so jump straight to the shard
    # containing start_idx and skip only within it. Global idx (and therefore
    # the bg4_<idx> ids and the idx%dev_every split) is preserved exactly.
    ds = None
    if args.shards_dir:
        # Fully offline path: local parquet shards + a known global offset.
        files = sorted(str(p) for p in pathlib.Path(args.shards_dir)
                       .rglob("*.parquet"))
        if not files:
            sys.exit(f"no parquet files under {args.shards_dir}")
        if args.first_idx is None:
            sys.exit("--shards-dir requires --first-idx (global index of the "
                     "first row of the first local shard)")
        local_skip = start_idx - args.first_idx
        if local_skip < 0:
            sys.exit(f"start_idx {start_idx} < first-idx {args.first_idx}: "
                     f"the local shards begin AFTER the resume point")
        print(f"local shards: {len(files)} files, skipping {local_skip} rows "
              f"(disk-speed)", flush=True)
        ds = load_dataset("parquet", data_files=files, split="train",
                          streaming=True)
        if local_skip:
            ds = ds.skip(local_skip)
    elif start_idx:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()
            pat = f"datasets/{args.dataset}/data/{args.split}-*.parquet"
            shards = sorted(fs.glob(pat))
            if not shards:
                raise RuntimeError(f"no parquet shards match {pat}")
            remaining, first = start_idx, None
            for i, sp in enumerate(shards):
                n = pq.ParquetFile(fs.open(sp)).metadata.num_rows
                if remaining < n:
                    first = i
                    break
                remaining -= n
            if first is None:
                print("resume: start_idx is past the end of the dataset — done")
                return
            print(f"resume: fast-forward to shard {first + 1}/{len(shards)} "
                  f"(+{remaining} rows within it)", flush=True)
            ds = load_dataset("parquet",
                              data_files=[f"hf://{s}" for s in shards[first:]],
                              split="train", streaming=True)
            if remaining:
                ds = ds.skip(remaining)
        except Exception as e:
            print(f"resume: shard fast-forward unavailable ({e}); "
                  f"falling back to full-stream skip", flush=True)
            ds = None
    if ds is None:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        if start_idx:
            ds = ds.skip(start_idx)
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass  # raw parquet already yields {bytes, path} dicts — same access path

    train_f = open(out / "train.jsonl", mode, encoding="utf-8")
    dev_f = open(out / "dev.jsonl", mode, encoding="utf-8")
    n_train = n_dev = n_skip = 0
    kept_sec = 0.0
    max_sec = args.max_hours * 3600

    bar = tqdm(enumerate(ds, start=start_idx), unit="clip", desc="ingest",
               mininterval=0.5)
    for idx, row in bar:
        if kept_sec >= max_sec:
            break
        text = (row.get("transcription") or "").strip()
        if not is_clean_text(text):
            n_skip += 1
            continue
        audio = row["audio"]
        wav_bytes = audio.get("bytes")
        if not wav_bytes:
            n_skip += 1
            continue
        try:
            info = sf.info(io.BytesIO(wav_bytes))
            dur = info.frames / info.samplerate
        except Exception:
            n_skip += 1
            continue
        if not (args.min_dur <= dur <= args.max_dur) or info.samplerate != TARGET_SR:
            n_skip += 1
            continue

        cid = f"bg4_{idx:06d}"
        wav_path = (clips / f"{cid}.wav").resolve()
        with open(wav_path, "wb") as fh:
            fh.write(wav_bytes)
        rec = {"id": cid, "audio_path": str(wav_path), "text": text, "language_id": "bg"}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        if args.dev_every > 0 and (idx % args.dev_every == 0):
            dev_f.write(line); n_dev += 1
        else:
            train_f.write(line); n_train += 1
        kept_sec += dur
        bar.set_postfix(h=f"{kept_sec/3600:.1f}", train=n_train, dev=n_dev, skip=n_skip)

    bar.close(); train_f.close(); dev_f.close()
    print(f"Done. kept {kept_sec/3600:.1f}h  train={n_train} dev={n_dev} skipped={n_skip}")
    print(f"Manifests: {out/'train.jsonl'}  {out/'dev.jsonl'}")


if __name__ == "__main__":
    main()
