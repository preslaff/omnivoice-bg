#!/usr/bin/env python3
"""Ingest the HF dataset `raditotev/bg-audiobooks-tts` into OmniVoice's prep format.

Streams the parquet dataset, writes each clip as a 24 kHz mono WAV, and builds
train/dev JSONL manifests using the TTS-ready `text_normalized` column. The
JSONL is then fed to `omnivoice.scripts.extract_audio_tokens` (Phase 3) exactly
like any locally-prepared data — see BG_FINETUNE_PLAN.md.

    ./venv/bin/pip install datasets        # one-time
    ./venv/bin/python prep_bg_hf.py --out data/bg

Output:
    data/bg/clips/<clip_id>.wav
    data/bg/train.jsonl
    data/bg/dev.jsonl

Notes:
- Uses `text_normalized` (numbers/dates expanded); falls back to `text`.
- Single speaker (Plamen Sivov), 24 kHz mono — no resampling needed.
- `--dev-every N` holds out every Nth clip for the dev split (deterministic).
- `--limit N` ingests only the first N clips (quick smoke test).
"""

import argparse
import json
import pathlib

from tqdm import tqdm

DATASET = "raditotev/bg-audiobooks-tts"
TARGET_SR = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bg", help="output root directory")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev-every", type=int, default=20, help="hold out every Nth clip for dev")
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from datasets import Audio, load_dataset

    out = pathlib.Path(args.out)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {args.dataset}:{args.split} -> {out}", flush=True)
    print("(first run downloads the parquet shards — the bar starts once rows arrive)", flush=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    # Grab raw WAV bytes instead of decoding (no torchcodec needed; also avoids
    # any re-encode — the stored 24 kHz/16-bit audio is written through verbatim).
    ds = ds.cast_column("audio", Audio(decode=False))

    train_f = open(out / "train.jsonl", "w", encoding="utf-8")
    dev_f = open(out / "dev.jsonl", "w", encoding="utf-8")
    n_train = n_dev = n_skip = 0

    total = args.limit if args.limit else None
    bar = tqdm(enumerate(ds), total=total, unit="clip", desc="ingest", mininterval=0.5)
    for idx, row in bar:
        if args.limit and idx >= args.limit:
            break

        text = (row.get("text_normalized") or row.get("text") or "").strip()
        dur = float(row.get("duration") or 0.0)
        if not text or not (args.min_dur <= dur <= args.max_dur):
            n_skip += 1
            continue

        clip_id = row.get("clip_id", idx)
        audio = row["audio"]  # {"bytes": b"...", "path": ...} (decode=False)
        wav_bytes = audio.get("bytes")
        if wav_bytes is None and audio.get("path"):
            with open(audio["path"], "rb") as fh:
                wav_bytes = fh.read()
        if not wav_bytes:
            n_skip += 1
            continue
        wav_path = (clips / f"bg_{int(clip_id):06d}.wav").resolve()
        with open(wav_path, "wb") as fh:
            fh.write(wav_bytes)

        rec = {
            "id": f"bg_{int(clip_id):06d}",
            "audio_path": str(wav_path),
            "text": text,
            "language_id": "bg",
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"

        if args.dev_every > 0 and (idx % args.dev_every == 0):
            dev_f.write(line)
            n_dev += 1
        else:
            train_f.write(line)
            n_train += 1

        bar.set_postfix(train=n_train, dev=n_dev, skip=n_skip)

    bar.close()
    train_f.close()
    dev_f.close()
    print(f"Done. train={n_train} dev={n_dev} skipped={n_skip}")
    print(f"Manifests: {out/'train.jsonl'}  {out/'dev.jsonl'}")
    print("Next (Phase 3): python -m omnivoice.scripts.extract_audio_tokens "
          f"--input_jsonl {out/'train.jsonl'} ...")


if __name__ == "__main__":
    main()
