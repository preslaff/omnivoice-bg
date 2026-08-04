#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Prep the FLEURS BG test split as lhotse cuts + 80-dim fbank for an OUT-OF-DOMAIN ASR row.

FLEURS (`google/fleurs`, config `bg_bg`) is Bulgarian **Wikipedia sentences read aloud** —
genuinely out-of-domain vs. the parliament corpus (read register, different speakers, clean
studio audio).

WHY NOT `datasets.load_dataset`: on datasets 5.0 it hangs resolving the 102-language FLEURS
repo tree before touching any data. We bypass it — the datasets-server parquet API hands back
the single test parquet URL instantly; we download that one file (with progress) and read it
locally with pyarrow. Audio is embedded as encoded bytes; we decode with soundfile.

Pipeline: parquet -> 16 kHz wav per clip -> reference normalized by bg_text_norm (same as the
in-domain / Whisper eval) -> 80-dim fbank (== compute_fbank_bg.py) -> data/fbank/bg_cuts_fleurs.jsonl.gz.

    pip install soundfile        # pyarrow ships with datasets; urllib is stdlib
    ./venv/bin/python prep_fleurs_bg.py --limit 50      # smoke (still caches the 407MB parquet once)
    ./venv/bin/python prep_fleurs_bg.py                 # full FLEURS test
    HF_ENDPOINT=https://hf-mirror.com ./venv/bin/python prep_fleurs_bg.py   # via mirror

Then decode:
    python zipformer/decode.py --epoch 30 --avg 5 --use-averaged-model 1 \
        --eval-set data/fbank/bg_cuts_fleurs.jsonl.gz \
        --decoding-method modified_beam_search --exp-dir zipformer/exp \
        --bpe-model data/lang_bpe_500/bpe.model --max-duration 600
"""
import argparse
import io
import json
import os
import pathlib
import urllib.request


def parquet_urls(dataset, config, split):
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    api = f"{base}/api/datasets/{dataset}/parquet/{config}/{split}"
    with urllib.request.urlopen(api, timeout=30) as r:
        urls = json.load(r)
    # honor HF_ENDPOINT for the file URLs too (API returns huggingface.co absolute URLs)
    if base != "https://huggingface.co":
        urls = [u.replace("https://huggingface.co", base) for u in urls]
    return urls


def download(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [cache] {dst.name} ({dst.stat().st_size/1e6:.0f} MB)")
        return
    tmp = pathlib.Path(str(dst) + ".part")
    print(f"  downloading {dst.name} …")
    req = urllib.request.Request(url, headers={"User-Agent": "omnivoice-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:                       # explicit chunked write (robust)
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done/1e6:6.0f} / {total/1e6:.0f} MB  "
                          f"({100*done/total:4.1f}%)", end="", flush=True)
    print()
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"download produced no file at {tmp}")
    os.replace(tmp, dst)                                 # atomic finalize


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="google/fleurs")
    ap.add_argument("--config", default="bg_bg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache-dir", default="data/fleurs_bg")
    ap.add_argument("--out-dir", default="data/fbank")
    ap.add_argument("--num-mel-bins", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0, help="only first N examples (smoke)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    from lhotse import (CutSet, Fbank, FbankConfig, LilcomChunkyWriter, MonoCut,
                        Recording, SupervisionSegment)
    from bg_text_norm import normalize

    dst = pathlib.Path(args.out_dir); dst.mkdir(parents=True, exist_ok=True)
    manifest = dst / "bg_cuts_fleurs.jsonl.gz"
    if manifest.exists():
        print(f"[skip] {manifest} exists — delete to rebuild"); return
    cache = pathlib.Path(args.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    adir = cache / "audio"; adir.mkdir(exist_ok=True)

    print(f"resolving parquet for {args.dataset}:{args.config} split={args.split}")
    urls = parquet_urls(args.dataset, args.config, args.split)
    print(f"  {len(urls)} parquet shard(s)")
    local = []
    for i, u in enumerate(urls):
        p = cache / f"{args.config}-{args.split}-{i:03d}.parquet"
        download(u, p)
        local.append(p)

    cuts, dropped, n = [], 0, 0
    for fi, pth in enumerate(local):
        pf = pq.ParquetFile(str(pth))
        if fi == 0:
            print("  parquet columns:", pf.schema_arrow.names, "| rows:", pf.metadata.num_rows)
        for batch in pf.iter_batches(batch_size=64):
            for ex in batch.to_pylist():
                a = ex.get("audio") or {}
                ref = normalize(ex.get("raw_transcription") or ex.get("transcription") or "")
                b, apath = a.get("bytes"), a.get("path")
                if not ref or (b is None and not (apath and os.path.exists(apath))):
                    dropped += 1
                    continue
                arr, sr = sf.read(io.BytesIO(b)) if b is not None else sf.read(apath)
                # FLEURS 'id' is the SENTENCE index (repeats across speakers) — use the
                # per-recording 'path' (unique) so cut ids / wav names don't collide.
                stem = pathlib.Path(ex.get("path") or f"idx{n}").stem
                cid = f"fleurs_{args.config}_{stem}"
                wav = adir / f"{cid}.wav"
                if not wav.exists():
                    sf.write(str(wav), arr, sr)             # seekable 16 kHz mono source
                rec = Recording.from_file(wav, recording_id=cid)
                d = rec.duration
                cuts.append(MonoCut(
                    id=cid, start=0.0, duration=d, channel=0, recording=rec,
                    supervisions=[SupervisionSegment(
                        id=cid, recording_id=cid, start=0.0, duration=d,
                        channel=0, text=ref, language="Bulgarian")]))
                n += 1
                if n % 100 == 0:
                    print(f"  {n} clips…")
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
        if args.limit and n >= args.limit:
            break
    print(f"built {len(cuts):,} cuts ({dropped} dropped for empty text/audio)")

    extractor = Fbank(FbankConfig(num_mel_bins=args.num_mel_bins, sampling_rate=16000))
    cs = CutSet.from_cuts(cuts).resample(16000)
    print(f"extracting {args.num_mel_bins}-dim fbank with {args.jobs} workers -> {manifest.name}")
    cs = cs.compute_and_store_features(
        extractor=extractor, storage_path=str(dst / "bg_feats_fleurs"),
        num_jobs=args.jobs, storage_type=LilcomChunkyWriter)
    cs.to_file(manifest)
    hrs = sum(c.duration for c in cs) / 3600
    print(f"wrote {len(cs):,} cuts ({hrs:.1f} h) -> {manifest}")
    print("Next: decode.py --eval-set", str(manifest),
          "--decoding-method modified_beam_search  (OOD row for docs/BG_ASR_BENCHMARK.md)")


if __name__ == "__main__":
    main()
