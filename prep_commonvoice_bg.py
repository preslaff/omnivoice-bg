#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Prep Common Voice BG test as lhotse cuts + 80-dim fbank — a (proxy) NEUTRAL both-OOD row.

Source: `fixie-ai/common_voice_17_0` (public parquet mirror of Mozilla Common Voice 17),
config `bg`, split `test` — crowdsourced read sentences, 32 kHz MP3, ~3.2k utts / 8 shards
(~112 MB). Out-of-domain for our parliament model.

CAVEAT — proxy, not truly neutral: Common Voice is a large public dataset that Whisper very
likely trained on, so this is NOT unseen by Whisper. Read it as "OOD for us, ~seen by Whisper"
(so it still favors Whisper). A self-recorded fresh set is the only clean both-OOD test; this
is the quick approximation.

Same `bg_text_norm` normalizer as the in-domain / FLEURS / Whisper eval; reuses
`prep_fleurs_bg`'s direct-parquet download (bypasses the datasets 5.0 resolution hang).

    pip install soundfile          # libsndfile decodes Common Voice MP3
    ./venv/bin/python prep_commonvoice_bg.py             # -> data/fbank/bg_cuts_cv.jsonl.gz
    ./venv/bin/python prep_commonvoice_bg.py --limit 50  # smoke

Decode both models on the SAME cuts:
    zipformer/decode.py --eval-set data/fbank/bg_cuts_cv.jsonl.gz --decoding-method greedy_search ...
    eval_whisper_bg.py  --cuts     data/fbank/bg_cuts_cv.jsonl.gz --model models/faster-whisper-large-v3
"""
import argparse
import io
import os
import pathlib

from prep_fleurs_bg import parquet_urls, download   # shared direct-parquet helpers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fixie-ai/common_voice_17_0")
    ap.add_argument("--config", default="bg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache-dir", default="data/commonvoice_bg")
    ap.add_argument("--out-dir", default="data/fbank")
    ap.add_argument("--out-name", default="bg_cuts_cv")
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
    manifest = dst / f"{args.out_name}.jsonl.gz"
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
            print("  parquet columns:", pf.schema_arrow.names)
        for batch in pf.iter_batches(batch_size=64):
            for ex in batch.to_pylist():
                a = ex.get("audio") or {}
                ref = normalize(ex.get("sentence") or "")      # CV transcript field
                b, apath = a.get("bytes"), a.get("path")
                if not ref or (b is None and not (apath and os.path.exists(apath))):
                    dropped += 1
                    continue
                arr, sr = sf.read(io.BytesIO(b)) if b is not None else sf.read(apath)
                stem = pathlib.Path(ex.get("path") or f"idx{n}").stem
                cid = f"cv_{stem}"
                wav = adir / f"{cid}.wav"
                if not wav.exists():
                    sf.write(str(wav), arr, sr)                 # seekable source (resampled later)
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
    feats = args.out_name.replace("bg_cuts_", "bg_feats_")
    cs = CutSet.from_cuts(cuts).resample(16000)                 # CV is 32 kHz -> 16 kHz
    print(f"extracting {args.num_mel_bins}-dim fbank with {args.jobs} workers -> {manifest.name}")
    cs = cs.compute_and_store_features(
        extractor=extractor, storage_path=str(dst / feats),
        num_jobs=args.jobs, storage_type=LilcomChunkyWriter)
    cs.to_file(manifest)
    hrs = sum(c.duration for c in cs) / 3600
    print(f"wrote {len(cs):,} cuts ({hrs:.1f} h) -> {manifest}")
    print("Next: decode.py --eval-set", str(manifest), "and eval_whisper_bg.py --cuts", str(manifest))


if __name__ == "__main__":
    main()
