#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Prep the FLEURS BG test split as lhotse cuts + 80-dim fbank for an OUT-OF-DOMAIN ASR row.

FLEURS (`google/fleurs`, config `bg_bg`) is Bulgarian **Wikipedia sentences read aloud** —
genuinely out-of-domain vs. the parliament corpus (read register, different speakers, clean
studio audio). This downloads the test split, writes each clip to a seekable 16 kHz wav,
normalizes the reference with the SAME normalizer as the in-domain / Whisper eval
(`bg_text_norm`: bg_verbalize numbers + lowercase / strip-punct / strip-U+0301), extracts
80-dim fbank identical to `compute_fbank_bg.py`, and writes
`data/fbank/bg_cuts_fleurs.jsonl.gz`.

Because the supervision text is pre-normalized, the stock icefall WER scorer is already fair.
Decode with the epoch-30 model (note the generalized --eval-set accepts a cut path):

    python zipformer/decode.py --epoch 30 --avg 5 --use-averaged-model 1 \
        --eval-set data/fbank/bg_cuts_fleurs.jsonl.gz \
        --decoding-method modified_beam_search --exp-dir zipformer/exp \
        --bpe-model data/lang_bpe_500/bpe.model --max-duration 600

    pip install datasets soundfile
    ./venv/bin/python prep_fleurs_bg.py                 # full test split
    ./venv/bin/python prep_fleurs_bg.py --limit 50      # smoke
"""
import argparse
import itertools
import os
import pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="google/fleurs")
    ap.add_argument("--config", default="bg_bg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--audio-dir", default="data/fleurs_bg/audio")
    ap.add_argument("--out-dir", default="data/fbank")
    ap.add_argument("--num-mel-bins", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0, help="only first N examples (smoke)")
    ap.add_argument("--no-streaming", action="store_true",
                    help="download the full split instead of streaming it (streaming avoids "
                         "pulling the large TRAIN parquet — only the requested split is fetched)")
    args = ap.parse_args()

    from datasets import load_dataset
    import soundfile as sf
    from lhotse import (CutSet, Fbank, FbankConfig, LilcomChunkyWriter, MonoCut,
                        Recording, SupervisionSegment)
    from bg_text_norm import normalize

    dst = pathlib.Path(args.out_dir); dst.mkdir(parents=True, exist_ok=True)
    manifest = dst / "bg_cuts_fleurs.jsonl.gz"
    if manifest.exists():
        print(f"[skip] {manifest} exists — delete to rebuild"); return
    adir = pathlib.Path(args.audio_dir); adir.mkdir(parents=True, exist_ok=True)

    stream = not args.no_streaming
    print(f"loading {args.dataset}:{args.config} split={args.split} (streaming={stream})")
    if stream:
        # streaming fetches ONLY the requested split, incrementally — avoids the big TRAIN
        # parquet and starts producing examples immediately.
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
        examples = itertools.islice(ds, args.limit) if args.limit else ds
    else:
        ds = load_dataset(args.dataset, args.config, split=args.split)
        examples = ds.select(range(min(args.limit, len(ds)))) if args.limit else ds
        print(f"{len(ds):,} examples")

    cuts, dropped = [], 0
    for i, ex in enumerate(examples, 1):
        if i % 100 == 0:
            print(f"  {i} examples...")
        a = ex["audio"]
        arr, sr = a["array"], a["sampling_rate"]
        # reference: raw (with punctuation) -> our normalizer, so it matches training targets
        ref = normalize(ex.get("raw_transcription") or ex.get("transcription") or "")
        if not ref:
            dropped += 1
            continue
        cid = f"fleurs_{args.config}_{ex['id']}"
        wav = adir / f"{cid}.wav"
        if not wav.exists():
            sf.write(str(wav), arr, sr)                 # seekable 16 kHz mono source
        rec = Recording.from_file(wav, recording_id=cid)
        dur = rec.duration
        cuts.append(MonoCut(
            id=cid, start=0.0, duration=dur, channel=0, recording=rec,
            supervisions=[SupervisionSegment(
                id=cid, recording_id=cid, start=0.0, duration=dur,
                channel=0, text=ref, language="Bulgarian")]))
    print(f"built {len(cuts):,} cuts ({dropped} dropped for empty text)")

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
          "--decoding-method modified_beam_search (OOD row for docs/BG_ASR_BENCHMARK.md)")


if __name__ == "__main__":
    main()
