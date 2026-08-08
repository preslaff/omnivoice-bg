#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Score the exported ONNX Zipformer (sherpa-onnx, CPU) on a frozen BG cut set.

The parity harness for the ONNX/int8 export: it answers "does the model we are about to
ship on a 2 GB VPS still score what the PyTorch model scored?". Deliberately a near-clone
of eval_whisper_bg.py — SAME cut sets, SAME bg_text_norm normalizer on both hypothesis and
reference, SAME jiwer scoring — so the numbers are comparable to the published ones. Only
the decoder is swapped.

Two gates it exists to measure (see docs/BG_ASR_BENCHMARK.md):
  1. ONNX fp32 vs the torch reference (dev greedy = 4.00). A gap here is the FEATURE
     pipeline, not quantization — sherpa-onnx's fbank defaults (low_freq=20,
     high_freq=-400, dither=0, snip_edges=False) match lhotse's, which is why this works.
  2. ONNX int8 vs ONNX fp32 — the cost of quantization alone.

Runs WITHOUT torch/lhotse/k2: cuts are parsed straight from the .jsonl.gz and audio is cut
out with ffmpeg (which also handles the 24 kHz -> 16 kHz resample the lhotse manifests
declare as a transform). Use ./venv-asr, not the training venv.

    ./venv-asr/bin/python eval_onnx_bg.py --int8 --limit 300      # quick signal
    ./venv-asr/bin/python eval_onnx_bg.py --int8                  # full dev set
"""
import argparse
import gzip
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import sherpa_onnx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bg_text_norm import normalize            # THE scoring authority — never inline a copy

REPO = pathlib.Path(__file__).resolve().parent
ICEFALL_ASR = pathlib.Path.home() / "icefall/egs/bg_parliament/ASR"


def iter_cuts(path, limit=None):
    """(cut_id, source_path, start, duration, ref_text) from a lhotse cuts manifest."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f):
            if limit and n >= limit:
                return
            c = json.loads(line)
            sup = c["supervisions"][0]
            rec = c["recording"]
            yield (c["id"], rec["sources"][0]["source"],
                   c.get("start", 0.0), c["duration"], sup.get("text", ""))


def decode_audio(src, start, duration):
    """16 kHz mono float32 for one cut. ffmpeg does the resample the manifest declares."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", src, "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace")[:200])
    return np.frombuffer(out.stdout, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuts", default=str(ICEFALL_ASR / "data/fbank/bg_cuts_dev.jsonl.gz"))
    ap.add_argument("--model-dir", default=str(ICEFALL_ASR / "zipformer/exp_gp"))
    ap.add_argument("--suffix", default="epoch-30-avg-5")
    ap.add_argument("--tokens", default=str(ICEFALL_ASR / "data/lang_bpe_500/tokens.txt"))
    ap.add_argument("--int8", action="store_true", help="use the .int8.onnx graphs")
    ap.add_argument("--decoding", default="greedy_search",
                    choices=["greedy_search", "modified_beam_search"])
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N cuts only (quick signal)")
    ap.add_argument("--out", default="", help="write per-utterance JSONL here")
    args = ap.parse_args()

    md, sfx = pathlib.Path(args.model_dir), args.suffix
    q = ".int8" if args.int8 else ""
    paths = {k: md / f"{k}-{sfx}{q}.onnx" for k in ("encoder", "decoder", "joiner")}
    for k, p in paths.items():
        if not p.exists():
            sys.exit(f"ERROR: {p} missing — run zipformer/export-onnx.py first")
    mb = sum(p.stat().st_size for p in paths.values()) / 1e6
    print(f"model: {'int8' if args.int8 else 'fp32'}  {mb:.1f} MB total  "
          f"decode={args.decoding}  threads={args.num_threads}")

    t0 = time.time()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(paths["encoder"]), decoder=str(paths["decoder"]),
        joiner=str(paths["joiner"]), tokens=args.tokens,
        num_threads=args.num_threads, sample_rate=16000, feature_dim=80,
        decoding_method=args.decoding, max_active_paths=args.beam,
        provider="cpu", debug=False,
    )
    print(f"recognizer built in {time.time()-t0:.1f}s")

    import jiwer
    refs, hyps, rows = [], [], []
    audio_s = infer_s = 0.0
    t_start = time.time()
    cuts = list(iter_cuts(args.cuts, args.limit or None))
    print(f"scoring {len(cuts):,} cuts from {pathlib.Path(args.cuts).name}\n")

    for i, (cid, src, start, dur, ref) in enumerate(cuts, 1):
        try:
            samples = decode_audio(src, start, dur)
        except Exception as exc:                      # a bad container must not kill the run
            print(f"  [skip] {cid}: {exc}")
            continue
        t1 = time.time()
        s = rec.create_stream()
        s.accept_waveform(16000, samples)
        rec.decode_stream(s)
        infer_s += time.time() - t1
        audio_s += dur

        hyp, r = normalize(s.result.text), normalize(ref)
        if not r:
            continue
        refs.append(r); hyps.append(hyp)
        rows.append({"id": cid, "ref": r, "hyp": hyp, "duration": dur})

        if i % 50 == 0 or i == len(cuts):
            el = time.time() - t_start
            wer = jiwer.process_words(refs, hyps).wer * 100
            rate = i / el
            eta = (len(cuts) - i) / rate if rate else 0
            print(f"  {i:6,}/{len(cuts):,}  WER {wer:5.2f}%  "
                  f"RTF {infer_s/max(audio_s,1e-9):.3f}  {rate:4.1f} cuts/s  "
                  f"ETA {eta/60:5.1f} min")

    m = jiwer.process_words(refs, hyps)
    n_words = m.hits + m.substitutions + m.deletions
    print(f"\n{'='*66}\n"
          f"{'int8' if args.int8 else 'fp32'} / {args.decoding} on "
          f"{pathlib.Path(args.cuts).name}  ({len(refs):,} utts, {n_words:,} ref words)\n"
          f"  WER  {m.wer*100:.2f}%   "
          f"[{m.insertions} ins, {m.deletions} del, {m.substitutions} sub]\n"
          f"  RTF  {infer_s/max(audio_s,1e-9):.3f}  "
          f"({audio_s/3600:.2f} h audio, {infer_s/60:.1f} min inference, "
          f"{args.num_threads} threads)\n{'='*66}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"per-utterance -> {args.out}")


if __name__ == "__main__":
    main()
