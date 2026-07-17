#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Condition a voice-cloning reference: mono + high-pass + RMS-normalize, and
report a health check. In-context cloning imitates the reference's PRODUCTION
(compression, EQ, loudness), not just its voice — so a clean, dry, neutral
reference clones cleanly while a commercial/podcast master scratches. This
tool removes what it can (level, sub-bass tilt, stereo); it CANNOT undo
compression or already-destructive clipping — for those, use a cleaner source.

See docs/BG_VOICE_CLONING.md for the full workflow (Audacity steps + how to
re-master the OUTPUT into an ad afterward).

Usage:
  ./venv/bin/python prep_reference.py --check --in examples/reklama_home_max.wav
  ./venv/bin/python prep_reference.py --in raw.wav --out ref_clean.wav
  ./venv/bin/python prep_reference.py --in raw.wav --out ref_clean.wav --hpf 120 --target-rms 0.1
"""
import argparse
import sys

import numpy as np
import soundfile as sf


def health(x, sr):
    """Return a dict of reference-health metrics."""
    pk = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x**2)))
    crest = 20 * np.log10(pk / rms) if rms > 0 else 0.0
    clip = float((np.abs(x) >= 0.999).mean() * 100)
    n = 4096
    fr = [x[i:i + n] * np.hanning(n) for i in range(0, len(x) - n, n // 2)]
    mag = (np.mean([np.abs(np.fft.rfft(f)) for f in fr], axis=0)
           if fr else np.ones(n // 2 + 1))
    fq = np.fft.rfftfreq(n, 1 / sr)
    tot = (mag**2).sum() or 1.0
    low = 100 * (mag[fq < 300]**2).sum() / tot
    sib = 100 * (mag[(fq >= 4000) & (fq < 8000)]**2).sum() / tot
    return dict(peak=pk, rms=rms, crest=crest, clip=clip, sub300=low, sib=sib)


def verdict(h):
    """Human-readable flags — the thresholds are the ones this project measured
    (clean male speech: rms ~0.05-0.15, crest >14dB, sub-300 ~10-20%)."""
    flags = []
    if h["rms"] > 0.18:
        flags.append(f"HOT (rms {h['rms']:.2f}; auto-fixed at encode, but a "
                     f"produced master)")
    if h["crest"] < 12:
        flags.append(f"COMPRESSED (crest {h['crest']:.1f}dB — can't be undone; "
                     f"use a drier source)")
    if h["clip"] > 0.1:
        flags.append(f"CLIPPED ({h['clip']:.1f}% — destructive; needs a cleaner "
                     f"recording)")
    if h["sub300"] > 35:
        flags.append(f"BASS-HEAVY ({h['sub300']:.0f}% sub-300Hz — a high-pass "
                     f"filter helps a lot)")
    return flags or ["looks clone-friendly"]


def highpass(x, sr, fc):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, fc / (sr / 2), btype="highpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None, help="omit with --check")
    ap.add_argument("--check", action="store_true",
                    help="report health only; write nothing")
    ap.add_argument("--hpf", type=float, default=85.0,
                    help="high-pass cutoff Hz (0 disables); kills sub-bass boom")
    ap.add_argument("--target-rms", type=float, default=0.1,
                    help="normalize to this RMS (matches the model's own target)")
    ap.add_argument("--sr", type=int, default=24000)
    args = ap.parse_args()

    x, sr = sf.read(args.inp, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)                       # mono downmix

    h0 = health(x, sr)
    print(f"IN  {args.inp}  ({sr} Hz, {len(x)/sr:.1f}s)")
    print(f"  peak={h0['peak']:.3f} rms={h0['rms']:.3f} crest={h0['crest']:.1f}dB "
          f"clip={h0['clip']:.2f}% sub300={h0['sub300']:.0f}% sib={h0['sib']:.1f}%")
    for f in verdict(h0):
        print(f"  -> {f}")

    if args.check:
        return
    if not args.out:
        sys.exit("ERROR: --out required (or use --check)")

    if args.hpf > 0:
        x = highpass(x, sr, args.hpf)
    rms = float(np.sqrt(np.mean(x**2)))
    if rms > 0:
        x = x * args.target_rms / rms
    x = np.clip(x, -1.0, 1.0)
    sf.write(args.out, x, sr)

    h1 = health(x, sr)
    print(f"OUT {args.out}")
    print(f"  peak={h1['peak']:.3f} rms={h1['rms']:.3f} crest={h1['crest']:.1f}dB "
          f"sub300={h1['sub300']:.0f}% sib={h1['sib']:.1f}%")
    print("Note: level + tilt conditioned; compression is NOT removed. Still "
          "trim to a clean 12-15 s voice-only segment for best cloning.")


if __name__ == "__main__":
    main()
