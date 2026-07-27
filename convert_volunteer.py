#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Turn volunteer recordings (from omnivoice.cli.collect) into a marked training
manifest for the ft9-multivoice imperative fine-tune.

The collector (`omnivoice/cli/collect.py`) drops browser recordings under
``data/bg/imperatives_volunteer/`` as webm/ogg blobs plus an append-only
``submissions.jsonl`` (one row per submission, with the ground-truth *marked*
text). This script is the deterministic CPU half of ingestion:

    submissions.jsonl (accepted)  ->  decode each clip to 24 kHz mono
      ->  full QC (RMS floor + clipping + duration)  ->  peak-normalise
      ->  wavs/<speaker>__<prompt>.wav  +  train.jsonl (marked, sparse-style)

Then tokenise on a GPU (the command is printed at the end — run by hand, no
``--brighten``: volunteer speech is natural, not the dark de-essed corpus).

Why many voices: the ft7/ft8 ceiling was that imperative stress is learned
voice-specifically. train.jsonl here spans every volunteer, so the fine-tune
sees прави́/ходи́/купи́ across many timbres and can abstract a voice-independent
text→stress rule (see docs/ACCENTOR_REFRESH_PLAN.md / the imperative memory).

Usage:
    ./venv/bin/python convert_volunteer.py                 # default dirs
    ./venv/bin/python convert_volunteer.py --min-rms 0.01 --min-sec 0.4
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 24000
ACUTE = "́"


def _decode_24k_mono(src: pathlib.Path):
    """ffmpeg-decode any container to 24 kHz mono float32, in memory. Returns the
    waveform or None if the clip is undecodable."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(src),
             "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
            capture_output=True, timeout=60,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        return np.frombuffer(proc.stdout, dtype="<f4").copy()
    except (subprocess.SubprocessError, ValueError):
        return None


def _qc(wav, min_sec, max_sec, min_rms, clip_frac):
    """Return (ok, reason). Rejects silence/too-quiet, clipping, bad duration."""
    dur = len(wav) / SR
    if dur < min_sec:
        return False, f"too short ({dur:.2f}s)"
    if dur > max_sec:
        return False, f"too long ({dur:.2f}s)"
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak == 0.0:
        return False, "silent"
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < min_rms:
        return False, f"too quiet (rms {rms:.4f})"
    clipped = float(np.mean(np.abs(wav) >= 0.999))
    if clipped > clip_frac:
        return False, f"clipped ({clipped*100:.1f}% of samples)"
    return True, ""


def _peak_normalize(wav, target_dbfs=-1.0):
    peak = float(np.max(np.abs(wav)))
    if peak <= 0:
        return wav
    return wav * (10 ** (target_dbfs / 20.0) / peak)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/bg/imperatives_volunteer",
                    help="collector output dir (holds submissions.jsonl + raw/)")
    ap.add_argument("--out-jsonl", default=None,
                    help="manifest path (default: <root>/train.jsonl)")
    ap.add_argument("--wav-dir", default=None,
                    help="normalized wav dir (default: <root>/wavs)")
    ap.add_argument("--min-sec", type=float, default=0.4)
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--min-rms", type=float, default=0.01,
                    help="reject clips quieter than this RMS (silence/hot-mic-off)")
    ap.add_argument("--clip-frac", type=float, default=0.01,
                    help="reject clips with more than this fraction of |x|>=0.999")
    ap.add_argument("--min-per-verb", type=int, default=0,
                    help="warn if a verb has fewer than this many accepted clips")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    subs_path = root / "submissions.jsonl"
    if not subs_path.exists():
        sys.exit(f"no submissions at {subs_path} — run the collector first")
    wav_dir = pathlib.Path(args.wav_dir) if args.wav_dir else root / "wavs"
    out_jsonl = pathlib.Path(args.out_jsonl) if args.out_jsonl else root / "train.jsonl"
    wav_dir.mkdir(parents=True, exist_ok=True)

    subs = [json.loads(l) for l in subs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    accepted = [s for s in subs if s.get("accepted") and s.get("audio_path")]
    print(f"{len(subs)} submissions, {len(accepted)} accepted at collect time")

    rows, rejects = [], []
    per_verb, per_speaker = {}, {}
    for s in accepted:
        src = pathlib.Path(s["audio_path"])
        if not src.exists():
            rejects.append((s.get("prompt_id"), "raw file missing")); continue
        wav = _decode_24k_mono(src)
        if wav is None:
            rejects.append((s["prompt_id"], "undecodable")); continue
        ok, reason = _qc(wav, args.min_sec, args.max_sec, args.min_rms, args.clip_frac)
        if not ok:
            rejects.append((s["prompt_id"], reason)); continue
        wav = _peak_normalize(wav)
        speaker = s["speaker"]
        # include the take index — a speaker may record the same prompt several
        # times (multi-take), so prompt_id alone is not unique.
        cid = f"vol_{speaker}__{s['prompt_id']}_t{s.get('take', 0)}"
        dst = wav_dir / f"{cid}.wav"
        sf.write(str(dst), wav, SR)
        rows.append({
            "id": cid,
            "audio_path": str(dst.resolve()),
            "text": s["marked"],                 # MARKED ground truth — the point
            "language_id": "bg",
            "speaker": speaker,                  # kept for balance analysis (ignored by tokenizer)
        })
        per_verb[s["verb"]] = per_verb.get(s["verb"], 0) + 1
        per_speaker[speaker] = per_speaker.get(speaker, 0) + 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if rejects:
        (root / "convert_rejects.txt").write_text(
            "\n".join(f"{pid}\t{why}" for pid, why in rejects) + "\n", encoding="utf-8")

    print(f"kept {len(rows)} clips -> {out_jsonl}  ({len(rejects)} rejected in QC)")
    print(f"speakers: {len(per_speaker)} | verbs: {len(per_verb)}")
    if args.min_per_verb:
        thin = sorted(v for v, n in per_verb.items() if n < args.min_per_verb)
        if thin:
            print(f"{len(thin)} verb(s) under {args.min_per_verb} clips (need more voices): "
                  + ", ".join(thin[:20]) + (" …" if len(thin) > 20 else ""))
    if not rows:
        return

    tok = f"{root}/tokens"
    print("\nNext (run by hand on a free GPU — natural speech, NO --brighten):")
    print(f"  mkdir -p {tok}/train/audios {tok}/train/txts")
    print(f"  CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m omnivoice.scripts.extract_audio_tokens \\")
    print(f"      --input_jsonl {out_jsonl} \\")
    print(f"      --tar_output_pattern {tok}/train/audios/shard-%06d.tar \\")
    print(f"      --jsonl_output_pattern {tok}/train/txts/shard-%06d.jsonl \\")
    print(f"      --tokenizer_path eustlb/higgs-audio-v2-tokenizer \\")
    print(f"      --nj_per_gpu 2 --shuffle True --skip_errors --min_length 0.4 --max_length 25.0")
    print(f"  # -> {tok}/train/data.lst  (add to config/bg/data_config_bg_ft9.json)")
    print(f"  # Optional QC before training: run the Kaldi stress-judge over")
    print(f"  #   train.jsonl (teacher_distill.py judge machinery) to drop")
    print(f"  #   mis-stressed submissions; then native spot-vet the rest.")


if __name__ == "__main__":
    main()
