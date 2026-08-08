# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Phase C3 — synthesize selected chitanka sentences with BgTTS-38M-V2.

Produces the synthetic-audiobook branch of the Accentor stress corpus: per
sentence one 16-bit 24 kHz mono FLAC + a manifest row. Because synthesis is
per-sentence, text<->audio alignment is exact BY CONSTRUCTION — the paper's
Levenshtein alignment step is unnecessary for this branch; only the later
stress-decoding pass (annotation AM, gated on the parliament crawl) remains.

The stress labels eventually decoded from this audio distill *BgTTS's* stress
knowledge (oracle/teacher role — see OMNIVOICE_BG.md). Validate the oracle on
the homograph battery before burning GPU-weeks:

    ./venv/bin/python synth_chitanka_bgtts.py \
        --bgtts-dir <dir> --ref-audio assets/voices/voice_bg_calm.wav \
        --battery config/bg/homograph_battery_sentences.json \
        --out data/chitanka_audio_battery

Bulk run (resumable — re-run the same command; done ids are skipped):

    ./venv/bin/python synth_chitanka_bgtts.py \
        --selected data/chitanka/selected.jsonl \
        --bgtts-dir <dir> --ref-audio assets/voices/voice_bg_calm.wav \
        --out data/chitanka_audio

Output layout (<out>/, gitignored under /data/):
    audio/<id>.flac        16-bit 24 kHz mono
    manifest.jsonl         {id, audio_path, text, language_id, duration, ...}
    errors.jsonl           failures + implausible-rate flags (kept out of manifest)
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from omnivoice.cli.bgtts_engine import BgTTSEngine, SAMPLE_RATE

# Plausible narrated-Bulgarian rate window; outside it the clip likely
# truncated or hallucinated (looped) — flagged, not manifested.
MIN_CHARS_PER_SEC = 5.0
MAX_CHARS_PER_SEC = 35.0


def iter_battery(path):
    """homograph_battery_sentences.json -> synthesis items.

    One clip per context sentence; the id encodes surface + expected reading
    so the listening pass is self-documenting
    (e.g. battery_03_вълна_вълна́.flac -> expect stress on the final а).
    """
    with open(path, encoding="utf-8") as f:
        battery = json.load(f)
    for i, entry in enumerate(battery["sentences"]):
        yield {"id": f"battery_{i:02d}_{entry['surface']}_{entry['expected']}",
               "text": entry["sentence"], "book": "battery", "reason": "battery"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected", default=None, help="selected.jsonl from phase C2")
    ap.add_argument("--battery", default=None,
                    help="homograph_battery_sentences.json (oracle validation mode)")
    ap.add_argument("--out", default="data/chitanka_audio")
    ap.add_argument("--bgtts-dir", required=True,
                    help="dir with BgTTS-38M-V2 code + checkpoint_inference.pt")
    ap.add_argument("--ref-audio", required=True,
                    help="fixed reference voice wav (speaker embedding)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="synthesize only N items (pilot)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="battery mode: synthesize each sentence N times with "
                         "different seeds (_r0.._rN-1) to test stress STABILITY "
                         "under sampling, not just single-shot correctness")
    args = ap.parse_args()
    if not args.selected and not args.battery:
        ap.error("need --selected or --battery")

    out_dir = os.path.abspath(args.out)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    errors_path = os.path.join(out_dir, "errors.jsonl")

    # Resume: ids already in the manifest are done.
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001 - tolerate a torn last line
                    pass

    items = []
    if args.battery:
        for it in iter_battery(args.battery):
            if args.repeats > 1:
                items.extend({**it, "id": f"{it['id']}_r{k}"}
                             for k in range(args.repeats))
            else:
                items.append(it)
    if args.selected:
        with open(args.selected, encoding="utf-8") as f:
            items.extend(json.loads(line) for line in f)
    todo = [it for it in items if it["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"items: {len(items):,} total, {len(done):,} done, {len(todo):,} to go")
    if not todo:
        return

    engine = BgTTSEngine(args.bgtts_dir, ref_audio=args.ref_audio, device=args.device)

    n_ok = n_err = 0
    t_audio = t_wall = 0.0
    with open(manifest_path, "a", encoding="utf-8") as mf, open(
        errors_path, "a", encoding="utf-8"
    ) as ef:
        pbar = tqdm(todo, desc="synth")
        for it in pbar:
            # Per-item reseed, stable across processes (NOT Python hash() —
            # that is salted per interpreter run): resumed and fresh runs
            # produce identical audio for the same id and --seed.
            torch.manual_seed(args.seed + (zlib.crc32(it["id"].encode()) & 0x7FFFFFFF))
            t0 = time.time()
            try:
                wav = engine.synth(it["text"], temperature=args.temperature)
            except Exception as e:  # noqa: BLE001 - log-and-continue crawl style
                n_err += 1
                ef.write(json.dumps({"id": it["id"], "error": repr(e)},
                                    ensure_ascii=False) + "\n")
                ef.flush()
                continue
            t_wall += time.time() - t0
            dur = len(wav) / SAMPLE_RATE
            rate = len(it["text"]) / max(dur, 1e-6)
            if not (MIN_CHARS_PER_SEC <= rate <= MAX_CHARS_PER_SEC):
                n_err += 1
                ef.write(json.dumps(
                    {"id": it["id"], "error": f"implausible rate {rate:.1f} chars/s",
                     "duration": round(dur, 2), "n_chars": len(it["text"])},
                    ensure_ascii=False) + "\n")
                ef.flush()
                continue
            path = os.path.join(audio_dir, f"{it['id']}.flac")
            tmp = path + ".tmp.flac"
            sf.write(tmp, np.clip(wav, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
            os.replace(tmp, path)
            mf.write(json.dumps({
                "id": it["id"], "audio_path": path, "text": it["text"],
                "language_id": "bg", "duration": round(dur, 3),
                "book": it.get("book"), "reason": it.get("reason"),
                "synthetic": "bgtts-38m-v2",
            }, ensure_ascii=False) + "\n")
            mf.flush()
            n_ok += 1
            t_audio += dur
            if t_wall > 0:
                pbar.set_postfix(rtf=f"{t_audio / t_wall:.1f}x",
                                 h=f"{t_audio / 3600:.2f}")

    print(f"ok={n_ok} err={n_err}  audio={t_audio/3600:.2f} h  "
          f"speed={t_audio/max(t_wall,1e-6):.1f}x realtime")
    print(f"-> {manifest_path}")


if __name__ == "__main__":
    main()
