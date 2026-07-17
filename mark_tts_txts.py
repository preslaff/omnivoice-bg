# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Re-mark the OmniVoice-BG TTS training text with the current Accentor.

Phase B of the end-to-end pipeline (docs/BG_PIPELINE_GUIDE.md 6f): the combined
acoustic model was trained on marks from an EARLIER Accentor that often
contradicted the audio (като́/дясна́ where the narrator says ка́то/дя́сна), so it
learned a NOISY mark->stress mapping ("stress moves but not correctly"). This
re-marks each shard's text with the current checkpoint so training marks agree
with the audio AND match what serve.py produces at inference.

Only the TEXT side is regenerated (`txts/` -> `txts_marked/`); the audio-token
tars (`audios/`) and the `data_marked.lst` manifests are untouched — same rows,
same order, same counts, just corrected marks. No re-tokenization.

    ./venv/bin/python mark_tts_txts.py --ckpt exp/accentor_ft4/best.pt \
        --overrides config/bg/bg_stress_overrides.json \
        --dirs data/bg/tokens/train data/bg/tokens/dev \
               data/bg400/tokens/train data/bg400/tokens/dev
"""
import argparse
import glob
import json
import os

from accentor_model import AccentorAccenter


def remark_dir(d, acc, field, subdir="txts_marked"):
    src = os.path.join(d, "txts")
    dst = os.path.join(d, subdir)
    os.makedirs(dst, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(src, "*.jsonl")))
    if not shards:
        print(f"  WARNING: no base shards in {src} — skipped")
        return
    n_rows = n_marked = 0
    for sp in shards:
        out = os.path.join(dst, os.path.basename(sp))
        with open(sp, encoding="utf-8") as f, \
                open(out + ".tmp", "w", encoding="utf-8") as o:
            for line in f:
                r = json.loads(line)
                t = r.get(field)
                if t:
                    m = acc.accent_text(t)
                    r[field] = m
                    n_marked += 1 if "́" in m else 0
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1
        os.replace(out + ".tmp", out)
    print(f"  {d}: {len(shards)} shards, {n_rows:,} rows re-marked "
          f"({n_marked:,} carry >=1 stress) -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="exp/accentor_ft4/best.pt")
    ap.add_argument("--overrides", default="config/bg/bg_stress_overrides.json")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="tokens dirs, each with a txts/ subdir")
    ap.add_argument("--field", default="text",
                    help="jsonl field the trainer reads (default: text)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sparse-keep", default=None,
                    help="if set, SPARSE dialect: mark only keep-list words "
                         "(+pins), write to txts_sparse/ and a data_sparse.lst "
                         "manifest — the dense txts_marked/ stays intact.")
    args = ap.parse_args()

    ov = args.overrides if os.path.exists(args.overrides) else None
    acc = AccentorAccenter(args.ckpt, ov, device="cpu", threshold=args.threshold,
                           sparse_keep=args.sparse_keep)
    subdir = "txts_sparse" if args.sparse_keep else "txts_marked"
    print(f"Accentor loaded (step {acc.step}, {len(acc.overrides)} overrides"
          + (f", SPARSE keep={len(acc.sparse_keep)}" if args.sparse_keep else "")
          + f"); re-marking field {args.field!r} -> {subdir}/")
    for d in args.dirs:
        remark_dir(d, acc, args.field, subdir)
        if args.sparse_keep:
            # sibling manifest pointing at txts_sparse (mirror data_marked.lst)
            dm = os.path.join(d, "data_marked.lst")
            ds = os.path.join(d, "data_sparse.lst")
            if os.path.exists(dm):
                with open(dm) as f, open(ds, "w") as o:
                    o.write(f.read().replace("/txts_marked/", "/txts_sparse/"))
                print(f"    manifest -> {ds}")
    print("done." + ("" if args.sparse_keep else
          " data_marked.lst manifests unchanged; re-fine-tune next."))


if __name__ == "__main__":
    main()
