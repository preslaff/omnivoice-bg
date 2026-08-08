#!/usr/bin/env python3
# Copyright 2024 Xiaomi Corp. (authors: OmniVoice-BG)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Re-mark training-text sidecars with rechko stress marks.

The token shards keep audio tokens (`audios/*.tar`) and text (`txts/*.jsonl`)
separate, so we can add stress marks to the text WITHOUT touching audio — no
re-tokenization needed. This rewrites each `txts/*.jsonl` through `BgAccenter`
into a parallel `txts_marked/` dir and emits a `data_marked.lst` pointing the
untouched audio shards at the marked text. Originals are left intact.

Usage:
    python bg_remark_tokens.py --tokens-dir data/bg/tokens/train \
        --lexicon data/bg_lexicon/bg_stress.sqlite \
        --overrides config/bg/bg_stress_overrides.json
"""
import argparse
import json
import os

from bg_accent import BgAccenter

ACUTE = "́"


def remark_split(tokens_dir: str, accenter: BgAccenter) -> None:
    txt_dir = os.path.join(tokens_dir, "txts")
    out_dir = os.path.join(tokens_dir, "txts_marked")
    os.makedirs(out_dir, exist_ok=True)

    shards = sorted(f for f in os.listdir(txt_dir) if f.endswith(".jsonl"))
    n_lines = n_marked = 0
    for shard in shards:
        with open(os.path.join(txt_dir, shard), encoding="utf-8") as fin, \
             open(os.path.join(out_dir, shard), "w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                rec["text"] = accenter.accent_text(rec["text"])
                n_lines += 1
                if ACUTE in rec["text"]:
                    n_marked += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Rewrite data.lst so the audio column is unchanged but text -> txts_marked.
    src_lst = os.path.join(tokens_dir, "data.lst")
    dst_lst = os.path.join(tokens_dir, "data_marked.lst")
    with open(src_lst, encoding="utf-8") as fin, open(dst_lst, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line.replace(f"{os.sep}txts{os.sep}", f"{os.sep}txts_marked{os.sep}"))

    pct = 100 * n_marked / n_lines if n_lines else 0
    print(f"  {tokens_dir}: {n_lines} lines, {n_marked} marked ({pct:.1f}%) "
          f"-> {out_dir} ; {dst_lst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-dir", action="append", required=True,
                    help="token dir holding txts/ and data.lst (repeatable)")
    ap.add_argument("--lexicon", default="data/bg_lexicon/bg_stress.sqlite")
    ap.add_argument("--overrides", default="config/bg/bg_stress_overrides.json")
    args = ap.parse_args()

    accenter = BgAccenter(args.lexicon, args.overrides)
    for td in args.tokens_dir:
        remark_split(td, accenter)


if __name__ == "__main__":
    main()
