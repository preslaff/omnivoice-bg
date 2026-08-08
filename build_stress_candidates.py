#!/usr/bin/env python3
# Copyright 2024 Xiaomi Corp. (authors: OmniVoice-BG)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""L1 of the Accentor lexicon build: turn the single-stress rechko lexicon into a
RECALL-oriented MULTI-candidate stress map.

rechko (via build_bg_lexicon.py -> bg_stress.sqlite) stores ONE canonical stress per
surface form, and lexically-distinct homographs (вълна́/въ́лна, замъ́к/за́мък, пара́/па́ра)
are absent entirely. The Accentor's decoding graph + DNN-HMM aligner pick the
acoustically-best path, so the lexicon only needs the TRUE stress to be AMONG the
candidates — we generate extra candidates with a BOUNDED policy (blanket over-generation
is infeasible: avg 5.77 vowels/form).

Policy (see docs/BG_PARLIAMENT_INGESTION.md -> "Building the canonical-stress lexicon"):
  - rechko canonical   : every form keeps its rechko stress(es)            [src=rechko]
  - 2-vowel over-gen   : 2-vowel forms get BOTH vowel positions            [src=overgen2]
                         (the замък/вълна/пара/тела class — the hotspot)
  - battery seed       : vetted lexically-distinct homographs rechko lacks [src=battery]
  - mobile inflections : mobile-flagged forms (unreliable projection) get
                         all positions (<=3 vowels) or canonical+final     [src=mobile]

Output: a new sqlite `candidates(name, mark, source, UNIQUE(name,mark))`, the artifact
L2 (G2P) consumes. Long unambiguous forms keep their single canonical mark.

    ./venv/bin/python build_stress_candidates.py \
        --src data/bg_lexicon/bg_stress.sqlite \
        --battery config/bg/homograph_battery.json \
        --out data/bg_lexicon/bg_stress_candidates.sqlite --validate
"""
import argparse
import json
import pathlib
import sqlite3
from collections import Counter

ACUTE = "́"  # U+0301 combining acute
VOWELS = set("аеиоуъюя")  # й is a consonant for stress; work lowercased


def vowel_indices(form: str):
    """Character indices of vowels in `form` (lowercased)."""
    return [i for i, c in enumerate(form) if c in VOWELS]


def mark_at(form: str, ordinal: int):
    """Place the acute after the `ordinal`-th vowel (1-based); None if too few vowels."""
    idx = vowel_indices(form)
    if ordinal < 1 or ordinal > len(idx):
        return None
    p = idx[ordinal - 1]
    return form[: p + 1] + ACUTE + form[p + 1:]


def all_positions(form: str):
    """Every single-stress marking of `form` (one per vowel)."""
    return [mark_at(form, k) for k in range(1, len(vowel_indices(form)) + 1)]


def load_battery(path):
    """surface -> set(marks) from the vetted homograph battery."""
    if not path:
        return {}
    obj = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out = {}
    for surface, readings in obj.get("homographs", {}).items():
        out[surface.lower()] = {r["mark"] for r in readings}
    return out


def load_always_final(path):
    """surface -> single final-stress mark. These are NOT homographs; forcing
    one candidate overrides rechko's mis-marks and suppresses over-generation
    (native-verified, config/bg/always_final_imperatives.json)."""
    if not path or not pathlib.Path(path).exists():
        return {}
    obj = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {k.lower(): v for k, v in obj.get("verbs", {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/bg_lexicon/bg_stress.sqlite")
    ap.add_argument("--battery", default="config/bg/homograph_battery.json")
    ap.add_argument("--always-final",
                    default="config/bg/always_final_imperatives.json",
                    help="verbs forced to a single final-stress candidate")
    ap.add_argument("--out", default="data/bg_lexicon/bg_stress_candidates.sqlite")
    ap.add_argument("--validate", action="store_true", help="report recall on the battery")
    args = ap.parse_args()

    battery = load_battery(args.battery)
    always_final = load_always_final(args.always_final)
    print(f"battery: {len(battery)} surface forms; "
          f"always-final overrides: {len(always_final)}")

    src = sqlite3.connect(f"file:{args.src}?mode=ro", uri=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(str(out))
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA journal_mode=MEMORY")
    con.execute(
        "CREATE TABLE candidates (name TEXT, mark TEXT, source TEXT, UNIQUE(name, mark))"
    )

    stats = Counter()
    forms = 0
    batch = []
    battery_used = set()

    def flush_form(name, rechko_marks, mobile):
        nonlocal forms
        if name is None:
            return
        forms += 1
        # Always-final override: NOT a homograph — force the single final
        # candidate, ignoring rechko + over-generation entirely.
        if name in always_final:
            batch.append((name, always_final[name], "always_final"))
            stats["always_final"] += 1
            return
        cands = {}  # mark -> source (first writer wins; rechko has priority)
        for m in rechko_marks:
            cands.setdefault(m, "rechko")
        vidx = vowel_indices(name)
        nv = len(vidx)
        # battery seed (lexically-distinct homographs rechko lacks)
        if name in battery:
            battery_used.add(name)
            for m in battery[name]:
                cands.setdefault(m, "battery")
        # 2-vowel over-generation: the homograph hotspot
        if nv == 2:
            for m in all_positions(name):
                if m:
                    cands.setdefault(m, "overgen2")
        # mobile-stress inflections: rechko projection unreliable -> widen, but ONLY for
        # short forms (<=3 vowels: plurals like тела́, aorist, definite). On long forms the
        # primary stress stays on the stem (canonical is right), so widening = pure noise.
        if mobile and nv <= 3:
            for m in all_positions(name):
                if m:
                    cands.setdefault(m, "mobile")
        for m, s in cands.items():
            batch.append((name, m, s))
            stats[s] += 1
        stats["_multi"] += 1 if len(cands) > 1 else 0

    cur_name = None
    cur_marks = []
    cur_mobile = 0
    for name, stressed, mobile in src.execute(
        "SELECT name, stressed, mobile FROM stress ORDER BY name"
    ):
        if name != cur_name:
            flush_form(cur_name, cur_marks, cur_mobile)
            cur_name, cur_marks, cur_mobile = name, [], 0
            if len(batch) >= 200_000:
                con.executemany("INSERT OR IGNORE INTO candidates VALUES (?,?,?)", batch)
                con.commit()
                batch.clear()
        cur_marks.append(stressed)
        cur_mobile |= int(mobile or 0)
    flush_form(cur_name, cur_marks, cur_mobile)
    if batch:
        con.executemany("INSERT OR IGNORE INTO candidates VALUES (?,?,?)", batch)
        con.commit()

    con.execute("CREATE INDEX cand_name_idx ON candidates (name)")
    con.commit()

    total = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    uniq = con.execute("SELECT COUNT(DISTINCT name) FROM candidates").fetchone()[0]
    print(f"\nforms processed     : {forms}")
    print(f"candidate rows      : {total}  (distinct forms: {uniq})")
    print(f"  by source         : "
          + ", ".join(f"{k}={stats[k]}" for k in ("rechko", "overgen2", "battery", "mobile")))
    print(f"  forms with >1 cand : {stats['_multi']}  ({100*stats['_multi']/forms:.2f}%)")
    print(f"  avg cands/form     : {total/forms:.3f}")

    if args.validate:
        print("\n=== recall on battery (true stress must be AMONG candidates) ===")
        miss = 0
        for surface, marks in battery.items():
            got = {r[0] for r in con.execute(
                "SELECT mark FROM candidates WHERE name=?", (surface,))}
            for m in marks:
                ok = m in got
                if not ok:
                    miss += 1
                print(f"  {'OK ' if ok else 'MISS'} {surface} -> {m}   (cands: {sorted(got)})")
        n = sum(len(v) for v in battery.values())
        print(f"recall: {n - miss}/{n} readings present "
              f"({'PASS' if miss == 0 else 'GAPS — see MISS above'})")

    con.close()
    print(f"\nsqlite: {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
