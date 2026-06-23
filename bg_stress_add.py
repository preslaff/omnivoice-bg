#!/usr/bin/env python3
"""Add words + correct stress to the override lexicon used by the TTS server.

Type the word with the STRESSED VOWEL IN UPPERCASE — the same way you'd point it
out by ear. Examples:

    ./venv/bin/python bg_stress_add.py натиснЕте бутОна полЕто
    ./venv/bin/python bg_stress_add.py запазЕте

The server hot-reloads the overrides file on the next request, so no restart is
needed. Overrides take priority over the rechko lexicon, so this fixes any word.

(Stress is meaningful only on poly-syllabic words; a word with one vowel is
skipped — a single syllable has nowhere else for the stress to go.)
"""

import argparse
import json
import pathlib

ACUTE = "́"  # U+0301
VOWELS = set("аеиоуъюя")


def parse_token(token: str):
    """'натиснЕте' -> ('натиснете', 'натисне́те'); None if no uppercase vowel."""
    for i, c in enumerate(token):
        if c.isupper() and c.lower() in VOWELS:
            low = token.lower()
            return low, low[: i + 1] + ACUTE + low[i + 1:]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="+",
                    help="words with the stressed vowel UPPERCASED, e.g. натиснЕте")
    ap.add_argument("--file", default="config/bg/bg_stress_overrides.json")
    args = ap.parse_args()

    p = pathlib.Path(args.file)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    for w in args.words:
        if sum(c.lower() in VOWELS for c in w) < 2:
            print(f"  SKIP {w}: monosyllable — stress mark not needed")
            continue
        key, val = parse_token(w)
        if not key:
            print(f"  SKIP {w}: mark the stressed vowel in CAPS, e.g. бутОн")
            continue
        action = "update" if key in data else "add"
        data[key] = val
        print(f"  {action}: {key} -> {val}")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(data)} overrides to {p}")


if __name__ == "__main__":
    main()
