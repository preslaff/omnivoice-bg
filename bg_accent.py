#!/usr/bin/env python3
"""Bulgarian stress-marking front-end.

Inserts a combining acute (U+0301) on the stressed vowel of each word using the
rechko-derived lexicon (built by build_bg_lexicon.py), so OmniVoice renders
correct lexical stress. Per word, lookup order is:

    curated overrides  ->  lexicon  ->  leave unmarked (unknown / monosyllable)

rechko lemmas + fixed-stress inflections are trusted as-is. Mobile-stress
inflections (e.g. imperatives) are projections and may be wrong — correct those
in the overrides JSON. Original case and surrounding punctuation are preserved.

    from bg_accent import BgAccenter
    acc = BgAccenter("data/bg_lexicon/bg_stress.sqlite",
                     "config/bg/bg_stress_overrides.json")
    acc.accent_text("Натиснете бутона.")  # -> "Натисне́те буто́на."
"""

import json
import os
import pathlib
import re
import sqlite3
import threading

ACUTE = "́"  # U+0301
WORD_RE = re.compile(r"[Ѐ-ӿ]+")  # runs of Cyrillic letters
VOWELS = set("аеиоуъюяАЕИОУЪЮЯ")  # й is a consonant for stress purposes


class BgAccenter:
    def __init__(self, lexicon_path, overrides_path=None):
        # check_same_thread=False: FastAPI serves requests on worker threads.
        # Concurrent access is serialized via self._lock below.
        self.con = sqlite3.connect(
            f"file:{lexicon_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._lock = threading.Lock()
        self.cache = {}
        self.overrides = {}
        self.overrides_path = overrides_path
        self._ov_mtime = None
        self._load_overrides()

    def _ov_disk_mtime(self):
        try:
            return os.path.getmtime(self.overrides_path) if self.overrides_path else None
        except OSError:
            return None

    def _load_overrides(self):
        self._ov_mtime = self._ov_disk_mtime()
        if self.overrides_path and pathlib.Path(self.overrides_path).exists():
            raw = json.load(open(self.overrides_path, encoding="utf-8"))
            self.overrides = {k.lower(): v for k, v in raw.items()}
        else:
            self.overrides = {}

    def _maybe_reload(self):
        # Hot-reload overrides when the file changes (e.g. bg_stress_add.py ran).
        # Overrides are checked before the cache in _lookup, so no cache flush needed.
        if self._ov_disk_mtime() != self._ov_mtime:
            with self._lock:
                self._load_overrides()

    def _lookup(self, low: str):
        if low in self.overrides:
            return self.overrides[low]
        if low in self.cache:  # cached (incl. cached None for "not found")
            return self.cache[low]
        with self._lock:
            if low in self.cache:  # re-check after acquiring the lock
                return self.cache[low]
            rows = self.con.execute(
                "SELECT stressed, mobile FROM stress WHERE name=?", (low,)
            ).fetchall()
            val = None
            if rows:
                # prefer a rechko-reliable (non-mobile) reading over a projected one
                nonmobile = [s for s, m in rows if not m]
                val = nonmobile[0] if nonmobile else rows[0][0]
            self.cache[low] = val
            return val

    @staticmethod
    def _apply_case(marked: str, original: str) -> str:
        if original.isupper() and len(original) > 1:
            return marked.upper()
        if original[:1].isupper():
            return marked[:1].upper() + marked[1:]
        return marked

    def accent_text(self, text: str) -> str:
        self._maybe_reload()

        def repl(m):
            w = m.group(0)
            # Monosyllables (incl. clitics like и/а/о) never need a stress mark.
            if sum(c in VOWELS for c in w) < 2:
                return w
            st = self._lookup(w.lower())
            return self._apply_case(st, w) if st else w
        return WORD_RE.sub(repl, text)


if __name__ == "__main__":
    import sys
    acc = BgAccenter("data/bg_lexicon/bg_stress.sqlite",
                     "config/bg/bg_stress_overrides.json")
    for line in (sys.stdin if not sys.argv[1:] else [" ".join(sys.argv[1:])]):
        print(acc.accent_text(line.rstrip("\n")))
