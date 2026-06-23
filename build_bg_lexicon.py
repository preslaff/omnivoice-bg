#!/usr/bin/env python3
"""Build a Bulgarian surface-form stress lexicon from the rechko MySQL dump —
no database import required.

Phase 1 — `word` table: ~119k lemmas with `name_stressed` (backtick after the
stressed vowel, e.g. `буто`н`). Backtick → combining acute U+0301 gives `буто́н`,
the form OmniVoice respects.

Phase 2 — `derivative_form` table: ~4M inflected forms (their own `name_stressed`
is empty) each linked to a base lemma via `base_word_id`. We PROJECT the lemma's
stress onto each form by vowel ordinal: lemma stressed on its N-th vowel → mark
the form's N-th vowel. Exact for fixed-stress paradigms (буто́н→буто́на,
поле́→поле́то); WRONG for mobile-stress cells (нати́сна→натисне́те), which are
flagged by their grammatical `description` for the curated override layer (step 3).

Only forms that receive a mark are stored (monosyllables / unmarkable forms are
absent → the front-end leaves them as-is, which is correct).

    ./venv/bin/python build_bg_lexicon.py \
        --dump ../rechko/db.sql --out data/bg_lexicon/bg_stress.sqlite
"""

import argparse
import mmap
import pathlib
import re
import sqlite3

ACUTE = "́"  # U+0301
VOWELS = set("аеиоуъюяАЕИОУЪЮЯ")  # й is a consonant for stress purposes

WORD_ROW = re.compile(r"\((\d+),'([^']*)',(?:'([^']*)'|NULL)")
# derivative_form: (id,'name',stressed,broken,condensed,description,is_infinitive,base_word_id,...)
DERIV_ROW = re.compile(
    r"\(\d+,'([^']*)',(?:'[^']*'|NULL),(?:'[^']*'|NULL),(?:'[^']*'|NULL),"
    r"('[^']*'|NULL),\d+,(\d+|NULL)"
)
# grammatical labels whose stress commonly shifts off the stem (projection unreliable)
MOBILE_HINTS = ("повелително", "пълен член", "мин. св", "аорист", "мн.ч.")


def table_region(mm, name: bytes):
    tag = b"INSERT INTO `" + name + b"` VALUES"
    s = mm.find(tag)
    if s < 0:
        raise SystemExit(f"no INSERT for `{name.decode()}`")
    pos = s + len(tag)
    head = b"INSERT INTO `"
    while True:
        nxt = mm.find(head, pos)
        if nxt < 0:
            return s, len(mm)
        other = mm[nxt + len(head):].split(b"`", 1)[0]
        if other != name:
            return s, nxt
        pos = nxt + len(head)


def stressed_ordinal(acute: str):
    """1-based index, among vowels, of the stressed vowel; None if unmarked."""
    n = 0
    for i, c in enumerate(acute):
        if c in VOWELS:
            n += 1
            if i + 1 < len(acute) and acute[i + 1] == ACUTE:
                return n
    return None


def project(form: str, ordinal: int):
    """Place the acute after the form's `ordinal`-th vowel; None if too few vowels."""
    n = 0
    out = []
    for c in form:
        out.append(c)
        if c in VOWELS:
            n += 1
            if n == ordinal:
                out.append(ACUTE)
    return "".join(out) if n >= ordinal else None


def stress_penult(form: str):
    """Acute on the penultimate vowel — the rule for 2pl imperative -ете/-йте."""
    idx = [i for i, c in enumerate(form) if c in VOWELS]
    if len(idx) < 2:
        return None
    p = idx[-2]
    return form[: p + 1] + ACUTE + form[p + 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="../rechko/db.sql")
    ap.add_argument("--out", default="data/bg_lexicon/bg_stress.sqlite")
    args = ap.parse_args()

    f = open(args.dump, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(str(out))
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA journal_mode=MEMORY")
    con.execute(
        "CREATE TABLE stress (name TEXT, stressed TEXT, src TEXT, "
        "mobile INT DEFAULT 0, UNIQUE(name, stressed))"
    )

    # ---- Phase 1: lemmas ----
    s, e = table_region(mm, b"word")
    region = mm[s:e].decode("utf-8", "replace")
    print(f"word region {(e-s)/1e6:.1f} MB ...")
    id_to_acute = {}     # word.id -> acute lemma (only marked lemmas)
    lemma_rows = []
    lemmas_marked = 0
    for m in WORD_ROW.finditer(region):
        wid, name, stressed = m.group(1), m.group(2), m.group(3)
        if not name or stressed is None:
            continue
        acute = stressed.replace("`", ACUTE)
        if acute.replace(ACUTE, "") != name:
            continue
        if ACUTE in acute:
            id_to_acute[int(wid)] = acute.lower()
            lemma_rows.append((name.lower(), acute.lower(), "lemma", 0))
            lemmas_marked += 1
    con.executemany("INSERT OR IGNORE INTO stress VALUES (?,?,?,?)", lemma_rows)
    con.commit()
    print(f"  lemmas with stress: {lemmas_marked}")
    del region

    # ---- Phase 2: inflected forms (projected) ----
    s, e = table_region(mm, b"derivative_form")
    region = mm[s:e].decode("utf-8", "replace")
    print(f"derivative_form region {(e-s)/1e6:.1f} MB ...")
    proj = skip_nobase = skip_nolemma = skip_vowel = mobile = 0
    batch = []
    impv = 0
    for m in DERIV_ROW.finditer(region):
        name, desc, bid = m.group(1), m.group(2), m.group(3)
        if not name:
            skip_nobase += 1
            continue
        low = name.lower()
        # 2pl imperative (-ете/-йте), labelled повелително OR сег.вр. 2л.: stress
        # the penult vowel — reliable, no lemma needed (overrides the projection).
        is_impv = low.endswith(("ете", "йте")) and desc and (
            "повелително" in desc or "2л." in desc
        )
        if is_impv:
            marked, src, is_mobile = stress_penult(low), "impv", 0
        else:
            if bid == "NULL":
                skip_nobase += 1
                continue
            acute = id_to_acute.get(int(bid))
            if acute is None:
                skip_nolemma += 1
                continue
            ordn = stressed_ordinal(acute)
            marked = project(low, ordn) if ordn else None
            src = "infl"
            is_mobile = 1 if (desc and any(h in desc for h in MOBILE_HINTS)) else 0
        if not marked or ACUTE not in marked:
            skip_vowel += 1
            continue
        impv += is_impv
        mobile += is_mobile
        batch.append((low, marked, src, is_mobile))
        proj += 1
        if len(batch) >= 200_000:
            con.executemany("INSERT OR IGNORE INTO stress VALUES (?,?,?,?)", batch)
            con.commit()
            batch.clear()
    if batch:
        con.executemany("INSERT OR IGNORE INTO stress VALUES (?,?,?,?)", batch)
        con.commit()
    del region

    con.execute("CREATE INDEX name_idx ON stress (name)")
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM stress").fetchone()[0]
    uniq = con.execute("SELECT COUNT(DISTINCT name) FROM stress").fetchone()[0]
    con.close()

    print(f"  inflected forms: {proj}  (imperative penult-rule: {impv}, flagged mobile: {mobile})")
    print(f"  skipped — no base_id: {skip_nobase} | base lemma unmarked: {skip_nolemma} | vowel mismatch: {skip_vowel}")
    print(f"\nlexicon rows: {total} | distinct surface forms: {uniq}")
    print(f"sqlite: {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
