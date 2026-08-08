# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure, unit-testable cleaner for chitanka.info plain-text book exports.

Structure of a chitanka txt export (verified on data/chitanka_archive, 8,225
books, ~97.5% strict match):

    <title / author / "Превод от ...:" / chitanka.info>     <- front matter
    ПЪРВА ЧАСТ | <Title>                                    <- section marker
    <prose paragraphs, blank-line separated>
    [1]                                                     <- footnote block
    <footnote text>
    ↑
    ...
    Допълнителна информация | <Title>                       <- metadata section
    Издание: ...
    <Title>                                                 <- bare title repeat
    Заслуги                                                 <- credits block
    Имате удоволствието да четете ... http://chitanka.info ...

`clean_book(text)` keeps only the prose: drops everything before the first
section marker, drops metadata sections by name, strips footnote blocks and
inline [N] references, bare-title repeats and the trailing credits block.

`split_sentences(paragraph)` is a conservative Bulgarian sentence splitter
(abbreviation-aware re-merge pass) producing the per-sentence units the
synthesis pipeline consumes.
"""
import re
from dataclasses import dataclass, field

# Section names (LHS of "NAME | Title" markers) that hold publication
# metadata, not prose.
META_SECTIONS = {
    "допълнителна информация",
    "информация за текста",
    "заслуги",
    "издание",
}

# "SECTION | Title" navigation marker emitted by the chitanka exporter.
# The pipe character does not occur in Bulgarian prose.
MARKER_RE = re.compile(r"^(.{1,120}?) \| (.{1,200})$")

FOOTNOTE_START_RE = re.compile(r"^\[\d+\]$")
INLINE_FOOTNOTE_RE = re.compile(r"\[\d+\]")

# Trailing credits block ("Заслуги") — fixed boilerplate.
CREDITS_HEAD = "заслуги"
CREDITS_PHRASES = (
    "имате удоволствието да четете",
    "моята библиотека",
    "http://chitanka.info",
    "работното ателие",
)


@dataclass
class CleanResult:
    title: str
    paragraphs: list
    stats: dict = field(default_factory=dict)


def _find_markers(lines):
    """Indices + parsed (name, title) of section-marker lines."""
    markers = []
    for i, ln in enumerate(lines):
        m = MARKER_RE.match(ln)
        if m:
            markers.append((i, m.group(1).strip(), m.group(2).strip()))
    return markers


def clean_book(text: str):
    """chitanka txt export -> CleanResult(title, prose paragraphs, stats).

    Returns None when the book does not follow the chitanka structure
    (no section markers) — caller should log and skip.
    """
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    lines = [ln.strip("﻿") for ln in lines]

    markers = _find_markers(lines)
    if not markers:
        return None

    # Book title = most common RHS across markers (robust to odd sections).
    from collections import Counter

    title = Counter(t for _, _, t in markers).most_common(1)[0][0]

    n_meta_sections = 0
    kept_lines = []
    # Walk sections: front matter (before the first marker) is dropped.
    for k, (idx, name, _t) in enumerate(markers):
        end = markers[k + 1][0] if k + 1 < len(markers) else len(lines)
        if name.lower() in META_SECTIONS:
            n_meta_sections += 1
            continue
        kept_lines.extend(lines[idx + 1 : end])

    # Drop footnote blocks:  ^[N]$ ... ^↑$   (translator notes, foreign quotes).
    out = []
    n_footnotes = 0
    in_footnote = False
    for ln in kept_lines:
        s = ln.strip()
        if in_footnote:
            if s == "↑":
                in_footnote = False
            continue
        if FOOTNOTE_START_RE.match(s):
            in_footnote = True
            n_footnotes += 1
            continue
        if s == "↑":
            continue
        out.append(ln)

    # Trailing credits block: a bare "Заслуги" line plus its boilerplate can
    # leak into the last kept section when it is not marker-delimited.
    pruned = []
    skip_credits = False
    for ln in out:
        s = ln.strip().lower()
        if s == CREDITS_HEAD:
            skip_credits = True
            continue
        if skip_credits and (not s or any(p in s for p in CREDITS_PHRASES)):
            continue
        skip_credits = False
        pruned.append(ln)

    # Paragraphs = blank-line separated runs; drop bare-title repeats and
    # inline footnote references.
    paragraphs = []
    cur = []
    for ln in pruned + [""]:
        s = INLINE_FOOTNOTE_RE.sub("", ln.strip())
        if s:
            cur.append(s)
            continue
        if cur:
            par = " ".join(cur)
            if par.strip() != title:
                paragraphs.append(par)
            cur = []

    stats = {
        "n_sections": len(markers),
        "n_meta_sections": n_meta_sections,
        "n_footnotes": n_footnotes,
        "n_paragraphs": len(paragraphs),
        "n_chars": sum(len(p) for p in paragraphs),
    }
    return CleanResult(title=title, paragraphs=paragraphs, stats=stats)


# ---------------------------------------------------------------------------
# Sentence splitting (synthesis units)
# ---------------------------------------------------------------------------

# Common Bulgarian abbreviations that end with "." mid-sentence.
_ABBREVS = (
    "г", "гг", "в", "вв", "св", "ул", "бул", "пл", "кв", "гр", "с", "о",
    "т", "т.е", "т.н", "и т.н", "напр", "проф", "акад", "инж", "доц",
    "д-р", "г-н", "г-жа", "г-ца", "др", "стр", "бр", "хил", "млн", "млрд",
    "сп", "вж", "срв", "гл", "ч", "чл", "ал", "изд", "ред", "прев", "н",
)
_ABBREV_RE = re.compile(
    r"(?:^|[\s(„«])(?:%s)\.$" % "|".join(re.escape(a) for a in _ABBREVS),
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[„«\"(—–А-ЯA-Z0-9])")


def split_sentences(paragraph: str):
    """Conservative BG sentence splitter: regex split + abbreviation re-merge."""
    parts = _SPLIT_RE.split(paragraph.strip())
    merged = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if merged and _ABBREV_RE.search(merged[-1]):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_SELF_TEST = """Заглавие




Автор
Заглавие


Превод от английски: Някой Си, 1999

chitanka.info




ПЪРВА ЧАСТ | Заглавие


Първо изречение тук. Второ изречение през 1972 г. в гр. София.


Реплика с бележка[1] под линия.


[1]
Текст на бележката.
↑


Допълнителна информация | Заглавие


Издание:
ИК „Нещо си“ — 2000


Заглавие

Заслуги
Имате удоволствието да четете тази книга благодарение на Моята библиотека и нейните всеотдайни помощници.
http://chitanka.info
Вие също можете да помогнете за обогатяването на Моята библиотека. Посетете работното ателие, за да научите повече.
"""


def _self_test():
    res = clean_book(_SELF_TEST)
    assert res is not None
    assert res.title == "Заглавие"
    assert res.paragraphs == [
        "Първо изречение тук. Второ изречение през 1972 г. в гр. София.",
        "Реплика с бележка под линия.",
    ], res.paragraphs
    assert res.stats["n_meta_sections"] == 1
    assert res.stats["n_footnotes"] == 1
    sents = split_sentences(res.paragraphs[0])
    assert sents == [
        "Първо изречение тук.",
        "Второ изречение през 1972 г. в гр. София.",
    ], sents
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
