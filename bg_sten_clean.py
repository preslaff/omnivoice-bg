#!/usr/bin/env python3
# Copyright 2024 Xiaomi Corp. (authors: OmniVoice-BG)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Clean Bulgarian Parliament stenographic transcript HTML into spoken text + turns.

Pure functions, no I/O — unit-testable on a saved `Pl_Sten_body` sample. Consumed by
`prep_bg_parliament.py`. See docs/BG_PARLIAMENT_INGESTION.md.

STENOGRAPHIC != VERBATIM: the record is lightly edited (disfluencies removed, grammar
tidied), so downstream forced alignment must tolerate insertions/deletions and must NOT
assume exact word-for-word correspondence with the audio.

Input shape (from `GET /api/v1/pl-sten/{ID}` -> Pl_Sten_body), e.g.:

    Председателствали: председателят Цецка Цачева ...<br />
    Секретари: Пламен Нунев и Милена Христова<br />
    ПРЕДСЕДАТЕЛ ЦЕЦКА ЦАЧЕВА (звъни): Налице е кворум. Откривам ...<br />
    ПРОЕКТ ЗА РЕШЕНИЕ ОТНОСНО ...<br />          # all-caps agenda heading (no colon)
"""
import html
import re

STENOGRAPHIC_NOT_VERBATIM = (
    "Parliament stenograms are lightly edited, not verbatim — align with tolerance "
    "for insertions/deletions; do not assume exact word-for-word audio correspondence."
)

# Procedural header lines (captured as session metadata, dropped from spoken text).
_HEADER_RE = re.compile(r"^\s*(Председателствал[аи]|Секретари|Секретар)\s*:", re.IGNORECASE)

# Roles that may prefix a speaker name in a label.
_ROLES = (
    "ЗАМЕСТНИК-ПРЕДСЕДАТЕЛ", "ПРЕДСЕДАТЕЛСТВАЩ", "ПРЕДСЕДАТЕЛ",
    "МИНИСТЪР-ПРЕДСЕДАТЕЛ", "ЗАМЕСТНИК МИНИСТЪР-ПРЕДСЕДАТЕЛ", "МИНИСТЪР",
    "ДОКЛАДЧИК", "ВОДЕЩ",
)
_ROLE_RE = re.compile(r"^(" + "|".join(_ROLES) + r")\b\s*(.*)$")

# A line-leading speaker label: an ALL-CAPS Cyrillic run, an optional (stage direction),
# then a colon and the utterance. Capping length/words keeps all-caps AGENDA HEADINGS
# (which are long and end without a colon) from being mistaken for speakers.
_LABEL_RE = re.compile(
    r"^\s*"
    r"([А-ЯЁ][А-ЯЁ́ \-\.]*?)"      # 1: the all-caps label (role + name)
    r"\s*(\([^)]*\))?"                   # 2: optional stage direction e.g. (звъни)
    r"\s*:\s*"
    r"(.*)$"                             # 3: the utterance remainder
)
_MAX_LABEL_WORDS = 6

# All-caps procedural keywords that can appear as "WORD:" but are NOT speakers
# (e.g. "Народното събрание РЕШИ:"). A single-word label matching one of these is rejected.
_PSEUDO_LABELS = {
    "РЕШИ", "РЕШЕНИЕ", "РЕШЕНИЯ", "ЗАКОН", "МОТИВИ", "ЗАБЕЛЕЖКА", "ПРИЛОЖЕНИЕ",
    "ПРОЕКТ", "ДЕКЛАРАЦИЯ", "СТАНОВИЩЕ", "ПРОТОКОЛ", "ДНЕВЕН РЕД", "ЧЛЕН",
}

# Inline stage directions / non-speech cues inside utterances, e.g. (Ръкопляскания.),
# (Шум и реплики.), (звъни). Removed from spoken text.
_PAREN_RE = re.compile(r"\([^)]*\)")
_WS_RE = re.compile(r"[ \t ]+")


def _strip_tags(s: str) -> str:
    """<br> -> newline, unescape entities, drop any remaining tags."""
    s = re.sub(r"<\s*br\s*/?\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)


def _is_label(label: str) -> bool:
    """True if `label` looks like a speaker label (all-caps, short), not a heading."""
    core = label.replace("́", "").strip(" .-")
    if not core:
        return False
    if len(core.split()) > _MAX_LABEL_WORDS:
        return False
    if core.upper() in _PSEUDO_LABELS:  # procedural keyword, not a person
        return False
    # All cased letters must be uppercase (names/roles are ALL CAPS here).
    letters = [c for c in core if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _split_role(label: str):
    """Split a label into (role, speaker). role is '' if none of the known roles."""
    label = label.strip(" .-")
    m = _ROLE_RE.match(label)
    if m and m.group(2).strip():
        return m.group(1), m.group(2).strip(" .-")
    if m:  # role with no trailing name (rare)
        return m.group(1), ""
    return "", label


def _clean_utterance(text: str) -> str:
    """Strip inline stage directions and normalize whitespace within an utterance."""
    text = _PAREN_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def clean_body(body_html: str):
    """Clean a `Pl_Sten_body` HTML string.

    Returns (full_text, turns, meta):
      full_text : str   — spoken text only, speaker labels removed, turns joined by '\n'
      turns     : list  — [{"speaker", "role", "text"}], one per speaker turn
      meta      : dict  — {"headers": [procedural header lines]}
    """
    text = _strip_tags(body_html)
    lines = [ln.strip() for ln in text.split("\n")]

    headers = []
    turns = []          # list of {"speaker","role","text"}
    cur = None          # current turn dict
    buf = []            # current turn's text lines

    def flush():
        nonlocal cur, buf
        if cur is not None:
            joined = _clean_utterance(" ".join(buf))
            if joined:
                cur["text"] = joined
                turns.append(cur)
        cur, buf = None, []

    for ln in lines:
        if not ln:
            continue
        if _HEADER_RE.match(ln):
            headers.append(_clean_utterance(ln))
            continue
        m = _LABEL_RE.match(ln)
        if m and _is_label(m.group(1)):
            flush()
            role, speaker = _split_role(m.group(1))
            cur = {"speaker": speaker, "role": role, "text": ""}
            buf = [m.group(3)] if m.group(3) else []
        else:
            # Continuation line (utterance text or an agenda heading read into record).
            if cur is None:  # text before the first speaker label — keep, unattributed.
                cur = {"speaker": "", "role": "", "text": ""}
            buf.append(ln)
    flush()

    full_text = "\n".join(t["text"] for t in turns if t["text"])
    return full_text, turns, {"headers": headers}


if __name__ == "__main__":  # offline self-test: python bg_sten_clean.py body.html
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python bg_sten_clean.py <Pl_Sten_body.html>")
        raise SystemExit(2)
    raw = open(path, encoding="utf-8").read()
    full, turns, meta = clean_body(raw)
    print(f"raw chars       : {len(raw)}")
    print(f"clean text chars: {len(full)}")
    print(f"turns           : {len(turns)}")
    print(f"headers         : {meta['headers']}")
    speakers = {}
    for t in turns:
        speakers[t["speaker"]] = speakers.get(t["speaker"], 0) + 1
    top = sorted(speakers.items(), key=lambda kv: -kv[1])[:8]
    print("top speakers    :", json.dumps(top, ensure_ascii=False))
    print("--- first 3 turns ---")
    for t in turns[:3]:
        print(f"  [{t['role']}|{t['speaker']}] {t['text'][:100]}")
    print("--- full_text head ---")
    print(full[:400])
