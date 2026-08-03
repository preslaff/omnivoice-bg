#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Shared ASR-eval text normalizer — the SINGLE source of truth for scoring.

Applied IDENTICALLY to hypotheses and references (parity is the whole point) across every
ASR eval: in-domain test, Whisper (eval_whisper_bg.py), and OOD sets (prep_fleurs_bg.py).
Numbers are expanded with the project's own gender-aware verbalizer (bg_verbalize), then the
text is lowercased, stress-stripped (U+0301) and de-punctuated to match the training targets.
"""
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bg_verbalize import normalize as _verbalize_bg   # digits -> BG words (gender-aware)

ACUTE = "́"
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)            # keeps Cyrillic letters, digits, spaces


def normalize(s):
    """hyp/ref -> comparable normalized string (numbers verbalized, lower, no punct/stress)."""
    s = _verbalize_bg(s.lower())                       # expand numbers -> BG words (no digits remain)
    s = s.replace(ACUTE, "")
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(t for t in s.split() if t != "<unk>")
