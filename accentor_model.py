# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Accentor — explicit lexical stress model for Bulgarian (Geneva et al.,
Interspeech 2023, §4), reproduced for OmniVoice-BG.

Character-level: embedding + sinusoidal PE -> 6 FFT blocks (multi-head
self-attention + 2-layer 1D conv, FastSpeech-style) -> conv predictor -> one
logit per character; trained with BCE on VOWEL positions only. Deviations from
the paper (recorded as ablation A1 in docs/BG_PARLIAMENT_INGESTION.md):
standard convolutions instead of depthwise-separable (`separable=True` for
paper-exact), no 6.5M-param cap (we run offline, not on-device), plus a
per-character casing feature (ablation A4 — names are the dominant error
class and Cyrillic case marks them).

Inference: `mark_text(model, text)` -> text with U+0301 after the stressed
vowel of each word — exactly what the OmniVoice-BG marked-finetune front-end
consumes (drop-in replacement for the rechko lookup in serve.py, later).
"""
import json
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

ACUTE = "́"
VOWELS = set("аъоуеиюя")
# 30 BG letters + punctuation the corpus quality gate admits + space (word
# boundary). PAD=0, UNK=1.
LETTERS = "абвгдежзийклмнопрстуфхцчшщъьюя"
PUNCT = " ,;:.!?…«»„“”\"'()—–-"
VOCAB = ["<pad>", "<unk>"] + list(LETTERS) + list(PUNCT)
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
WORD_CHARS = set(LETTERS) | {"-"}


def encode_text(text):
    """-> (ids, caps) lists; ѝ folds to и, case moves to the caps feature."""
    ids, caps = [], []
    for ch in text:
        caps.append(1 if ch.isupper() else 0)
        c = ch.lower().replace("ѝ", "и").replace("ё", "е")
        ids.append(CHAR2ID.get(c, 1))
    return ids, caps


class FFTBlock(nn.Module):
    def __init__(self, d=256, heads=4, head_dim=128, conv_hidden=1024,
                 kernel=(9, 1), dropout=0.1, separable=False):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.qkv = nn.Linear(d, inner * 3)
        self.attn_out = nn.Linear(inner, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

        def conv(cin, cout, k):
            if separable and k > 1:
                return nn.Sequential(
                    nn.Conv1d(cin, cin, k, padding=k // 2, groups=cin),
                    nn.Conv1d(cin, cout, 1))
            return nn.Conv1d(cin, cout, k, padding=k // 2)

        self.conv1 = conv(d, conv_hidden, kernel[0])
        self.conv2 = conv(conv_hidden, d, kernel[1])

    def forward(self, x, pad_mask):
        # x: (B,T,d); pad_mask: (B,T) True at real tokens
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        attn_mask = pad_mask[:, None, None, :]  # broadcast over heads/queries
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a.transpose(1, 2).reshape(B, T, -1)
        x = self.ln1(x + self.drop(self.attn_out(a)))
        c = x.transpose(1, 2)
        c = self.conv2(F.relu(self.conv1(c))).transpose(1, 2)
        x = self.ln2(x + self.drop(c))
        return x * pad_mask.unsqueeze(-1)


class Accentor(nn.Module):
    def __init__(self, vocab_size=len(VOCAB), d=256, n_blocks=6, heads=4,
                 head_dim=128, conv_hidden=1024, kernel=(9, 1), dropout=0.1,
                 separable=False, max_len=512):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d=d, n_blocks=n_blocks,
                           heads=heads, head_dim=head_dim,
                           conv_hidden=conv_hidden, kernel=tuple(kernel),
                           dropout=dropout, separable=separable,
                           max_len=max_len)
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.caps_emb = nn.Embedding(2, d)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList(
            FFTBlock(d, heads, head_dim, conv_hidden, kernel, dropout,
                     separable) for _ in range(n_blocks))
        # Predictor (§4): two 1D convs, kernel 3, 256 channels -> scalar logit.
        self.pred_conv1 = nn.Conv1d(d, 256, 3, padding=1)
        self.pred_conv2 = nn.Conv1d(256, 256, 3, padding=1)
        self.proj = nn.Linear(256, 1)

    def forward(self, ids, caps, pad_mask):
        x = self.emb(ids) + self.caps_emb(caps) + self.pe[: ids.size(1)]
        for blk in self.blocks:
            x = blk(x, pad_mask)
        p = x.transpose(1, 2)
        p = F.relu(self.pred_conv2(F.relu(self.pred_conv1(p)))).transpose(1, 2)
        return self.proj(p).squeeze(-1)  # (B,T) logits


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_WORD_SPAN_RE = re.compile("[%s-]+" % LETTERS, re.IGNORECASE)


@torch.no_grad()
def mark_text(model, text, device="cpu", threshold=0.5):
    """One stressed vowel per word (argmax over its vowels) when confident."""
    ids, caps = encode_text(text)
    ids_t = torch.tensor([ids], device=device)
    caps_t = torch.tensor([caps], device=device)
    mask = torch.ones_like(ids_t, dtype=torch.bool)
    probs = torch.sigmoid(model(ids_t, caps_t, mask))[0].tolist()
    out, last = [], 0
    low = text.lower()
    for m in _WORD_SPAN_RE.finditer(text.replace("ѝ", "и")):
        vpos = [i for i in range(*m.span()) if low[i] in VOWELS]
        out.append(text[last : m.end()])
        last = m.end()
        if not vpos:
            continue
        best = max(vpos, key=lambda i: probs[i])
        if probs[best] >= threshold:
            w = out.pop()
            cut = best + 1 - (m.end() - len(w))  # position within this piece
            out.append(w[:cut] + ACUTE + w[cut:])
    out.append(text[last:])
    return "".join(out)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# words in already-marked text = letters + the combining acute + hyphen
_OV_WORD_RE = re.compile("[%s%sѝЍ%s-]+" % (LETTERS, LETTERS.upper(), ACUTE),
                         re.IGNORECASE)


class AccentorAccenter:
    """serve.py stress front-end: CONTEXTUAL Accentor marking, a drop-in for
    bg_accent.BgAccenter (same accent_text() interface). Runs the trained
    Accentor over the request text and returns it with U+0301 on each word's
    stressed vowel — the marking the OmniVoice-BG acoustic model was fine-tuned
    to obey. The overrides JSON (bare word -> marked word, lowercase keys) is a
    FINAL human veto layered on top of the model, same as the rechko path.

    CPU is the default: 19M params, a few ms/sentence — keeps the GPU for TTS.
    Thread-safe under FastAPI workers (eval-mode forward passes are read-only).
    """

    def __init__(self, ckpt_path, overrides_path=None, device="cpu",
                 threshold=0.5, max_len=240, sparse_keep=None,
                 sparse_final_stress=True):
        ck = torch.load(ckpt_path, map_location=device)
        cfg = dict(ck["config"])
        cfg["kernel"] = tuple(cfg["kernel"])
        self.model = Accentor(**cfg).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.step = ck.get("step")
        # The sinusoidal PE caps sequence length; the model was also trained on
        # <=250-char sentences, so mark longer text in chunks (at sentence/word
        # boundaries) to stay both in-bounds AND in-distribution.
        self.max_len = min(max_len, int(self.model.pe.size(0)) - 8)
        # Sparse dialect (E3, 2026-07-17): dense marking (77% of words) makes
        # the TTS over-articulate — mark-triggered sibilant emphasis. With a
        # sparse_keep JSON, marks survive ONLY on the keep-list words (tier-3
        # homographs + hard words) and on override-PINNED words; all other
        # marks are stripped, letting the fine-tuned prior carry the prosody.
        self.sparse_keep = None
        self.sparse_final_stress = sparse_final_stress
        if sparse_keep and os.path.exists(sparse_keep):
            with open(sparse_keep, encoding="utf-8") as f:
                self.sparse_keep = set(json.load(f)["words"])
        self.overrides = {}
        if overrides_path and os.path.exists(overrides_path):
            with open(overrides_path, encoding="utf-8") as f:
                # keys may carry trailing punctuation ("таблото.") — strip so
                # word-level matching works.
                self.overrides = {
                    k.lower().strip(".,;:!?"): v.strip(".,;:!?")
                    for k, v in json.load(f).items()
                }

    def _apply_overrides(self, marked):
        def sub(m):
            w = m.group(0)
            bare = w.replace(ACUTE, "")
            ov = self.overrides.get(bare.lower())
            if not ov:
                return w
            i = ov.find(ACUTE)
            # override wins over the model: transfer its mark onto the original
            # casing (i <= 0 means the override forces the word UNstressed).
            return bare if i <= 0 else bare[:i] + ACUTE + bare[i:]

        return _OV_WORD_RE.sub(sub, marked)

    @staticmethod
    def _final_stress_multi(w):
        """True if the acute is on the LAST vowel of a word with >=2 vowels —
        final stress on a polysyllable, where the TTS prior most often errs
        (imperatives вземи́/говори́). Monosyllables (То́й/два́) are excluded:
        their stress is unambiguous, so a mark there only over-articulates."""
        if ACUTE not in w:
            return False
        bare = w.replace(ACUTE, "").lower()
        nv = sum(c in VOWELS for c in bare)
        if nv < 2:
            return False
        seen = -1
        for c in w:
            if c == ACUTE:
                return seen == nv - 1
            if c.lower() in VOWELS:
                seen += 1
        return False

    def _apply_sparse(self, marked):
        keep = self.sparse_keep | {
            k for k, v in self.overrides.items() if ACUTE in v}

        def sub(m):
            w = m.group(0)
            bare = w.replace(ACUTE, "")
            # keep if: on the list / pinned, OR (final-stress rule) a
            # polysyllable stressed on its last vowel — catches imperatives
            # the Accentor gets right but the prior would miss (вземи́).
            if bare.lower() in keep or (self.sparse_final_stress
                                        and self._final_stress_multi(w)):
                return w
            return bare

        return _OV_WORD_RE.sub(sub, marked)

    def _mark_chunked(self, text):
        """Mark text longer than max_len by cutting at the rightmost sentence
        or word boundary within each window. mark_text preserves each piece
        exactly, so concatenating the marked pieces == original text + marks."""
        out, i, n = [], 0, len(text)
        while i < n:
            end = min(i + self.max_len, n)
            if end < n:
                cut = max(text.rfind(c, i, end) for c in ".!?…\n ")
                if cut > i:
                    end = cut + 1
            out.append(mark_text(self.model, text[i:end], device=self.device,
                                 threshold=self.threshold))
            i = end
        return "".join(out)

    def accent_text(self, text: str) -> str:
        marked = (mark_text(self.model, text, device=self.device,
                            threshold=self.threshold)
                  if len(text) <= self.max_len else self._mark_chunked(text))
        if self.overrides:
            marked = self._apply_overrides(marked)
        # Sparse LAST: strips marks off non-kept words, but keeps override-
        # PINNED words (handled inside _apply_sparse), so pins always survive.
        if self.sparse_keep is not None:
            marked = self._apply_sparse(marked)
        return marked


if __name__ == "__main__":
    m = Accentor()
    print(f"params: {n_params(m)/1e6:.1f}M (paper w/ separable: "
          f"{n_params(Accentor(separable=True))/1e6:.1f}M)")
    ids, caps = encode_text("Наляга́нето расте.")
    x = torch.tensor([ids])
    logits = m(x, torch.tensor([caps]), torch.ones_like(x, dtype=torch.bool))
    assert logits.shape == x.shape
    print("marked:", mark_text(m, "Налягането расте."))
    print("self-test OK")
