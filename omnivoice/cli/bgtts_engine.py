# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Optional BgTTS-38M-V2 backend for the LAN server (an alternative engine).

BgTTS-38M-V2 (beleata74/BgTTS-38M-V2, Apache-2.0) is a 38M-param encoder-decoder
TTS that takes **plain** Bulgarian text (no stress front-end) and clones a voice
from a MioCodec speaker embedding. This thin wrapper loads its vendored code +
checkpoint from a directory and exposes a single `synth(text) -> np.ndarray`.

The model ships with flat imports (`from config import ...`), so its directory is
prepended to `sys.path` before importing. Modules are named config/model/
tokenizer/codec/inference — none collide with OmniVoice's `omnivoice.*` package.
"""
import os
import sys

import numpy as np
import torch

SAMPLE_RATE = 24000


class BgTTSEngine:
    def __init__(self, model_dir: str, ref_audio: str | None, device: str = "cuda"):
        self.device = device
        self.sample_rate = SAMPLE_RATE
        self._dir = os.path.abspath(model_dir)
        if self._dir not in sys.path:
            sys.path.insert(0, self._dir)

        # Vendored BgTTS modules (resolved via sys.path above).
        from model import load_for_inference
        from tokenizer import TTSTokenizer
        from codec import CodecV6
        import inference as _inf

        self._generate = _inf.generate
        self._split_text = _inf._split_text

        ckpt = os.path.join(self._dir, "checkpoint_inference.pt")
        self.model = load_for_inference(ckpt, device=device)
        self.tokenizer = TTSTokenizer()
        self.codec = CodecV6(device=device)

        self.speaker_emb = None
        if ref_audio:
            self.speaker_emb = self.codec.encode(ref_audio)["global_embedding"].to(device)

    @torch.no_grad()
    def synth(
        self,
        text: str,
        temperature: float = 0.3,
        top_k: int = 250,
        top_p: float = 0.95,
        rep_penalty: float = 1.1,
        max_new_tokens: int = 512,
    ) -> np.ndarray:
        """Plain text -> 1-D float32 waveform at 24 kHz. Requires a fixed voice."""
        if self.speaker_emb is None:
            raise RuntimeError("BgTTS engine needs a fixed --ref-audio (speaker embedding)")
        chunks = self._split_text(text, self.tokenizer, max_len=250)
        all_codes = []
        for ch in chunks:
            codes = self._generate(
                self.model, self.tokenizer, ch, self.speaker_emb,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, rep_penalty=rep_penalty, device=self.device,
            )
            if codes is not None and len(codes) > 0:
                all_codes.append(codes)
        if not all_codes:
            raise RuntimeError("BgTTS produced no audio")
        wav = self.codec.tokens_to_wav(torch.cat(all_codes), self.speaker_emb)
        if hasattr(wav, "numpy"):
            wav = wav.numpy()
        return np.asarray(wav, dtype=np.float32).reshape(-1)
