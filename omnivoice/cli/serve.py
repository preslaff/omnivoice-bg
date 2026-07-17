#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal HTTP server exposing OmniVoice TTS over a LAN.

Loads the model once and serves synthesis requests, so other machines on the
same network (e.g. an app server) can use OmniVoice without a local GPU.

Run on the GPU machine:

    python -m omnivoice.cli.serve --model k2-fsa/OmniVoice --host 0.0.0.0 --port 8002

DETERMINISTIC / CONSISTENT VOICE
--------------------------------
Voice *design* (the ``instruct`` field) only constrains attributes (gender,
age, pitch, accent) — it does NOT pin a speaker, so different texts come out in
different voices. To get the SAME voice + accent across every clip, pin one
reference audio at startup (voice cloning); every request then clones it:

    python -m omnivoice.cli.serve \
        --ref-audio narrator_bg.wav --ref-text "<exact transcript of the clip>" \
        --seed 42 --host 0.0.0.0 --port 8002

By default decoding is greedy (``position_temperature=0``, ``class_temperature=0``)
and seeded, so identical inputs yield identical audio. Per-request fields can
override any of this.

Endpoints:
    GET  /health  -> {"status": "ok", "sampling_rate": 24000, "fixed_voice": bool}
    POST /synth   -> audio/wav (24 kHz mono)
        body (JSON), all optional except ``text``:
            {
                "text": "Добър ден",          # required
                "instruct": "female, ...",      # voice design (ignored if a ref is used)
                "language": "bg",
                "speed": 1.0,
                "duration": 10.0,               # overrides speed
                "ref_audio_b64": null,          # base64 WAV -> per-request cloning
                "ref_text": null,               # transcript of ref_audio_b64
                "num_step": 32,
                "guidance_scale": 2.0,
                "position_temperature": 0.0,    # 0 = deterministic ordering
                "class_temperature": 0.0,       # 0 = greedy token choice
                "seed": 42
            }

No authentication — intended for a trusted LAN only. Do not expose to the internet.
"""

import argparse
import base64
import hashlib
import io
import logging
import threading
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from omnivoice import OmniVoice

logger = logging.getLogger("omnivoice.serve")

SAMPLING_RATE = 24000


class SynthRequest(BaseModel):
    text: str
    instruct: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[float] = None
    duration: Optional[float] = None
    ref_audio_b64: Optional[str] = None
    ref_text: Optional[str] = None
    # Decoding overrides (fall back to the server defaults when omitted).
    num_step: Optional[int] = None
    guidance_scale: Optional[float] = None
    position_temperature: Optional[float] = None
    class_temperature: Optional[float] = None
    seed: Optional[int] = None
    accent: Optional[bool] = None  # Bulgarian stress marking; None = use server default
    engine: Optional[str] = None  # "omnivoice" | "bgtts"; None = server default


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_app(
    model: OmniVoice,
    defaults: dict,
    default_prompt=None,
    accenter=None,
    bgtts_engine=None,
    default_engine: str = "omnivoice",
) -> FastAPI:
    app = FastAPI(title="OmniVoice TTS")
    # Serialize generation: one model on one GPU; concurrent calls would race
    # and can OOM. Requests queue and run one at a time. Serialization also keeps
    # seeding deterministic (no interleaved RNG consumption between requests).
    # The lock is shared across engines — both share the single GPU.
    lock = threading.Lock()
    # Cache encoded voice prompts so a repeated reference (same client voice) is
    # not re-encoded on the GPU every call. Keyed by hash(raw audio + transcript).
    prompt_cache = {}

    @app.get("/health")
    def health():
        engines = ["omnivoice"] + (["bgtts"] if bgtts_engine is not None else [])
        return {
            "status": "ok",
            "sampling_rate": SAMPLING_RATE,
            "fixed_voice": default_prompt is not None,
            "bg_accent": accenter is not None,
            "accent_frontend": type(accenter).__name__ if accenter else None,
            "engines": engines,
            "default_engine": default_engine,
        }

    @app.post("/synth")
    def synth(req: SynthRequest):
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="`text` is empty")

        engine = (req.engine or default_engine).lower()
        if engine not in ("omnivoice", "bgtts"):
            raise HTTPException(status_code=400, detail=f"unknown engine: {engine!r}")

        # --- BgTTS-38M engine: plain text (no accenting), fixed cloned voice. ---
        if engine == "bgtts":
            if bgtts_engine is None:
                raise HTTPException(
                    status_code=400,
                    detail="bgtts engine not loaded (start server with --bgtts-dir)",
                )
            try:
                with lock:
                    audio = bgtts_engine.synth(req.text)
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise HTTPException(status_code=503, detail=f"GPU OOM: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                logger.exception("bgtts generation failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            buf = io.BytesIO()
            sf.write(buf, audio, SAMPLING_RATE, format="WAV")
            return Response(content=buf.getvalue(), media_type="audio/wav")

        # Bulgarian stress marking (server-side). Only touches Cyrillic words.
        text = req.text
        if accenter is not None and req.accent is not False:
            text = accenter.accent_text(text)

        kwargs = {"text": text}

        # --- Voice selection (priority: per-request ref > fixed server ref > instruct) ---
        # `ref_wav`/`ref_key` are prepared here (CPU, cheap); the GPU encode of a
        # cache-missed reference happens inside the generation lock below.
        ref_wav = ref_key = None
        if req.ref_audio_b64:
            try:
                raw = base64.b64decode(req.ref_audio_b64)
                data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400, detail=f"bad ref_audio_b64: {exc}"
                ) from exc
            wav = data.T if data.ndim > 1 else data  # -> (C, T) / (T,)
            ref_wav = (np.atleast_2d(wav), sr)
            ref_key = hashlib.sha256(raw + (req.ref_text or "").encode()).hexdigest()
        elif default_prompt is not None:
            kwargs["voice_clone_prompt"] = default_prompt
        elif req.instruct:
            kwargs["instruct"] = req.instruct

        if req.language:
            kwargs["language"] = req.language
        if req.speed is not None:
            kwargs["speed"] = req.speed
        if req.duration is not None:
            kwargs["duration"] = req.duration

        # --- Decoding params: request overrides, else server defaults ---
        def pick(name):
            val = getattr(req, name)
            return defaults[name] if val is None else val

        kwargs["num_step"] = pick("num_step")
        kwargs["guidance_scale"] = pick("guidance_scale")
        kwargs["position_temperature"] = pick("position_temperature")
        kwargs["class_temperature"] = pick("class_temperature")
        seed = pick("seed")

        try:
            with lock:
                if ref_key is not None:
                    if ref_key not in prompt_cache:
                        prompt_cache[ref_key] = model.create_voice_clone_prompt(
                            ref_wav, ref_text=req.ref_text
                        )
                    kwargs["voice_clone_prompt"] = prompt_cache[ref_key]
                _set_seed(seed)
                audio = model.generate(**kwargs)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail=f"GPU OOM: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        buf = io.BytesIO()
        sf.write(buf, audio[0], SAMPLING_RATE, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    return app


def main():
    parser = argparse.ArgumentParser(description="OmniVoice LAN TTS server")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 to accept LAN clients")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 / cpu (default: auto)")

    # Fixed voice (voice cloning) — gives the same speaker + accent on every call.
    parser.add_argument("--ref-audio", default=None, help="reference WAV to clone for all requests")
    parser.add_argument(
        "--ref-text",
        default=None,
        help="exact transcript of --ref-audio (omit to auto-transcribe with Whisper)",
    )

    # Deterministic decoding defaults.
    parser.add_argument("--seed", type=int, default=42, help="RNG seed per request (-1 to disable)")
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument(
        "--position-temperature", type=float, default=0.0, help="0 = deterministic unmask order"
    )
    parser.add_argument(
        "--class-temperature", type=float, default=0.0, help="0 = greedy token choice"
    )

    # Bulgarian stress marking (server-side). Marks Cyrillic words before TTS.
    parser.add_argument("--lexicon", default=None, help="rechko stress sqlite (enables BG accenting)")
    parser.add_argument(
        "--accentor", default=None,
        help="Accentor checkpoint (.pt) for CONTEXTUAL stress marking — "
             "resolves homographs by sentence context; supersedes --lexicon",
    )
    parser.add_argument("--accent-threshold", type=float, default=0.5,
                        help="Accentor: min prob to place a stress mark")
    parser.add_argument("--overrides", default=None, help="JSON of word->stressed overrides")
    parser.add_argument("--sparse-keep", default=None,
                        help="JSON keep-list: SPARSE marking (mark only "
                             "homographs + hard words + pins). Match the served "
                             "TTS checkpoint's marking dialect (F9): use with a "
                             "sparse-trained model, omit with a dense one.")

    # Optional alternative engine: BgTTS-38M-V2 (plain text, MioCodec cloning).
    parser.add_argument(
        "--bgtts-dir", default=None,
        help="dir with BgTTS-38M-V2 code + checkpoint_inference.pt (enables the 'bgtts' engine)",
    )
    parser.add_argument(
        "--default-engine", default="omnivoice", choices=["omnivoice", "bgtts"],
        help="engine used when a request omits 'engine'",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    logger.info("Loading %s on %s ...", args.model, device)
    # load_asr only needed when a reference is given without its transcript.
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        load_asr=bool(args.ref_audio and not args.ref_text),
    )

    default_prompt = None
    if args.ref_audio:
        logger.info("Pre-encoding fixed reference voice from %s ...", args.ref_audio)
        default_prompt = model.create_voice_clone_prompt(
            args.ref_audio, ref_text=args.ref_text
        )
        logger.info("Fixed voice ready; all requests will clone it.")

    defaults = {
        "num_step": args.num_step,
        "guidance_scale": args.guidance_scale,
        "position_temperature": args.position_temperature,
        "class_temperature": args.class_temperature,
        "seed": None if args.seed < 0 else args.seed,
    }

    accenter = None
    if args.accentor:
        from accentor_model import AccentorAccenter  # repo-root; run from root

        accenter = AccentorAccenter(args.accentor, args.overrides,
                                    device="cpu",
                                    threshold=args.accent_threshold,
                                    sparse_keep=args.sparse_keep)
        logger.info("BG stress accenting ON — CONTEXTUAL Accentor "
                    "(ckpt=%s step=%s, overrides=%s, marking=%s)",
                    args.accentor, accenter.step, args.overrides,
                    "sparse" if args.sparse_keep else "dense")
    elif args.lexicon:
        from bg_accent import BgAccenter  # repo-root module; run from repo root

        accenter = BgAccenter(args.lexicon, args.overrides)
        logger.info("BG stress accenting ON — rechko lexicon "
                    "(lexicon=%s, overrides=%s)", args.lexicon, args.overrides)

    bgtts_engine = None
    if args.bgtts_dir:
        from omnivoice.cli.bgtts_engine import BgTTSEngine

        logger.info("Loading BgTTS-38M engine from %s ...", args.bgtts_dir)
        bgtts_engine = BgTTSEngine(args.bgtts_dir, ref_audio=args.ref_audio, device=str(device))
        if bgtts_engine.speaker_emb is None:
            logger.warning("BgTTS loaded WITHOUT a fixed voice — pass --ref-audio to enable it.")
        else:
            logger.info("BgTTS engine ready (fixed voice cloned from %s).", args.ref_audio)

    if args.default_engine == "bgtts" and bgtts_engine is None:
        parser.error("--default-engine bgtts requires --bgtts-dir")

    logger.info(
        "Decoding defaults: %s | fixed_voice=%s | bg_accent=%s | engines=%s | default=%s",
        defaults, default_prompt is not None, accenter is not None,
        ["omnivoice"] + (["bgtts"] if bgtts_engine else []), args.default_engine,
    )
    logger.info("Serving on http://%s:%d", args.host, args.port)

    app = build_app(model, defaults, default_prompt, accenter, bgtts_engine, args.default_engine)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
