# Voice Cloning & Production Guide — OmniVoice-BG

How to get a **clean, artifact-free** cloned voice out of OmniVoice-BG, and
how to turn it into a **produced** ad/narration afterward. This is operational
guidance (audio production), separate from the training pipeline
([BG_PIPELINE_GUIDE.md](BG_PIPELINE_GUIDE.md)).

## The one principle: separate voice identity from production

OmniVoice clones by **in-context imitation** — it encodes your reference audio
and *continues that voice*. Crucially, it imitates the reference's **production,
not just its voice**: compression, EQ, loudness, reverb, even mic character all
transfer to the clone. So:

> **Clone from a clean, dry, neutral sample. Apply production to the OUTPUT.**

Cloning a finished commercial master is close to worst-case: the model tries to
reproduce the compression and EQ, which fight the neural codec and produce
**scratchy, hot, unstable** audio. You cannot un-bake a master — but you *can*
give the clone a clean source and re-master its output to any spec you like.

## What makes a good vs bad reference

Measured thresholds (clean male speech in parentheses):

| metric | clean target | bad sign |
|---|---|---|
| **RMS** | 0.05–0.15 | > 0.18 = hot/produced (auto-fixed at encode, but a symptom) |
| **crest factor** | > 14 dB | < 12 dB = compressed/limited (**cannot be undone**) |
| **clipping** | 0 % | > 0.1 % = destructive (needs a cleaner recording) |
| **sub-300 Hz energy** | 10–20 % | > 35 % = bass-heavy "announcer" EQ (high-pass helps) |
| **duration** | 12–15 s clean | 3 min of produced audio is *worse* than 12 s of clean |
| **channels** | mono voice | stereo with a music/SFX bed (separate the voice first) |

Check any reference before using it:

```bash
./venv/bin/python prep_reference.py --check --in your_ref.wav
```

Real example — the `reklama_home_max.wav` ad that caused scratchy clones:
`rms 0.29 (HOT), crest 10.7 dB (COMPRESSED), sub-300 65 % (BASS-HEAVY)` — a
textbook bad reference on three axes at once.

## Preparing a reference (two paths)

### Automatic
```bash
./venv/bin/python prep_reference.py --in raw.wav --out ref_clean.wav
# heavy boom in the voice fundamentals? raise the high-pass:
./venv/bin/python prep_reference.py --in raw.wav --out ref_clean.wav --hpf 150
```
Does: mono downmix → high-pass (removes sub-bass) → normalize to RMS 0.1.
Does **not**: remove compression or un-clip (impossible), or trim to a clean
segment (do that by ear). Prints before/after health.

### Audacity (hands-on, and for the manual steps the tool can't do)
1. **Mono**: Tracks → Mix → Mix Stereo Down to Mono.
2. **Kill sub-bass boom**: Effect → EQ and Filters → Filter Curve EQ — steep
   roll-off below ~85 Hz. If it's still boomy (deep-voice fundamentals in
   100–250 Hz), add a **low-shelf cut of −3 to −5 dB below ~200 Hz** to flatten
   the "announcer" EQ back toward neutral. *(This is the step the auto tool's
   high-pass can't fully do — a shelf cut in the voice band needs your ear.)*
3. **Trim to a clean 12–15 s voice-only segment** — natural sentences, no music
   stings, no long silences. Quality over quantity.
4. **De-ess if harsh**: a narrow EQ cut around 6–8 kHz, gently.
5. **Export** → WAV, and use that as `--ref-audio`.

Note: the model **auto-normalizes reference level** to RMS 0.1 at encode-time
(two-sided, since 2026-07-17), so you no longer need to fix loudness on the
reference — focus on **spectral cleanup and a clean segment**.

## Mastering the output into an ad (afterward)

On the generated `book.wav`, re-apply the production you kept off the reference:

1. **EQ**: presence boost (~3–5 kHz) and warmth (low-shelf ~100–200 Hz) to taste
   — add back the "commercial voice" character here, on the clean clone.
2. **Compression**: Effect → Compressor (broadcast-style, ~3:1).
3. **Loudness**: Effect → Loudness Normalization — **−16 LUFS** for web/social,
   **−23 LUFS** for broadcast. This is the "turn the parameters back up" step,
   done correctly on a clean source.
4. **Music/SFX bed**: on a separate track, then mix.

## Why this works (the mechanism)

In-context cloning has no separate "voice-identity" encoder to isolate timbre
from production (unlike a speaker-embedding model) — the same tokens carry
both. So conditioning the *reference* and mastering the *output* is the only way
to decouple them, and it is strictly better than cloning a master: the clone
stays artifact-free (it never had to reproduce compression), and you keep full,
repeatable control over the final production. One clean clone → many
productions.

## Quick reference

```bash
# 1. check a candidate reference
./venv/bin/python prep_reference.py --check --in raw.wav
# 2. condition it (then trim to a clean segment in Audacity)
./venv/bin/python prep_reference.py --in raw.wav --out ref_clean.wav [--hpf 150]
# 3. synth with the clean reference (sparse marking, bright model)
./venv/bin/python read_book.py --book text.txt --marking sparse \
    --model exp/omnivoice_bg_t5_bright/checkpoint-3000 \
    --ref-audio ref_clean.wav --out reading/mybook
# 4. master reading/mybook/book.wav in Audacity -> ad
```
