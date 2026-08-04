# Stress-emitting ASR — plan (FUTURE WORK / TODO)

**Status:** drafted 2026-08-04 as **FUTURE WORK** — a parallel *specialist* track, gated behind
the general-purpose data campaigns (BNR/YT harvests, v2/GP retrain). Nothing here blocks or
modifies the general-purpose model, which **stays plain-text**.

## Idea

Train a Zipformer that transcribes into **stress-annotated Bulgarian** (U+0301 on the stressed
vowel), and/or combine it with the Accentor as a two-teacher system.

**Key insight:** stress is *in the audio* (vowel quality / duration / energy — the same signal
the Kaldi pron-variant harvest exploits), but the Accentor is text→stress and can only *guess
from context*. The homograph residual (въ́лна/вълна́, prior dominance, single-variant classes,
the manual-vetting bottleneck) is exactly where context under-determines stress while the audio
states it outright. A stress-emitting ASR is a **neural stress-harvester at corpus scale** — the
upgrade of the GMM pron-variant harvest that fed the Accentor in the first place.

## Option A — direct stress-ASR

1. **Marked targets:** re-mark the ASR training transcripts the same way the corpus was built —
   lexicon marks for unambiguous words + the **tri4b aligner's per-token variant decisions** for
   homographs (`get_prons` / `harvest_stress_labels.py` machinery, already exists).
2. **BPE:** retrain on marked text (U+0301 combining char changes tokenization).
3. **Train:** same Zipformer recipe, marked targets; markable corpus only = parliament + AB400
   (+ the recovered ~257 h). ~1 GPU-day + prep.
4. **Eval, two axes:** (a) plain WER with marks stripped — must NOT degrade vs. the plain model;
   (b) **stress accuracy** on the vetted battery / oracle assets (bon_eval, 287K lexicon).
5. **"Is it actually listening?" test:** homograph minimal pairs where context is ambiguous but
   audio differs — if the model tracks the audio (not the text prior), it's genuinely hearing
   stress rather than acting as an implicit text-context Accentor. (That failure mode is still
   ≥ Accentor-quality, but the acoustic tie-break is the payoff.)

## Option B — ASR × Accentor two-teacher stress harvest

**Acoustic teacher** (stress-ASR: what was *said*) × **contextual teacher** (Accentor: what the
text *predicts*) — same agreement logic as the YT caption filter, applied to stress:
- **Agreement** → high-confidence stress labels, harvested automatically at corpus scale —
  orders of magnitude beyond the manual-vetting throughput that capped the Accentor's
  data-starved classes.
- **Disagreement** → the gold dust: Accentor errors (→ retraining data) or genuine speaker
  variation/exceptions — precisely the cases manual vetting was rationing.
- Later extendable to *clean, verified* subsets of BNR/YT (never to raw pseudo-labeled audio).

## Guards

- **CIRCULARITY (hard rule):** never train the stress-ASR on **Accentor-marked** text. Marks
  must stay anchored in lexicon + acoustic alignment (human-derived), or the two "teachers"
  collapse into one and agreement is meaningless. (Mirror of the existing guard that stripped
  U+0301 from ASR supervisions.)
- **Scope:** the general-purpose ASR stays plain-text — BNR/YT pseudo-labels can't carry
  reliable marks (no verified transcripts). The stress-ASR is a separate specialist tool.

## Flow-back (why bother)

- **Accentor:** massive audio-verified homograph labels → real supervision for the starved
  classes (see ACCENTOR_REFRESH_PLAN / ft7d history).
- **TTS (ft9 voice-mining):** mined clean segments get **stress-marked transcripts matching
  what was actually said** — exactly the "marked finetune" recipe OmniVoice-BG needs.
- **Product niche:** transcription mode outputting pedagogically-marked Bulgarian (language
  learning / accessibility).

## TODO (when picked up)

1. Re-mark parliament+AB400 transcripts from existing tri4b alignments → marked `text`.
2. Marked BPE-500 (or bump) → `data/lang_bpe_500_marked/`.
3. Train stress-Zipformer (30 ep, same recipe) → eval (a)+(b)+listening test.
4. Build the Option-B harvest + disagreement report; feed Accentor refresh.

## Cross-references

[BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md) (general-purpose track this must not derail) ·
[ACCENTOR_REFRESH_PLAN.md](ACCENTOR_REFRESH_PLAN.md) (the consumer of the harvested labels) ·
[BG_ASR_BNR_PLAN.md](BG_ASR_BNR_PLAN.md) (two-teacher precedent: `--agree-subs`).
