# BNR archive harvest — plan (diverse audio for general-purpose ASR)

**Status:** drafted 2026-08-04, exploratory. Feeds the **general-purpose** pivot
([BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md) phase 2 / [bg-asr-general-purpose-pivot memory]).
Goal: harvest **domain-diverse Bulgarian audio** from the Bulgarian National Radio archives
(`https://archives.bnr.bg/`) to widen ASR coverage beyond parliament (formal read) + AB400
(read literary) + FLEURS/CV (read). BNR = interviews, radio drama, spoken-word, culture, sport
— **spontaneous / performed speech**, a genuinely new axis. **ASR only; not for TTS** (see below).

## Feasibility — CONFIRMED (structure, 2026-08-04)

The archive is a **Next.js** app; the crawl is fully scriptable (no headless browser):

| Step | Mechanism |
|---|---|
| Policy | `robots.txt` **allows** crawling (only `/*/search` disallowed); `sitemap.xml` exists |
| Enumerate | Sitemap index → `/api/program-sitemap/archives` (all post URLs) + `/api/cms-sitemap/archives`. Categories: `vremena-i-hora`, `kultura`, `muzika`, `sport`. Section listings also paginate. |
| Per post | Page embeds `<script id="__NEXT_DATA__">` JSON; audio at `props.pageProps.data.Sections[].Audio = {Id, FileName, Source, Description}`. One post holds several segments (e.g. post 13230 = 9 sections, 4 with audio). |
| Download | **`https://archives.bnr.bg/api/media/<Audio.Id>`** → real MP3 (**128 kbps, 44.1 kHz stereo**, HTTP-range supported). |
| Filter | `Source` tags provenance (drop non-BNR/foreign, e.g. a British-institute segment); `Description` = rich per-segment Bulgarian context (not a transcript). |

**Crawl recipe:** sitemap → post pages → parse `__NEXT_DATA__` → `Sections[].Audio.Id` →
GET `/api/media/<Id>`. Store audio + `Source`/`Description`/post metadata. Rate-limit + polite UA.

## Caveats (decide before investing)

1. **Copyright.** `robots.txt` allowing a crawl is NOT a training license — BNR archives are
   copyrighted public-broadcaster content. Training on them (transformative) is more defensible
   than redistributing; do **not** republish the audio. BNR is a public institution — **asking
   for a research-use license is worth it** and de-risks the effort.
2. **No transcripts.** These are pseudo-labeled (Whisper), so they are *weak supervision*, not gold.

## Music / musical-play quarantine (data quality — REQUIRED)

BNR archives contain **music, opera, and musical radio-plays** (songs, orchestral beds, sung
drama). This is **poison for ASR**: Whisper transcribes music/lyrics unreliably, and music beds
corrupt the acoustic targets. It MUST be quarantined. Two tiers:

1. **Metadata is NOT enough** (verified 2026-08-04): segment `Description` is the generic
   "Аудио файл" for ~all posts, and titles/slugs only sometimes hint at music — a text filter
   would miss embedded music and wrongly drop *spoken* drama. Only cheap, reliable metadata move:
   **drop the `muzika` category wholesale** (all music).
2. **Audio-based speech/music segmentation (the real quarantine).** Run a broadcast speech/music
   segmenter — **`inaSpeechSegmenter`** (INA, purpose-built for radio archives: labels each region
   speech / music / noise) is the natural fit. Keep only **speech** regions, drop **music**. This
   handles musical plays perfectly (drops the sung/music spans, keeps spoken dialogue) AND doubles
   as the VAD/segmentation step feeding Whisper. NB: plain VAD (Silero) is NOT enough — singing
   registers as "voice", so it wouldn't separate sung music from speech; use a speech-vs-MUSIC model.

Order in the pipeline: **segment (speech/music) → keep speech → Whisper → faithfulness filter.**

## Pseudo-labeling pipeline — faithfulness over fluency

Whisper transcribes (~15% WER on general BG — it's Whisper's own weak-supervision recipe applied
to Bulgarian). **The trap:** "have BgGPT correct it" is risky — an LLM that does not hear the audio
can only fix grammar/punctuation; where Whisper *misheard*, the LLM will drift the text further
from what was said, and ASR training on text that reads well but does not match the audio is
actively harmful. So:

- **Keep labels FAITHFUL by FILTERING, not rewriting.** Speech/music-segment (keep speech, drop
  music — above) → Whisper →
  keep only high-confidence segments and/or those where **forced alignment** (align Whisper text
  back to the audio, drop low-score spans) agrees. Faithful-but-noisy > fluent-but-wrong.
- **BgGPT (if used at all) = light normalization only** — punctuation, casing, number spelling —
  never word substitution. Or drop it entirely.
- Optional: **two-teacher agreement** (Whisper + a second ASR) to keep only agreeing spans.

## Deep Domain Adaptation — the honest take

For general-purpose ASR, **formal DDA (adversarial/DANN, gradient reversal, domain-invariant
features) is NOT the winning approach** here. Those are for a fixed labeled source + unlabeled
target you *cannot* label — but we *can* label the target (Whisper), turning it into a simpler,
more effective **weakly-supervised multi-domain** problem. What actually works:

- **Pseudo-labeling / self-training / noisy-student** — label BNR with Whisper, train, iterate.
  This is "domain adaptation" in the practical sense, and it scales.
- **Multi-condition training** — just mix parliament + AB400 + FLEURS + CV + BNR; the model learns
  domain-invariant features implicitly (the diversify pivot).
- **Self-supervised pretraining on raw audio** (wav2vec2/HuBERT) — the one strong lever needing
  **no transcripts**: pretrain an encoder on raw BNR audio, then fine-tune. Big robustness win, but
  a different architecture from the Zipformer transducer (separate track).
- **SpecAugment / augmentation** — cheap robustness (already on).

**TTS: BNR is a poor fit; DDA won't rescue it.** TTS needs clean single-speaker audio + *precise*
transcripts; BNR is multi-speaker, music-bedded, interview/drama with pseudo-labels — the two
things TTS is most sensitive to. General-purpose TTS is a multi-speaker + voice-cloning problem
(OmniVoice already does this via in-context acoustic prompts). Use BNR for ASR; for TTS only if
cherry-picking clean solo-narration segments with verified transcripts.

## Plan

1. **(Optional, smart)** email BNR for a research-use nod.
2. **Scale-check crawl** — enumerate the sitemap, count posts / audio segments / est. hours, so we
   know if this is ~50 h or ~5000 h before committing. *(Do this first.)*
3. **Crawler** — sitemap → posts → `__NEXT_DATA__` → download `/api/media/<Id>`; store audio +
   metadata, Bulgarian-filtered, rate-limited.
4. **Pseudo-label** — **speech/music-seg (inaSpeechSegmenter, drop music/musical-plays)** → Whisper
   large-v3 → **faithfulness filter** (confidence + forced align);
   light normalization only → cuts (80-dim fbank @ 16 kHz, same `bg_text_norm`).
5. **Fold into ASR training** (multi-domain, phase 2+) → general-purpose model. Optionally
   **self-training** iterations, or a **wav2vec2 SSL** side-track on the raw audio.
6. Evaluate on frozen parliament test + held-out FLEURS/CV test + a **fresh unseen set** (the true
   OOD probe — BNR held-out slice or self-recorded).

## Open questions
- Total scale (hours) — **preliminary ~300 h** (1,126 posts × ~5.2 audio segs × ~3.0 min,
  from an 8-post sample via `bnr_scale_check.py`; run `--sample 60` to tighten). On par with AB400
  (397 h), a new axis. NOTE: `Source` field is often empty → filter Bulgarian by Whisper language-id.
- Licensing posture (crawl-and-train vs. request permission).
- Whisper faithfulness filter thresholds — needs a small vetted calibration set.

## Cross-references
[BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md) (general-purpose phase 2) ·
[BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md) (specialist↔generalist evidence).
