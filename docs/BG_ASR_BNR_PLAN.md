# BNR archive harvest — plan (diverse audio for general-purpose ASR)

> **✅ AXIS CLOSED 2026-08-05: `data/fbank/bg_cuts_bnr.jsonl.gz` = 34,425 cuts / 102.9 h.**
> Funnel: 1,035 posts / 3,006 mp3s / ~220 h crawled → 126.7 h speech after the music
> quarantine (58 %) → 87.8 h accepted by the faithfulness filter → +15.1 h recovered by
> Zipformer arbitration (79 % of bg-lang rejects) → 102.9 h cut at 80-dim fbank/16 kHz.
> Audited: 24.9 % inter-model disagreement on accepts (≈ single-digit label error), native
> listening calibrated --lang-prob. Broken-header lesson lives in BG_ASR_V2_PLAN.

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

## YouTube conversational axis — @DnevenRed (added 2026-08-04)

Second harvest source through the SAME pipeline: `yt_harvest.py` pulls a channel's audio +
Bulgarian **auto**-captions (default `@DnevenRed`, publicistic talk/interviews). Scale checked:
**353 videos ≈ 552 h** — nearly the size of the entire v1 corpus, on the missing
conversational/spontaneous axis. ~30 GB m4a; multi-day download at ~1 MB/s; resumable.

- **Captions are YouTube auto-ASR** (`kind=asr`, verified — „автоматично генерирано“), NOT gold.
  They are harvested as a **second teacher**: the pseudo-label pipeline's `--agree-subs` mode
  keeps only spans where Whisper and the YT caption **agree** (independent ASR errors rarely
  coincide → higher-precision labels than confidence alone). Disagreeing spans are dropped.
- Same TOS/gray-zone call as BNR (YouTube TOS prohibits downloading; research practice; do not
  redistribute).
- Combined trajectory: v1 649 h + parliament recovery ~257 h + BNR ~300 h + YT ~552 h + FLEURS/CV
  → **~1,700+ h**, past the ~1,800 h in-domain saturation scale of the Finnish reference.

```bash
./venv/bin/python yt_harvest.py --limit 3     # smoke
./venv/bin/python yt_harvest.py               # full channel (resumable, per-video progress)
# then the same pseudo-label stages with the YT manifest + two-teacher agreement:
./venv/bin/python bnr_pseudolabel.py --stage segment    --manifest data/ytdr/yt_manifest.jsonl --workdir data/ytdr
./venv/bin/python bnr_pseudolabel.py --stage transcribe --manifest data/ytdr/yt_manifest.jsonl --workdir data/ytdr --agree-subs
./venv/bin/python bnr_pseudolabel.py --stage cuts       --workdir data/ytdr --out-name bg_cuts_ytdr
```

## Fourth axis — Bloomberg TV Bulgaria (added 2026-08-04, CAPPED by design)

`@bloombergtvbulgaria`: broadcast **business/finance news** — anchors, expert interviews,
financial vocabulary no other axis covers. Captions AUTO (verified) → `--agree-subs` applies.
Channel archive is enormous (full enumeration times out ⇒ thousands of uploads; newest 50 avg
19.2 min ≈ 16 h), so the harvest is **deliberately capped**, not exhaustive:

1. **Register balance** — uncapped, business news would dominate the corpus and bias the
   transducer's internal LM priors toward financial phrasing; registers should stay within
   ~0.5–2× of each other for a general-purpose model.
2. **Diversity-per-hour decays** — a TV channel is a few recurring anchors in one studio; the
   first ~150 h buys the register/acoustics/vocabulary, hour 2,000 is near-duplicate.
3. **Pseudo-labeling is the binding cost** — every crawled hour costs Whisper GPU time
   (~160 h ≈ a GPU-day; a full archive ≈ weeks) for mostly-duplicate content.

Safety valves: train-time source **sampling weights** (lhotse mux) can rebalance regardless,
and the resumable harvest extends with a higher `--limit` if evaluation shows the register is
under-served. (`--limit` also caps the *enumeration* — needed for huge channels.)

```bash
# after the DnevenRed download completes (serialize on the ~1 MB/s link):
./venv/bin/python yt_harvest.py \
  --channel https://www.youtube.com/@bloombergtvbulgaria/videos \
  --out-dir data/ytbb --limit 500        # newest ~500 eps ≈ ~160 h / ~10 GB
# then the same pipeline:
./venv/bin/python bnr_pseudolabel.py --stage segment    --manifest data/ytbb/yt_manifest.jsonl --workdir data/ytbb
./venv/bin/python bnr_pseudolabel.py --stage transcribe --manifest data/ytbb/yt_manifest.jsonl --workdir data/ytbb --agree-subs
./venv/bin/python bnr_pseudolabel.py --stage cuts       --workdir data/ytbb --out-name bg_cuts_ytbb
```

## Fifth axis — «Говори интернет» podcasts (added 2026-08-05, casual conversational)

`UC4rydPOr1yMo4y33Md3ExFg`: Bulgaria's leading podcast network — **casual, NON-political
conversation** (food, travel, personal stories), the register the scarcity assessment flagged
as the top new-data need. 400+ episodes, median 87 min, AUTO captions (agree-subs applies).
**Capped `--limit 180`** (~260 h raw → ~200 h post-filter) so the conversational register
(with DnevenRed) stays inside the 0.5–2× balance band.

```bash
./venv/bin/python yt_harvest.py \
  --channel https://www.youtube.com/channel/UC4rydPOr1yMo4y33Md3ExFg/videos \
  --out-dir data/ytgi --limit 180
# segment stage MUST use the batched loop (87-min files): --limit 30 per batch
# then: transcribe --agree-subs --workdir data/ytgi ... ; cuts --out-name bg_cuts_ytgi
```

## Pipeline commands (tooling complete 2026-08-04; crawl launched)

All steps are **resumable** — after any interruption (power cut), re-run the same line; each
stage skips work already recorded in its output file.

```bash
cd ~/WORK/OmniVoice

# 1. CRAWL (running since 2026-08-04) — ~1,126 posts, ~300h+/~17GB, several hours.
#    Prints one line per segment + per-post summary with ETA. Resume = same line.
./venv/bin/python bnr_crawl.py
#    variants:  --category vremena-i-hora   (one section)    --delay 1.5   (gentler)
# -> data/bnr/audio/<Id>.mp3  +  data/bnr/bnr_manifest.jsonl (metadata per segment)

# 2. one-time install for stage 1 (TensorFlow; CPU is fine):
./venv/bin/python -m pip install inaSpeechSegmenter

# 3. PSEUDO-LABEL — three stages, run as separate invocations (TF vs torch).
#    Each is resumable; use --limit N to process in batches.
./venv/bin/python bnr_pseudolabel.py --stage segment      # speech/music quarantine (CPU)
./venv/bin/python bnr_pseudolabel.py --stage transcribe   # Whisper large-v3 + faithfulness filter (GPU, the long one)
./venv/bin/python bnr_pseudolabel.py --stage cuts         # accepted labels -> fbank cuts
# -> data/bnr/segments.jsonl -> data/bnr/pseudolabels.jsonl -> data/fbank/bg_cuts_bnr.jsonl.gz
#    filter thresholds (conservative, tune after calibration):
#    --lang-prob 0.8 --min-logprob -0.7 --max-nospeech 0.5 --max-compression 2.4

# 4. FOLD INTO TRAINING (hold out a BNR slice first if BNR should also be an eval domain):
./venv/bin/python merge_train_corpus.py --train data/fbank/bg_cuts_bnr.jsonl.gz \
    ... (other --train sources) --holdout ... --out data/fbank/bg_cuts_train_gp.jsonl.gz
```

## Open questions
- Total scale (hours) — **preliminary ~300 h** (1,126 posts × ~5.2 audio segs × ~3.0 min,
  from an 8-post sample via `bnr_scale_check.py`; run `--sample 60` to tighten). On par with AB400
  (397 h), a new axis. NOTE: `Source` field is often empty → filter Bulgarian by Whisper language-id.
- Licensing posture (crawl-and-train vs. request permission).
- Whisper faithfulness filter thresholds — needs a small vetted calibration set.

## Cross-references
[BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md) (general-purpose phase 2) ·
[BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md) (specialist↔generalist evidence).
