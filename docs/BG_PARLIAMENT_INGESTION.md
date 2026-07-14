# Bulgarian Parliament Ingestion — Phase 1 of the Accentor reproduction

This document describes the **data-ingestion pipeline** that builds a corpus of per-session
*audio + transcript* pairs from the Bulgarian National Assembly archive. It is **Phase 1** of an
effort to reproduce the **Accentor** contextual lexical-stress model (Geneva et al., Interspeech
2023) for OmniVoice-BG.

## Why we're building this

OmniVoice-BG produces natural, voice-cloned Bulgarian narration; the combined fine-tune
(`exp/omnivoice_bg_combined/checkpoint-2000`, exported to `models/omnivoice-bg-v2/`) honors
stress marks while keeping cloning quality. The remaining stress errors are **homographs**
(за́мък/замъ́к, вълна́/въ́лна, говори́/гово́ри…) that the rechko lexicon **structurally cannot**
resolve — a word→one-stress lookup can't disambiguate by context. The hard-word battery proved
it (lexicon feeds the wrong mark in 6/8 homograph cases).

The fix is a **contextual** stress front-end — an Accentor — that resolves stress by
whole-sentence context and feeds the existing acoustic model. The Accentor's power comes from
training on stress labels **decoded from real human speech**. The Bulgarian Parliament archive
(2010–2018, ~11.7k h) is the paper's source and gives that human ground truth.

> **Scope of Phase 1 (this doc):** download session videos → extract 24 kHz mono audio → fetch
> + clean stenographic transcripts → per-session pairs + manifest. **Out of scope:** segmentation,
> forced alignment, stress annotation, Accentor training — those are later phases.

## Source: the parliament.bg API (verified)

One JSON call returns transcript **and** video URLs together — no SPA scraping, no video↔text
matching:

```
GET https://www.parliament.bg/api/v1/pl-sten/{ID}
→ { Pl_Sten_id, Pl_Sten_date, Pl_Sten_sub,
    Pl_Sten_body,                # full HTML transcript (~1 MB; <br>, inline speaker labels)
    video: { Vidate, Vicount,    # Vicount = number of video parts
             default,            # first-part MP4 URL
             playlist: [ {item, file}, … ] },   # ordered ready-made MP4 URLs
    files, videoArchive }
```

- **Video:** direct MP4 at `https://parliament.bg/Gallery/video/archive-YYYY_MM_DD_N.mp4`
  (`N` = 1..`Vicount`). `Content-Type: video/mp4`, `Accept-Ranges: bytes`, ~482 MB/part
  (~1 GB/session). Use `playlist[].file` URLs directly; don't construct by hand.
- **Enumeration caveat:** stenogram IDs are **not** globally monotonic by date. IDs below ~2540
  are a jumbled re-import of 1990s–2000s sessions; **IDs ~2540–7400 form a monotonic "modern
  block"** that contains all of 2010–2018. Binary search is valid only with a lower guard at ~2540.
- Enumerate by **date**, not by assembly — `A_ns_id` is an internal id (44th NS = 52), not the
  NS number; the mapping is unsafe.

## Output layout (`data/bg_parliament/`, gitignored under `/data/`)

```
audio/plsten_<ID>.flac          # 24 kHz mono FLAC, one per session
transcripts/plsten_<ID>.txt     # cleaned spoken text (== manifest "text")
turns/plsten_<ID>.jsonl         # {speaker, role, text} turns — anchors for next-phase alignment
manifest.jsonl                  # one row per completed session
errors.jsonl                    # {id, stage, error, ts}
enum_cache.jsonl                # {id, date} — makes re-enumeration instant
state/plsten_<ID>.ok            # atomic completion marker (written LAST)
raw/plsten_<ID>/part_N.mp4      # TRANSIENT — deleted immediately after extraction
```

Manifest row (superset of `{id, audio_path, text, language_id}` consumed by
`omnivoice/scripts/extract_audio_tokens.py`; extra keys are ignored downstream):

```json
{"id":"plsten_2694","audio_path":"<ABS>/audio/plsten_2694.flac",
 "text":"<cleaned full-session spoken text>","language_id":"bg",
 "session_date":"2012-01-27","pl_sten_id":2694,"n_parts":2,
 "duration":12831.4,"turns_path":"<ABS>/turns/plsten_2694.jsonl"}
```

## Components

| File | Role |
|---|---|
| `prep_bg_parliament.py` | CLI orchestrator: enumeration → per-session download/extract/clean → manifest. Resumable, rate-limited. |
| `bg_sten_clean.py` | Pure, unit-testable transcript cleaner: `clean_body(html) -> (full_text, turns)`. |

Convention match: mirrors `prep_bg400.py` (argparse, tqdm, absolute paths, manifest writing);
the next phase reuses `omnivoice/utils/audio.py` (`load_audio`, `remove_silence`, `trim_long_audio`).

### Enumeration
`find_edge(date, lo_guard=2540, hi_guard=10591, want=first_ge|last_le)` binary-searches the
monotonic modern block for boundary IDs (≈ 2540–7400 for 2010–2018), then a linear walk fetches
`Pl_Sten_date`, keeps IDs in range, caches `{id,date}` in `enum_cache.jsonl`. A runtime
monotonicity assertion falls back to pure linear filtering on any backward date jump.

### Per-session processing
Skip if `state/*.ok` exists → fetch JSON → `playlist = sorted(by item)`, assert `len == Vicount`
→ per part: stream-download (resumable Range), `ffmpeg -nostdin -i part.mp4 -vn -ac 1 -ar 24000
-c:a flac scratch/part_N.flac`, **delete MP4 immediately** → concat ordered part-FLACs
(`ffmpeg -f concat -c copy`; single-part skips concat) → `clean_body()` → write transcript +
turns → `ffprobe` sanity (`sr==24000, ch==1`) → append manifest row → write `.ok` last.

### Transcript cleaning (`bg_sten_clean.clean_body`)
`<br>`→`\n`, `html.unescape`, strip tags; drop procedural headers (`Председателствали:`,
`Секретари:`); detect ALL-CAPS line-leading speaker labels
(`ПРЕДСЕДАТЕЛ|ЗАМЕСТНИК-ПРЕДСЕДАТЕЛ|МИНИСТЪР…|ДОКЛАДЧИК` + name + optional `(...)` + `:`);
strip inline stage directions `(...)` (`(Звъни)`, `(Ръкопляскания.)`); build speaker turns;
`full_text` = turn texts joined, labels removed.

> **STENOGRAPHIC ≠ VERBATIM.** The record is lightly edited (disfluencies removed, grammar
> tidied), so next-phase forced alignment must tolerate insertions/deletions and must **not**
> assume exact word-for-word correspondence.

## CLI

```bash
python prep_bg_parliament.py \
    --out data/bg_parliament \
    --date-start 2010-01-01 --date-end 2018-12-31 \
    --sleep 1.5 --max-retries 5 --backoff 2.0 \
    --min-free-gb 60 --audio-format flac
```

Flags: `--out`, `--date-start/--date-end`, `--id-start/--id-end` (override edges), `--ids a,b,c`
(explicit session list, bypasses enumeration), `--limit N` (pilot), `--reverse` (newest date first),
`--dry-run` (enumerate only), `--sleep`, `--max-retries`, `--backoff`, `--min-free-gb` (disk
guardrail — STOP before crossing), `--audio-format flac|wav` (default flac), `--keep-video` (debug).

### Newest-first, disk-budgeted crawl (chosen plan: ~1.2 TB, skip corrupt 2010–2012)

ID map (verified 2026-07): **2020–2026 → IDs 10425–~11140** (recent block, after the 1879–1927
re-import at 6300–10420); **2013–2019 → IDs ~2900–6300**. Run recent block first, then the mid block;
`--min-free-gb 200` auto-stops each at the ~1.2 TB budget (200 GB floor). Runs are resumable — re-run
the same command and `state/*.ok` skips finished sessions.

```bash
cd ~/WORK/OmniVoice

# sanity-check first (enumerate only, no downloads):
./venv/bin/python prep_bg_parliament.py --out data/bg_parliament \
    --date-start 2020-01-01 --date-end 2026-12-31 --id-start 10425 --id-end 11150 \
    --dry-run --sleep 0.3

# Block A — 2020–2026 (newest, cleanest) — run in background, survives logout:
nohup ./venv/bin/python prep_bg_parliament.py --out data/bg_parliament \
    --date-start 2020-01-01 --date-end 2026-12-31 --id-start 10425 --id-end 11150 \
    --reverse --sleep 1.0 --min-free-gb 200 --audio-format flac \
    > data/crawl_2020_2026.log 2>&1 &

# Block B — 2013–2019, for the remaining budget (run after A, or when A idles):
nohup ./venv/bin/python prep_bg_parliament.py --out data/bg_parliament \
    --date-start 2013-01-01 --date-end 2019-12-31 --id-start 2900 --id-end 6400 \
    --reverse --sleep 1.0 --min-free-gb 200 --audio-format flac \
    > data/crawl_2013_2019.log 2>&1 &
```

Monitor:  `tail -f data/crawl_2020_2026.log` ·  ok/skip counts:
`grep -c '^\[ok\]' data/crawl_2020_2026.log ; grep -c truncated data/crawl_2020_2026.log` ·
disk: `df -h /`. Enumeration (~12 min for Block A) runs before the first download.

**Robustness:** resumable (done-set from `state/*.ok` + manifest at startup); `requests.Session`
+ retry/backoff (429/5xx, honor `Retry-After`, honest `User-Agent` + contact email); atomic
`os.replace` for all outputs; one bad session logs to `errors.jsonl` and never aborts the crawl.
Run **year-by-year**.

## Storage

Measured **1.40 TB free** (≈1.5 TB after the `exp/` prune below). 11.7k h as 24 kHz WAV ≈ 2.0 TB
(infeasible); as **24 kHz mono FLAC ≈ ~1.1 TB → fits with headroom**, acoustic-grade and reusable
for a later OmniVoice acoustic fine-tune. Video is deleted per-session (peak transient ~1–2 GB,
never cumulative). Safeguards: `--min-free-gb` guardrail, crawl year-by-year and re-measure FLAC
size after year 1 (single-speaker + pauses likely compress below 1.1 TB).

> **Verified on the pilot (ID 2694, 112.7 min):** the FLAC estimate holds **only at 16-bit**.
> ffmpeg decodes the source AAC to float, so without `-sample_fmt s16` FLAC stores **24-bit**
> (`sample_fmt=s32`, `bits_per_raw_sample=24`) — bits 17–24 are lossy-codec reconstruction noise
> that barely compresses, ballooning the file to **410 MB/h → 4.8 TB** (would NOT fit). With the
> `-sample_fmt s16` extraction (now in `ffmpeg_extract`), a direct AAC→16-bit FLAC is **89 MB/h →
> ~1.04 TB** for 11.7k h, confirming the estimate. 16 bit / 24 kHz mono is acoustic-grade and is
> exactly what OmniVoice + the Higgs tokenizer consume, so the dropped bits cost no usable quality.

### `exp/` prune (done before the crawl to reclaim space)
- Backed up the winner: `exp/omnivoice_bg_combined/checkpoint-2000` → `models/omnivoice-bg-v2/`.
- Deleted superseded runs `exp/omnivoice_bg` (35 GB) and `exp/omnivoice_bg_marked` (28 GB).
- Stripped `optimizer.bin` (4.6 GB each) from `bg_keep/*` and non-winner combined checkpoints;
  kept `checkpoint-2000` fully intact. `exp/` 139 GB → ~30 GB.

## Verification

1. [x] Pilot: `python prep_bg_parliament.py --out data/_smoke --date-start 2012-01-27
   --date-end 2012-01-27 --limit 1` → got ID 2694, 2 parts, `ok=1 err=0`.
2. [x] Assert `ffprobe` → `sample_rate=24000, channels=1` ✓; FLAC duration 6761 s ≈ Σ parts ✓;
   transcript starts with spoken text (label stripped) ✓; turns plausible (212 turns) ✓; `raw/`
   empty (MP4s deleted) ✓. **Found + fixed the 24-bit FLAC bloat** (see Storage note).
3. [x] Offline unit-test `bg_sten_clean.clean_body` on 2694 body → 205 406 clean chars, 212 turns,
   3 headers, top speaker = chair (Цецка Цачева, 65 turns), roles parsed.
4. [ ] Downstream smoke through `omnivoice/scripts/extract_audio_tokens.py` confirms the manifest is
   consumable. (Manifest schema is identical to `data/bg`/`data/bg400`, which already tokenized
   cleanly; a literal run is only meaningful **after** the next-phase segmentation, since
   `extract_audio_tokens --max_length 25` drops a whole 112-min session.)
5. [ ] Then `--limit 5` across years (varying `Vicount`, a single-part, a missing-video case) before
   the full year-by-year crawl.

## Risks

- **Bandwidth/time/IP-ban:** ~1.5 TB of MP4 downloads + ~1500 API calls → `--sleep`, backoff,
  honest UA, full resumability so the crawl can be throttled/paused.
- **Stenographic ≠ verbatim:** next-phase alignment must tolerate insert/delete.
- **Multi-part concat:** sort by `playlist.item`, assert `count == Vicount`; a short/failed part
  fails the session rather than silently truncating.
- **Missing video:** null `default`/empty `playlist` → skip + log (paired output needs audio).
- **Enumeration non-monotonicity:** IDs are only PIECEWISE monotonic (modern runs interleaved with
  historical re-imports), so binary search is UNSAFE — `enumerate_sessions` linear-scans the guarded
  range + date-filters (fixed 2026-07; drive with explicit `--id-start/--id-end` per year).
- **Server-truncated / malformed video (MAJOR, early years):** many 2010–2012 recordings are
  incomplete on the server — the file's own `mdat` box declares more bytes than the served
  Content-Length, or the file doesn't even start with `ftyp` (garbage header) → no `moov` index →
  `ffmpeg 'moov atom not found'`. `probe_mp4_complete()` detects this from a **64 KB prefix** (require
  `ftyp`; accept faststart `moov`; else verify a real `moov` follows `mdat` within the file) and skips
  the session for ~64 KB instead of a wasted ~450 MB download. Downloads also loop-until-complete
  (resumable, exact Content-Length) to defeat transient mid-stream truncation of *healthy* files.
  **Measured corruption is concentrated in early years:** 2012 ≈ 23% good; 2014/2015/2016 ≈ 100% good
  (sampled). The bulk of the corpus (2013–2018) is clean; skips are cheap, so yield loss ≠ bandwidth loss.

## Prior art & reuse — Finnish Parliament ASR corpus (Virkkunen et al., 2022)

`2203.14876v1.pdf` is the published reference implementation of this exact pattern (parliament
video + lightly-edited transcripts → ASR corpus). Read it before building Phase 2 — it lets us
**reuse, not rebuild**. Our Phase 1 already mirrors its first three pipeline boxes (video DL →
wav extraction → transcript parse/preprocess).

**Reuse instead of building:**
- **[`aalto-speech/fi-parliament-tools`](https://github.com/aalto-speech/fi-parliament-tools)** —
  full retrieval + processing pipeline (incl. segmentation + speaker matching). Template for the
  Phase-2 half we haven't built.
- **[`aalto-speech/fin-parl-models`](https://github.com/aalto-speech/fin-parl-models)** — tuned
  HMM-GMM / TDNN-F / AED recipes (GMM optimised over 27 runs).
- **Kaldi `steps/cleanup`** segmentation (wsj/s5) **is our Phase 2**: force-aligns long audio to the
  edited transcript and *discards what won't align* — the sanctioned way to handle stenographic ≠
  verbatim. Basis: Manohar et al. 2017 (JHU Kaldi MGB-3, diarization + audio-transcript alignment).
- Acoustic model: Kaldi Librispeech `run_tdnn_1d.sh` (TDNN-F, LF-MMI). Subword LM: VariKN +
  SentencePiece. The **final HMM-GMM produces phone-level alignments** — exactly what the Accentor
  needs to decode lexical stress from speech.

**Bulgarian shortcut (check before crawling 11.7k h):** the paper cites
**Geneva, Shopov, Mihov (2019), "Building an ASR corpus based on Bulgarian Parliament speeches"**
(249 h, 572 speakers) — *both* the Accentor team's own corpus *and* an already-built BG-parliament
ASR corpus. Obtain/inspect it first: a ready validation set + worked BG segmentation reference, and
possibly stress annotations.

**Planning numbers (measured by the paper, reuse directly):**
- **Yield ≈ 73 %** of raw audio survives segmentation + cleanup (22 % lost: muted/untranscribed/
  unalignable; 5 % in speaker matching).
- **86 %** of segments are ≤ 15 s (Kaldi default max-segment).
- **Saturation ≈ 1 800 h**: in-domain ASR plateaus there — more parliament data only helped
  *out-of-domain*. So the alignment/ASR engine needs only a **subset**, not all 11.7k h. (The full
  crawl still serves acoustic-FT + maximal stress coverage; we just don't have to *process* it all.)
- ASR uses **16 kHz** mono; we keep 24 kHz and downsample a working copy for alignment.

**Nuance — our two consumers want different segmentation:**
- *Accentor stress labels* → **phone-level forced alignment** (HMM-GMM alignment output) ✓ reuse.
- *TTS acoustic fine-tune* → **clean sentence-unit** segments, NOT Kaldi's mid-sentence 15 s chunks
  (punctuation stripped, UNK tokens inserted). Use the `turns/` speaker-turn anchors + sentence
  splitting for the TTS branch; reuse only the *alignment*, not the ASR-style chunking.

## Missing dependencies & open gates (Phases 2+)

Source: the Accentor paper (`interspeech2023.pdf`, Geneva et al., Interspeech 2023) now gives the
**full methodology** for the stress front-end. What remains is **data + compute**, not knowledge.
Resolve the make-or-break item (the lexicon) before committing to Phase 2's annotation step.

**Scope decision (recorded):** reproduce **only Phases 1–2 (the Accentor)**; its output is text with
explicit U+0301 stress marks — exactly what our OmniVoice marked-finetune front-end consumes. The
Accentor **replaces `rechko`** as the stress front-end *and* re-marks our finetune data with
contextually-correct stress (the proper fix for the stress-mark training mismatch). **Drop
StreamSpeech [27] + LPCNet [30]** (paper's §5 backend) — OmniVoice is our acoustic model + vocoder.

| # | Missing dependency | What it's for | Spec (from the paper) | How to get it |
|---|---|---|---|---|
| 1 | **287K-type canonical multi-stress BG lexicon** + stressed/unstressed **phonetic inventory** | alignment lexicon **and** building per-sentence stress-decoding graphs | 287K word types, 99% token coverage, **all canonical phonetizations + stress patterns per word form**; extended +551K unambiguous → 829K (§3.2–3.4) | **DECIDED: build our own** (2026-06-30) — extend `rechko` to all candidate stresses + add phonetization. See **"Building the canonical-stress lexicon"** below. |
| 2 | **Stressed/unstressed-vowel TDNN annotation AM** | decode audio → ground-truth stress labels | Kaldi + LibriSpeech TDNN recipe, phonetic system with separate stressed/unstressed vowel phonemes, trained on **749 h** (BG-PARLAMA + aligned audiobooks) (§3.3) | train ourselves with Kaldi (reuse `fin-parl-models` recipes); needs item 1's phone set + aligned speech. |
| 3 | **BG-PARLAMA corpus [18]** (Geneva, Shopov, Mihov 2019) | training data for the annotation AM (item 2) | the authors' BG parliament ASR corpus | request/obtain; **our own parliament crawl can substitute/supplement** (this doc's Phase 1). |
| 4 | **Verbalization + phoneme-Levenshtein aligner** [19] | align edited transcripts, handle numbers/dates/abbrev. | phoneme-level Levenshtein over all verbalizations; keep sentences where audio **completely matches** (§3.2) | method described in Geneva & Shopov 2019; reimplement or reuse `fi-parliament-tools` alignment + a BG verbalizer. |
| 5 | **Compute** | Accentor + annotation AM training | Accentor: 6.5M params, batch 512, **4.3M iters ≈ 6 days × 1×V100**; annotation AM: Kaldi TDNN on 749 h | our 16 GB consumer GPU is slow for 6×V100-days — plan a rented V100/A100 or a long local run. |
| 6 | **Accentor model impl** | the front-end itself | 6 FFT blocks, dim 256, 4×128 heads, conv k=9/1, depthwise-sep convs, BCE loss, vocab 41 (§4) | implementable now from the spec (FastSpeech2-style FFT blocks); no external dep beyond items 1–3 for *data*. |

**Aligned-data yield to plan for:** raw 11.7K h → **~7,970 h aligned** in the paper (68%; matches the
Finnish paper's ~73%). Corpus split (Table 2): Parliament 2,695 h / Audiobooks 5,275 h.

**Critical path:** item 1 (lexicon) gates items 2 and 4; without it neither the annotation AM nor the
decoding graphs can be built. Recommend resolving item 1 (request-vs-build) as the **next gate**,
in parallel with finishing the Phase-1 crawl.

## Building the canonical-stress lexicon (gap item 1 — DECIDED: build)

The make-or-break asset. We build it rather than request it from the authors.

### Starting point (measured 2026-06-30)
`build_bg_lexicon.py` already turns the rechko MySQL dump (`../rechko/db.sql`, 730 MB) into
`data/bg_lexicon/bg_stress.sqlite`: **2.03M surface forms** = lemma stress (Phase 1) + stress
*projected* onto inflected forms by vowel ordinal (Phase 2) + an imperative penult rule. Schema
`stress(name, stressed, src, mobile, UNIQUE(name, stressed))` is **already multi-candidate-capable**.

**The gap (why rechko alone is insufficient):** it stores **one canonical stress per surface form**,
so the homographs that motivate the whole Accentor are collapsed — `замък`→only `за́мък` (missing
`замъ́к`), `вълна`→only `въ́лна` (missing `вълна́`), `тела`→only `те́ла`, `коса`→only `коса́`. The 7,956
forms with >1 candidate are mostly compound primary/secondary-stress artifacts, **not** real
homographs. Two further gaps: **no phonetization** (only stress-marked spellings, no phonemes), and
**projection is wrong for mobile-stress cells** (flagged `mobile=1`).

### Key reframe — the lexicon needs RECALL, not precision
Its two consumers — the DNN-HMM **aligner** and the per-sentence **stress-decoding graph** — both pick
the acoustically-best path. They only need the **true stress to be *among* the candidates**; the
stressed/unstressed-vowel acoustic model chooses from the audio. So we can **over-generate** stress
candidates for ambiguous forms without knowing which is correct — this de-risks both the
homograph-collapse and mobile-stress problems (the *annotation* output, not the lexicon, is the
precise artifact).

### Build layers
- **L0 (have):** rechko canonical layer — primary stress per surface form (`build_bg_lexicon.py`).
- **L1 — homograph / multi-candidate generation (new, the hard part):**
  rechko is **limited** (measured): one stress per surface form, and lexically-distinct homographs
  (`вълна́`, `замъ́к`, `пара́`) are **absent entirely** — so rechko-internal recovery can't find them.
  Candidates must come from over-generation + external data. **Blanket over-generation is infeasible:**
  avg **5.77 vowels/form** (most forms 4–8 vowels) → all-positions would explode the graph and poison
  labels. So the policy is **bounded + targeted**:
  - (a) **rechko canonical** kept as the default single candidate for the majority — especially long
    forms (≥4 vowels), which are ~never stress-ambiguous; keeps the graph tight and labels clean.
  - (b) **targeted alternatives** only for the ambiguity hotspots: **short content words (2–3 vowels)**
    — the замък/вълна/пара class, ~28K two-vowel forms — plus **functional words** and **verb
    imperative(-ете/-йте)/3sg** pairs.
  - (c) **curated homograph seed** for the lexically-distinct pairs rechko lacks: `config/bg/
    homograph_battery.json` (vetted; expand from **Bulgarian Wiktionary** stress-marked inflection
    tables — the external source that fills rechko's gap).
  - (d) **mobile-stress inflections** (derivative_form flagged `mobile=1`): add the shifted-stress
    alternative since rechko's projection is unreliable there (`тела́` case).
- **L2 — G2P phonetization (new):** surface form + stress → phoneme string with **separate
  stressed/unstressed vowel phonemes** (rule-based; BG is near-phonemic). Defines the phonetic
  inventory. Emits a **Kaldi `lexicon.txt`** (word → phones, multiple lines per ambiguous word).
- **L3 — coverage + OOV:** measure token coverage on the parliament + audiobook corpus, target
  **≥99%**; numbers/dates via a verbalizer; OOV / foreign via G2P fallback + a casing/Latin-script
  signal (ties to ablation A4).

### Outputs
1. **Kaldi pronunciation lexicon** (`lexicon.txt`) + **`phones.txt`** (stressed/unstressed vowels) —
   alignment + decoding graphs.
2. **Stress-candidate map** (surface form → list of stress-marked spellings) — decoding-graph
   construction and text re-marking.

### L3 status (2026-06-30): coverage + OOV done — lexicon BUILT
`lexicon_coverage.py` then `extend_lexicon_coverage.py` (+ `config/bg/lexicon_supplement.json`).
Measured on the audiobook corpora (`data/bg` + `data/bg400`, 578K tokens; parliament folds in later):
the stress lexicon alone covered only **57%** (it stores only *marked* forms — unmarked function
words/monosyllables excluded by design). Extending to the **full rechko vocab** (+497K forms) +
a curated **съм-paradigm / в / с supplement** (irregulars absent from rechko) →
`lexicon_full.txt` (**2.53M surface forms**). Result: **94.74% known-canonical-stress**,
**99.99% effective pronunciation coverage** (lexicon ∪ rule-based G2P fallback — any Cyrillic string
phonetizes), **0 truly unhandled**. Residual ~5% = proper names/loanwords (the paper's own dominant
residual, §6) → G2P fallback + over-generated stress; Latin 0.007% → transliterate. Target ≥99% met.

### Lexicon artifacts (gap item 1 — DONE)
| File | Layer | Contents |
|---|---|---|
| `data/bg_lexicon/bg_stress.sqlite` | L0 | rechko canonical (2.03M marked forms) |
| `data/bg_lexicon/bg_stress_candidates.sqlite` | L1 | multi-candidate stress (4.16% multi; battery recall 10/10) |
| `data/bg_lexicon/lexicon.txt` | L2 | marked-form pronunciations (2.16M, multi-pron homographs) |
| `data/bg_lexicon/phones.txt` | L2 | 34-phone inventory (12 stressed/unstressed vowels + 22 cons) |
| `data/bg_lexicon/lexicon_full.txt` | L3 | full alignment lexicon (2.53M forms, 99.99% effective) |
| `config/bg/homograph_battery.json` | L1 | vetted homograph validation set + seed |
| `config/bg/lexicon_supplement.json` | L3 | irregulars absent from rechko (съм paradigm, в, с) |

Scripts: `build_stress_candidates.py` (L1) · `build_lexicon_phones.py` (L2 G2P) ·
`extend_lexicon_coverage.py` (L3) · `lexicon_coverage.py` (measurement).
**Remaining (data, not code):** expand the homograph battery from Wiktionary (task #7) — raises
known-stress recall on the long tail; the generators already merge whatever it holds.

### Validation
- **Recall** on a homograph battery: does the true stress appear among candidates for
  замък/вълна/тела/коса/пара/брава…? (today: no.)
- **Coverage** % of corpus tokens.
- **G2P** spot-check against known phonetizations.

> Reuse: `build_bg_lexicon.py` rechko parser + the multi-candidate schema. New work = L1 (homograph
> generation) + L2 (G2P). Recall-oriented, so over-generation is acceptable and expected.

### L1 status (2026-06-30): generator built + validated
`build_stress_candidates.py` reads `bg_stress.sqlite` → `bg_stress_candidates.sqlite`
(`candidates(name, mark, source)`). Run: 2.03M forms → 2.16M candidate rows, **avg 1.066 cands/form**,
**4.16% multi-candidate** (vs the paper's ~4.3% true ambiguity — well-calibrated), sources
rechko=2.04M / overgen2(2-vowel)=28K / mobile(≤3-vowel)=98K / battery=5. **Battery recall: 10/10
PASS** — every vetted homograph now carries both readings. Mobile policy tightened to short forms only
(long mobile forms keep canonical — widening them was pure noise: 35.6%→4.16%).
**Remaining L1 (data, not code):** expand `config/bg/homograph_battery.json` (5 → real coverage) from
**Bulgarian Wiktionary** stress-marked inflection tables + functional-word / verb imperative-vs-3sg
pairs. The generator already merges whatever the battery contains.

### L2 status (2026-06-30): G2P + lexicon built
`build_lexicon_phones.py` (rule-based BG G2P) → `data/bg_lexicon/lexicon.txt` (**2.16M
pronunciations**, 132 MB) + `phones.txt` (**34 phones**: 12 stressed/unstressed vowels + 22
consonants). Each L1 stress candidate → one pronunciation line, so homographs are multi-pronunciation
(`вълна` → `v y1 l n a0` / `v y0 l n a1`) — the aligner + decoding-graph input. Handles iotation
(я/ю/й/ьо as a j-glide), digraphs (дж/дз/щ), affricates (ц/ч), and **voicing assimilation + final
devoicing** incl. the **в-non-trigger exception** (`сватба`→`svadba`, `град`→`grat`). Verified on a
20-word self-test (`--self-test`). *Approximation to revisit:* palatalization modeled as a /j/ glide
rather than palatalized consonant phones — fine for alignment/stress; refine later if needed.

## Accentor architecture — considerations & ablation plan

The paper's Accentor (gap item 6: 6 FFT blocks, dim 256, 4×128 heads, depthwise-sep convs, vocab 41,
per-vowel BCE, 6.5M params) is **sound and validated** (real-world SER ~0.37%, 12× better than
end-to-end). **Reproduce it first as the baseline**, then run the improvements below as ablations
*against* it — don't replace the paradigm blind.

**Two framing facts that set priorities:**
- Residual errors are **data, not capacity** (§6): mostly foreign names + loanwords under-represented
  in training. No architecture change fixes that — coverage does.
- Depthwise-sep convs and the 6.5M cap exist only for the paper's **on-device StreamSpeech** backend.
  We feed OmniVoice and run the Accentor **once at the front-end, offline** → no latency/size
  constraint → free to trade size for accuracy.

### Ablation matrix (each vs. the reproduced baseline)

| # | Change | Hypothesis | Risk / caveat | Priority |
|---|---|---|---|---|
| A0 | **Baseline** — paper-exact | reference SER to beat | none (validated) | **do first** |
| A1 | Drop efficiency constraints — standard convs, larger dim / more FFT blocks, lift 6.5M cap | cheap accuracy; the LightSpeech compromises are pure downside for us | more compute/params (fine — front-end only) | **high (cheap win)** |
| A2 | **Pretrained BG/multilingual subword encoder** (token-classification head) instead of from-scratch FFT blocks | subword priors help the long tail (homographs, loanwords, names) — exactly the failure mode | with 70M auto-labeled tokens, from-scratch is already strong → gain may concentrate in the tail; real A/B needed | **high (biggest lever)** |
| A3 | One-stress-per-content-word structural prior (per-word softmax or light CRF) vs. free per-vowel BCE | enforces BG's single-primary-stress rule; fixes 0-/2-stress inconsistencies | clitics, compounds, unstressed function words break strict one-hot → use a **soft** constraint; verify no regression on those classes | medium |
| A4 | Loanword/foreign-name coverage: extend the +551K dict augmentation with a foreign pronunciation+stress set; keep **casing feature** + Latin-script fallback | attacks the actual dominant error source directly | needs a foreign-name/loanword stress source | **high (targets real errors)** |

### System-level constraints (hold across all ablations)

- **Consistency > absolute SER.** Whatever Accentor we pick must mark the **OmniVoice finetune corpus
  with the same model** so train-time and inference-time stress conventions agree — this is the root
  cause behind the stress-mark training mismatch. A slightly-less-accurate-but-consistent Accentor on
  *both* sides beats a great one on only one.
- **Evaluate on what matters to us**, not the paper's auto-SER (0.119%, self-referential — can inherit
  decoding errors). Metric = a **human-annotated held-out BG set + the TrainGen hard-word/homograph
  battery run through OmniVoice end-to-end**.

> All ablations are **downstream of gap item 1** (the 287K canonical-stress lexicon): no lexicon → no
> auto-annotated labels → nothing to train or compare. Architecture is the fun question; the lexicon
> is the gate.

## Handoff to the next phase

This phase stops at per-session pairs. The next phase (segmentation → forced alignment → stress
annotation) consumes `manifest.jsonl` (session rows: FLAC `audio_path` + `text` + `duration` +
`turns_path`), the `turns/` files (speaker-turn boundaries as alignment anchors), and
`omnivoice/utils/audio.py` for chunking. No alignment or annotation is performed here. See
**Prior art & reuse** above for the tooling to adapt rather than write from scratch, and **Missing
dependencies & open gates** for what Phases 2+ still need (chiefly the 287K canonical-stress lexicon).
