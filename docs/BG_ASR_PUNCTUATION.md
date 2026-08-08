# Punctuation, casing and number formatting for BG ASR output

**Status: options analysed, nothing built (2026-08-08).** CPU-only work — can proceed while
GP-S occupies the GPU.

## The situation: there is nothing to "enable"

GP-v2 cannot emit punctuation. This is not a decoding flag, it is the **vocabulary**: the
BPE-500 table (`models/asr/gp-v2/tokens.txt`, 502 entries) contains Cyrillic word pieces, `-`,
and the meta tokens `<blk>/<sos-eos>/<unk>/#0/#1` — and **zero punctuation characters**.

That follows from the training targets, which went through `bg_text_norm.normalize`: lowercase,
punctuation stripped, U+0301 stripped, numbers verbalized. So raw output looks like

> `така че тук не става въпрос за натиск от президента`

and a date comes out as `две хиляди и седем`, not `2007`.

Three separate things are therefore missing, and they are normally solved by **one
post-processing stage**:

1. **Punctuation** — `.` `,` `?` `!`
2. **Truecasing** — sentence starts, proper nouns
3. **Inverse text normalization (ITN)** — `две хиляди и седем` → `2007`, `деветдесет и осем
   лева` → `98 лв.` We have `bg_verbalize` for digits→words; **the inverse does not exist yet.**

## Why this is safe to add

`bg_text_norm.normalize` is applied to hypothesis *and* reference in every eval, and it strips
punctuation and case. So a formatting layer **cannot change any WER number** in
`BG_ASR_BENCHMARK.md`, and cannot disturb the frozen-test discipline. It is a pure readability
layer, addable and removable at will.

Where it matters: TrainGen narration and any transcript a human reads. Where it does not: the
playground, which scores normalized text either way.

## The constraint that shapes the choice

From `BG_ASR_DATA_PIPELINE.md`:

> **No LLM rewriting of labels, ever** — labels are kept faithful by *filtering*, never made
> fluent by editing (an LLM that can't hear the audio drifts text away from what was said).

Punctuating an ASR transcript with an LLM is exactly that risk: the model cannot hear the
audio, and while fixing commas it will also silently "fix" words it finds implausible.

**The normalizer defuses it.** Accept a formatter's output only when

```python
normalize(formatted) == normalize(asr_output)     # bg_text_norm
```

Equality proves that **only punctuation, casing and number formatting changed** — any word
substitution breaks it. It even permits ITN, since `normalize` maps `2007` and
`две хиляди и седем` to the same string. This converts "trust the LLM" into "verify the LLM",
which is the project's existing filtering philosophy rather than an exception to it.

**Any option below should ship behind this guard**, with the unpunctuated text as fallback.

## Options

| | Quality | Latency | Can corrupt words? | Cost to build | Needs GPU |
|---|---|---|---|---|---|
| **A. Token classifier** | high | ms (CPU) | **structurally impossible** | few GPU-hours | to train |
| **B. BgGPT + guard** | highest | slow | possible → caught by guard | none (tooling exists) | to run well |
| **C. KenLM n-gram** | weakest | ms (CPU) | no | a CPU afternoon | no |
| **D. Retrain ASR with punctuation** | — | — | — | full training run | yes |

### A. Token-classification encoder — best end state

A multilingual encoder (XLM-R base or similar) with a per-token head predicting the label that
follows each token: `NONE / COMMA / PERIOD / QUESTION / EXCLAM`, plus a parallel casing head
(`lower / Title / UPPER`). Standard architecture for this task.

Its decisive property: it **assigns labels to existing tokens and cannot generate new ones**, so
word corruption is impossible by construction rather than by check. Fast enough for the live
playground path.

Training data is abundant and already ours — see below. Cost is a few GPU-hours, so it waits
for GP-S.

### B. BgGPT + the equality guard — best quality without training

Tooling already exists in the repo (`bggpt_teacher.py`, `bggpt_validate.py`,
`adapt_teacher_rows.py`, `data/bggpt_teacher/`). Prompt it to insert punctuation and casing and
change nothing else, then enforce the equality guard and fall back to raw text on mismatch.

Right choice for **batch** formatting (finished transcripts, narration scripts). Too slow for a
per-request path. Log the guard's rejection rate — a high rate is itself the signal that the
prompt is letting the model rewrite.

### C. KenLM n-gram — cheap, honest baseline

Train an n-gram on punctuated text with punctuation as tokens, then insert punctuation to
maximize LM probability. Reuses `build_kenlm_bg.sh` and needs no GPU.

More viable for Bulgarian than English intuition suggests: comma placement here is unusually
**mechanical** — before `че`, `който`, `да`-clauses, `но`, `защото` — and that is local context,
exactly what an n-gram sees. It will not match a transformer, but it establishes a floor and
costs almost nothing.

### D. Retrain the ASR with punctuation in the targets — not recommended

Needs a new BPE containing punctuation and a re-normalized corpus, and it **breaks comparability
with every number in the benchmark**. Punctuation from audio alone is also genuinely hard —
prosody helps but is unreliable — so you would pay WER for a job a text model does better. Note
the roadmap already routes punctuation elsewhere: the phase-2c Whisper-BG LoRA is specified to
train on *raw punctuated* text precisely to preserve that ability in the Whisper track.

## Training data — already on disk

Raw Whisper output, before normalization, is punctuated **and** carries digits:

| Source | Punctuated text |
|---|---|
| `data/ytdr/pseudolabels.jsonl` | 16.1 M chars |
| `data/bnr/pseudolabels.jsonl` | 5.9 M chars |
| `data/ytgi/`, `data/ytbb/` | more |
| parliament stenograms (pre-normalization), chitanka | more again |

Sample: `Почнах като инженер, после 1993 г. станах началник КАТ.` — punctuation, casing and a
digit-year in one line, i.e. supervision for all three sub-tasks at once. Build pairs by
stripping punctuation/case from this text and predicting what was removed.

## Evaluation — the trap to avoid

**Do not evaluate against Whisper's punctuation.** Those 22 M chars are excellent *training*
data (abundant, in-domain), but as ground truth they measure *agreement with Whisper*, not
correctness.

Use **human-punctuated** Bulgarian for the eval set:
- **parliament stenograms** — written by human stenographers, and in-domain
- **chitanka** — professionally edited books, out-of-domain prose as a second register

Method: hold the eval text out of training, strip punctuation and case, restore, and report
**per-class precision/recall/F1** for `. , ? !` plus a casing accuracy. Per-class matters
because commas are ~10× more frequent than question marks; a single accuracy number hides
total failure on the rare classes.

Also report the **guard rejection rate** for option B, and sanity-check on real ASR output —
eval text is clean, whereas actual transcripts contain the model's own errors, and a formatter
that only works on clean text is not the one you need.

## Suggested order

1. **Now (CPU, GPU busy):** build the human-punctuated eval set from stenograms + chitanka, and
   the KenLM baseline (C). Establishes a measurable floor before spending anything.
2. **Then:** run BgGPT (B) with the guard over the same eval set and compare. If it wins clearly
   and you only need batch formatting, stop — no training required.
3. **After GP-S:** train the token classifier (A) only if punctuation is needed in the live
   playground path, where LLM latency is unacceptable.
4. **Separately:** ITN. Worth writing as a rule-based inverse of `bg_verbalize` rather than
   learning it — the mapping is deterministic in the direction that matters, and a rule engine
   cannot invent numbers that were never spoken.
