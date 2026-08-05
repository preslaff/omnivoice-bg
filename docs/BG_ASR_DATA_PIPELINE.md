# BG ASR data-preprocessing pipeline & corpus balancing

**Status:** written 2026-08-05, mid-campaign (BNR cuts + DnevenRed transcribe running).
The end-to-end record of how every training hour is produced, filtered, and balanced for the
**general-purpose** model (GP-v2). Live corpus composition: run **`docs/BG_ASR_Corpus.ipynb`**
(re-runnable — reads the manifests on disk). Companions: [BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md)
(sequencing) · [BG_ASR_BNR_PLAN.md](BG_ASR_BNR_PLAN.md) (harvest commands) ·
[BG_ASR_V2_PLAN.md](BG_ASR_V2_PLAN.md) (parliament recovery) · [BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md).

## Shared invariants (every axis ends the same way)

- **Features:** 80-dim log-mel fbank @ 16 kHz (`Fbank(FbankConfig(num_mel_bins=80, sampling_rate=16000))`),
  LilcomChunkyWriter shards + a lhotse cuts manifest `data/fbank*/bg_cuts_<axis>.jsonl.gz`.
- **Text normalization:** ONE normalizer everywhere — `bg_text_norm.normalize` =
  `bg_verbalize` (gender-aware number/date/currency expansion) + lowercase + strip punctuation
  + strip U+0301. Applied to refs and (in evals) hypotheses identically.
- **No LLM rewriting of labels, ever** — labels are kept faithful by *filtering*, never made
  fluent by editing (an LLM that can't hear the audio drifts text away from what was said).
- **Leakage guard at merge:** `merge_train_corpus.py` drops any train cut whose cut-id or
  recording-id appears in a `--holdout` eval manifest (frozen parliament dev/test + FLEURS/CV
  test), and dedups cut-ids across sources.

## The axes

### A. Parliament + AB400 (v1 gold, 649.2 h)
crawl (`prep_bg_parliament.py`) → Kaldi GMM chain: `segment_long_utterances` (tri2b_comb
biased-LM) → `align_fmllr` (tri4b) → `clean_and_segment_data` → `prep_lhotse_manifests.py`
(seekable file-sources, U+0301/`<unk>` strip, **frozen whole-session split**: train 419,915
cuts / dev 6.2 h / test 16.4 h) → `compute_fbank_bg.py` (20 resumable chunks).
*Labels: human stenograms + audiobook scripts, forced-aligned — the gold anchor.*

### B. Parliament recovery (+334.4 h train-only)
`detect_stale_sessions.py` quarantined 110 sessions → root cause: **byte-concatenated
multi-part FLACs with lying STREAMINFO headers** (audio intact; header-seeking readers saw
only part 1) → `repair_flac_headers.py` (lossless re-encode, verified vs Kaldi utt2dur) →
`resegment_stale_sessions.sh` (same GMM chain, fresh `parl_restale` workspace; 110/110
recovered, 93.9 % cleanup retention) → recovery-only manifests (`data/lhotse/bg_reseg`) →
`compute_fbank_bg.py --out-dir data/fbank_reseg`. *Labels: stenograms, forced-aligned (gold).*

### C. BNR radio archive (~103 h)
`bnr_crawl.py` (sitemap → `__NEXT_DATA__` → `/api/media/<Id>`; 3,006 mp3s / 13 GB ≈ 220 h) →
`bnr_pseudolabel.py`:
1. **`segment`** — inaSpeechSegmenter speech/music regions: the **musical-play quarantine**
   (metadata insufficient; plain VAD fails since singing = "voice"). Kept 126.7 h speech (58 %).
2. **`transcribe`** — faster-whisper large-v3 per region + **faithfulness filter**
   (lang-id ≥ 0.8, avg_logprob ≥ −0.7, no_speech ≤ 0.5, compression ≤ 2.4) → 87.8 h accepted.
3. **Audit (3 tiers,** `audit_pseudolabels.py`**)** — stats; Zipformer cross-decode of 500
   accepted windows (24.9 % inter-model disagreement ⇒ implied label error ~single-digit);
   native listening sheets.
4. **Recovery** — `bnr_recover_rejects.py`: `lang=bg<0.8` rejects (archival audio depresses
   language-id) re-arbitrated by the v1 Zipformer; windows with inter-model WER ≤ 0.5 flipped
   to accept → **+15.1 h at two-teacher quality** (79 % of candidates).
5. **`cuts`** → `bg_cuts_bnr.jsonl.gz`. *Labels: Whisper weak supervision, confidence-filtered
   + partially two-teacher.*

### D. YouTube — DnevenRed (368 h) & Bloomberg BG (149 h)
`yt_harvest.py` (audio-only m4a + Bulgarian **auto**-caption vtt; `--min-duration 60` skips
shorts; enumeration capped for huge channels; `--cookies-from-browser` ready for members-only
top-up) → same `bnr_pseudolabel.py` stages with:
- **stage 1 in `--limit 40` batches** (Keras predict-loop leak grows with audio processed —
  OOM-killed once at 47 GB RSS on hour-long files); DnevenRed kept 328.9 h speech (89 %).
- **stage 2 `--agree-subs`** — the **two-teacher agreement filter**: the YT auto-caption is an
  independent ASR; windows are kept only when Whisper and the caption agree (normalized
  inter-teacher WER ≤ 0.5). Independent errors rarely coincide ⇒ higher precision than
  confidence alone. Captions are never treated as gold.
*Labels: Whisper × YT-caption agreement (strongest weak-supervision tier).*

### E. FLEURS + Common Voice (read speech, ~25 h train)
Direct-parquet preps (`prep_fleurs_bg.py`, `prep_commonvoice_bg.py` — bypass datasets-5.0
resolution hang; FLEURS cut-ids keyed on unique `path`, not the repeating sentence `id`).
**Train** splits go to training; **test** splits stay held-out eval sets (they are the
pre-diversification OOD baselines: 20.35 / 22.27 to beat).

## Quality gates (summary)

| Gate | Axis | Catches |
|---|---|---|
| Kaldi cleanup (junk ≤ max-proportion) | A, B | transcript/audio mismatch in gold data |
| FLAC header repair + utt2dur verify | B | lying containers |
| Speech/music segmentation | C, D | music, musical plays, jingles |
| Language-id + confidence + compression | C, D | foreign speech, mishearings, hallucination loops |
| Caption agreement (`--agree-subs`) | D | any single-teacher error |
| Zipformer arbitration (`zwer ≤ 0.5`) | C recovery | degraded-audio false rejects/accepts |
| 3-tier audit (stats / cross-decode / listening) | C (template for D) | threshold calibration, systematic label error |
| Merge leakage guard + dedup | all | eval contamination, double-counting |

## Data balancing

**Why balance:** the transducer learns an implicit LM from the training mix — a register that
dominates the corpus biases decoding priors everywhere (e.g. financial phrasing). Working rule:
**keep registers within ~0.5–2× of each other**; more of one register saturates fast
(diversity-per-hour decays — a channel is few voices in one studio).

**Mechanism 1 — harvest caps (applied).** Bloomberg capped at `--limit 500` (~149 h) though the
archive holds thousands of hours; BNR/DnevenRed taken whole (their sizes already fit the band).

**Mechanism 2 — natural-mix merge (GP-v2 choice).** `merge_train_corpus.py` concatenates the
axes as-is. Approximate GP-v2 composition (post-filter; live numbers in the notebook):

| Register | Source(s) | ~h | share |
|---|---|---|---|
| Formal/spontaneous parliament | A-parl + B | ~596 | ~38 % |
| Read literary | A-AB400 | ~397 | ~25 % |
| Conversational political talk | DnevenRed | ~250–300 (post-agree) | ~18 % |
| Broadcast news/business | Bloomberg | ~100–130 (post-agree) | ~7 % |
| Archival spoken-word | BNR | ~103 | ~7 % |
| Read modern (wiki/crowd) | FLEURS+CV train | ~25 | ~2 % |

Parliament-family (~38 %) and the max/min spread (~parl 596 vs BNR 103 ≈ 6×) stretch the band,
but GP-v2 deliberately keeps the natural mix: it isolates the **data effect** vs v1 (one
variable at a time — see the roadmap's model ladder), and gold-labeled hours deserve weight.

**Mechanism 3 — train-time mux weights (the GP-v3 lever, no data duplication).** If GP-v2's
eval shows a register under-served, rebalance at the datamodule with lazy mux instead of
re-merging:

```python
from lhotse import CutSet
sources = [CutSet.from_file(p) for p in per_axis_manifests]
hours   = [sum(c.duration for c in cs)/3600 for cs in sources]
# e.g. equalize toward target shares t_i:  w_i ∝ t_i / hours_i
train_cuts = CutSet.mux(*sources, weights=w, stop_early=False)
```
Weights change the *sampling* proportions while every hour stays available — preferable to
merge-time duplication (which inflates disk and epoch length). Small sets (FLEURS/CV) may
alternatively be duplicated 2–3× at merge if mux is not wired into the datamodule.

**Mechanism 4 — self-training round 2 (later).** Once GP-v2 beats Whisper on Bulgarian, it
relabels the weak-supervised axes (C, D), replacing the weaker teacher and lifting label
quality corpus-wide.

## Order of operations (the one-line version)

harvest → speech/music segment → weak-label + filter (+agreement) → audit/calibrate →
recover-by-arbitration → cuts (shared fbank + normalizer) → merge (leakage guard) →
[balance: caps now, mux later] → train GP-v2 → eval on frozen parliament + held-out
FLEURS/CV test + a fresh unseen domain.
