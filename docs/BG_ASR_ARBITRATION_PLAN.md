# Reject arbitration — recovering discarded weak-supervised audio with GP-v2

**Status: tool written and dry-run validated (2026-08-08), full pass not yet run.** Designed to
run on **CPU alongside GP-S training**, so it does not compete for the GPU.

## What this is, and what it is not

The weak-supervised axes were labelled by Whisper and gated against the YouTube caption — an
independent ASR. A window survived only if the two **agreed**. Where they disagreed there was no
tiebreaker and the audio was dropped. GP-v2 (4.12 % on parliament vs Whisper large-v3's 15.37 %)
is now a stronger third opinion than either original teacher, so it can adjudicate.

Measured from `data/<axis>/pseudolabels.jsonl`:

| Axis | Accepted (in GP-v2) | Rejected windows | **Discarded hours** | Dominant reason |
|---|---|---|---|---|
| ytdr | 88,133 / 292.6 h | 32,885 | 36.3 h | disagree (22,833) |
| ytgi | 55,771 / 214.0 h | 18,449 | 22.1 h | disagree (13,283) |
| ytbb | 25,323 / 135.1 h | 5,443 | 8.6 h | disagree (2,546) |
| BNR | 34,426 / 102.9 h | 15,424 | 23.8 h | lang-ID (14,923) |
| | | | **90.8 h** | |

**It does not touch the accepted windows.** Those already passed two-teacher agreement, audited
at single-digit implied label error, and native listening called them "nearly perfect" on
ytbb+ytgi. Overwriting them with GP-v2 output would risk replacing correct human-caption text
with the model's own errors and entrenching them — the classic self-training trap. Operating on
rejects only means a bad decision can add noise but can never **degrade** a label that is
already good.

This also corrects an earlier misreading: the 42.7 % weak-labelled share is *not* raw Whisper
output at ~15 % WER. It is the agreement-filtered subset, which is much better than that. The
prize here is **recovered hours**, not repaired labels.

## Decision rule

Three independent opinions per window: Whisper's stored text, the caption re-sliced from the
`.bg.vtt` still on disk, and a fresh GP-v2 decode. All compared through `bg_text_norm` (the
single scoring authority), so "127 лева" and "сто двадесет и седем лева" count as agreement.

```
zip ≈ whisper  (zwer ≤ 0.5)  -> accept, label = WHISPER's text
zip ≈ caption  (zwer ≤ 0.5)  -> accept, label = GP-V2's text   (caption only corroborates)
neither                       -> stay rejected
```

**Why the caption never supplies the label.** Whisper's text and GP-v2's were produced *from*
the window and are aligned to it. The caption is document-level text sliced by timestamp, and
measurement showed it routinely bleeds in words from the neighbouring sentence — one dry-run
case carried "държавен фонд на земеделие и там примерно е" into a window where GP-v2 heard none
of it. Writing that verbatim puts words in the label that are not in the audio, which teaches
hallucination. So when the caption turns out to be the correct one, we keep GP-v2's aligned
decode instead.

A speaking-rate band (`--min-wps 0.8`, `--max-wps 6.0`) catches residual misalignment and
Whisper truncation — a 9 s window labelled with four words teaches deletions.

`zwer ≤ 0.5` follows the precedent already recorded in `BG_ASR_DATA_PIPELINE.md`
("Zipformer arbitration (zwer ≤ 0.5) → C recovery").

## Dry-run result (ytdr, 120 disagree windows)

```
recovered 89 of 120 (74.2%)
  backed whisper 85 · corroborated by caption 4 · still ambiguous 28
  failed rate check 0 · too short 3 · errors 0
  ~18.6 windows/s on 4 threads
```

Extrapolating ~74 % across the 67 h of YT disagreements suggests **~50 h recoverable**, plus
whatever the BNR lang-ID pass yields. At ~18 win/s the full YT pass is roughly 1 hour of
wall-clock on 4 threads.

## Running it

```bash
cd ~/WORK/OmniVoice
for ax in ytdr ytgi ytbb; do
  ./venv-asr/bin/python arbitrate_rejects.py --axis $ax \
      --why-prefix disagree --num-threads 4 --audit-sample 40
done
# BNR: language-ID false rejects, no captions -> 2-way against the stored text
./venv-asr/bin/python arbitrate_rejects.py --axis bnr \
    --why-prefix lang --num-threads 4 --audit-sample 40
```

`--num-threads 4` on a 24-core box leaves ample room for GP-S's dataloader workers. Writes
`data/<axis>/arbitrated.jsonl` plus `data/<axis>/arbitration_audit.md`.

## Gate before use: listen to the audit sheet

**Do not merge recovered audio without listening to the sheet first.** These are windows the
original gate rejected; the sheet is the check on whether the arbiter is trustworthy *on that
axis*. GP-v2's 4.12 % is parliament — its accuracy on casual YouTube conversation and archival
radio has never been measured, and that is exactly the material here.

Each entry shows the chosen label alongside all three opinions. If a sheet reads poorly for an
axis, raise `--max-wer` (stricter) for that axis or skip it. A pass rate below ~90 % on the
sheet should stop the merge.

## Downstream — and the shuffle

Recovered rows become cuts, then features, then part of the next corpus:

1. `arbitrated.jsonl` → cuts + 80-dim fbank via the existing per-axis prep path.
2. Merge with **`merge_train_corpus.py`**, which since `cd5ef6f` performs a **global seeded
   shuffle by default** — the train-all-shuf lesson. Do not pass `--no-shuffle`. A
   source-ordered manifest makes the sampler's windowed shuffle sweep whole registers in
   sequence, which is what that commit fixed. Output should again carry a `_shuf` suffix.
3. Pass every eval manifest as `--holdout` (dev, test, FLEURS, CV) so the leakage guard runs.
4. Retrain **offline** as GP-v3 and compare against GP-v2's 4.12 % — one variable (corpus),
   attributable, exactly as v1 → GP-v2 was.

**GP-S is unaffected.** It trains on the frozen `bg_cuts_train_gp_shuf.jsonl.gz`, which is
already globally shuffled, so its comparison against GP-v2 isolates causality alone. Recovered
audio lands in GP-v3, and in GP-S2 afterwards if streaming becomes the product direction.

## Honest expectations

~50–70 h on top of 1,745 h is a **3–4 % corpus increase**, which on its own is worth maybe a
percent or two relative. Two reasons it is still worth doing: the audio is already paid for
(harvested, segmented, stored), and disagreement windows are disproportionately *hard* —
overlapping speech, noise, accents, unusual vocabulary — which is worth more per hour than
average material and is precisely the register where the model is weakest.
