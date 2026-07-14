# OmniVoice-BG — Lessons Learnt

Distilled from findings F1–F14 (full evidence in `BG_Accentor_Paper.ipynb`
and `BG_PIPELINE_GUIDE.md`). This is the education document: each lesson is
the transferable rule, one line of evidence, and where it bit us.

## Label manufacturing (Kaldi as annotator)

**L1 — A GMM aligner with a multi-pronunciation stress lexicon is a
label factory.** Forced alignment *chooses* a stress variant per spoken
token; over 400 h that manufactured 1.22 M contextual homograph labels no
text-only method could produce. The same mechanism, pointed at TTS output,
later became the best-of-N judge — build annotators so they can be reused
as judges.

**L2 — Alignment health gates beat retraining reflexes.** Log-likelihood
(−45.8), retry rate (0.37 %), fail rate (0.21 %) told us the 63 h-trained
GMM aligned the full 400 h fine — the incremental retrain we almost ran was
unnecessary. Measure before you scale the annotator.

**L3 — Vet labels with a native BEFORE training, per class, and use it as
a veto.** Pre-training vetting precision *predicts* downstream damage:
че́та at 19 % precision inverted its class exactly as forecast (F3); a
per-token veto of 169 native-rejected tokens (ft5b) repaired it. Vetting a
few dozen tokens per boundary class costs minutes and saves a training run.

**L4 — OOV anatomy is worth measuring, not just counting.** 5.59 % token
OOV decomposed into registers; the по-/най- comparative slice (0.64 %) is
mechanically recoverable with one lexicon rule (F10). Aggregate OOV rates
hide the recoverable structure.

## Teaching a small model rare contrasts (Accentor)

**L5 — Coverage floors are floors, not guarantees.** ~20 contexts is the
observed minimum for a homograph class to be learnable (F2), but пара́ at
24 contexts still didn't flip — sufficiency depends on class balance and
context diversity, not raw count.

**L6 — There is an absolute-count wall, with two escape hatches.**
Boundary classes flip somewhere in [50, 100] absolute tokens (прави́ at
~99, F4). Escapes: cross-item transfer within a morphological family
(ходи́ flipped at 3 tokens once sibling verbs were learned) and extreme
ratios re-entering the fight (само́ at 100:1).

**L7 — Sparse + noisy minority data is worse than none.** Below the wall,
noisy minority labels don't just fail — they *invert* the class (F3). If
you can't vet it, exclude it (tier system) rather than include-and-hope.

**L8 — Validation SER is blind to exactly what you care about.**
Homographs are too rare to move corpus-level SER, so best-val checkpoints
can be worse on the boundary set, which *oscillates* between checkpoints
(F11). Select checkpoints on a purpose-built battery + wild-text reads;
keep a human-overrides file as the last-resort veto.

**L9 — Prosodically weak words need suppression at inference, not just
exclusion from training.** We excluded още/ако/между/като… from ft5b
*training* (tier-2), but the Accentor still marks them at *inference* —
the bon run caught it asking the TTS for още́ and ако́. A training-set
policy must have an inference-time twin.

## Making the TTS obey the marks (OmniVoice)

**L10 — Marking consistency is a contract (F9).** The TTS must be
fine-tuned on marks from the *exact* Accentor version being served.
Upgrading the Accentor costs one ~40-min re-mark + 3000-step refresh;
audio tokens are reusable, so the cycle is cheap — budget for it on every
front-end change.

**L11 — Acoustic data volume saturates before mark obedience does (F12).**
63 h → 413 h same-narrator data moved eval loss 3.88 → 3.43 yet barely
moved stress-rendering fidelity. Once the mark→acoustics mapping is
learned, more clean data reinforces the *prior*, not the *instruction*.

**L12 — The model's prior can dominate the instruction, and sampling
reveals it (F13/F14).** Best-of-8 oracle: single-sample 12/27 sentences
fully correct, judge-picked 19/27 — and the judge captured 100 % of the
oracle headroom. But correctly-marked rare readings — the final-stress
imperatives прави́ and ходи́, plus сгреши́, пеша́, въ́лна — failed in
**all 8 candidates**: the prior wins regardless of the mark, and it is
per-word (учи́ imperative renders 8/8 while ходи́ renders 0/8), not
per-class. Selection fixes sampling variance; only preference-style
training (RAFT on judge winners) or stronger conditioning fixes prior
dominance.

**L13 — Your judge's blind spots become your blind spots.** The aligner
can only adjudicate words with >1 lexicon variant, and by-ear review found
real stress errors on single-variant words the judge scored as clean. Every
automated judge needs a coverage statement and a human sampling loop.

**L17 — Obedience is not correctness; unanimous disobedience audits the
instructor.** The judge scores whether the TTS followed the mark — but
when all 8 candidates contradicted a mark, the mark itself was usually
wrong: the bon table exposed real Accentor homograph failures (пара́ for
*steam*, чета́ for the *noun*, у́чи for the *imperative*, Си́нът for
синъ́т) that the TTS prior silently corrected. A judged pipeline audits
itself in both directions — mine 8/8-disobedience for front-end errors
before treating them as model failures, and never report an obedience
score as a quality score until the instructions are vetted.

## Process & infrastructure

**L14 — Every pipeline stage needs a resume story before you need it.**
Power surges hit mid-crawl, mid-download, and mid-fine-tune. What saved
each: crawler `state/*.ok` idempotency, parquet-footer shard fast-forward +
`wget -c` loops (HF streaming `.skip()` re-reads ~60 GB), and
`resume_from_checkpoint` with optimizer/scheduler/RNG state. Also: check
`journalctl` for OOM kills before blaming the power grid.

**L15 — Ship launcher scripts, not long commands.** Two training launches
died on paste artifacts (trailing space after `\`, terminal line-wrap).
Anything longer than one line goes in a `.sh` with `set -euo pipefail` and
prerequisite guards.

**L16 — Keep the native speaker in the loop at decision points, not just
review.** промените's correct contextual reading (проме́ните), боя's
pinning, tier-2 exclusions, and the pending по́-добре́ double-stress call
were all judgment calls no metric could make.
