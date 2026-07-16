# OmniVoice-BG — Final Effort Plan (Milestone A cycle)

Trigger: parliament crawl approaching completion (316.8 h banked of the
≥400 h gate at last count). This document is the execution plan for the
last big push. Lessons feeding it: `BG_LESSONS.md`; artifact map:
`BG_ARTIFACTS_INDEX.md`.

## 1. Where we stand

| Component | State | Number |
|---|---|---|
| Accentor (served) | ft5b `best.pt` @11 K | battery 15/25, OOD SER 0.020 %, wild-text clean |
| TTS (best) | `omnivoice_bg_full/checkpoint-6500` (413 h + ft5b marks) | eval loss 3.4315 |
| Best-of-8 oracle | judge = oracle | single 12/27 → best-of-N **19/27** |
| Parliament corpus | crawling 2020–2026 block | 316.8 h / 400 h gate |

## 2. Error decomposition (what remains broken, and whose fault)

From the bon run + by-ear review, four distinct failure classes — each
with a different fix, which is why the plan has parallel tracks:

| Class | Evidence | Fix lives in |
|---|---|---|
| **A. Sampling variance** | 7/27 sentences recovered by picking among 8 candidates; splits like брега́ 4/8, коса́ 4/8, черта́ 4/8, купи́ 4/8 | Serve-time best-of-N (judge already optimal) |
| **B. Prior dominance** (mark RIGHT, TTS refuses) | прави́ & ходи́ (final-stress imperatives) 0/8, сгреши́ 0/8, пеша́ 1/8, въ́лна 1/8 (model wants вълна́ in both contexts). Per-word priors, not a class rule — учи́ imperative renders 8/8 | Preference training (RAFT on judge-picked winners, §P4); more minority-class *labels* from milestone-A harvest |
| **C. Accentor errors** (mark WRONG, TTS's unanimous disobedience exposed it) | Real homograph failures: пара́ for *steam* (s02), чета́ for the *noun* (s05), у́чи for the *imperative* (s17), Си́нът for синъ́т (s16), verb чета left unmarked (s04); plus tier-2 weak words още́/ако́/между́ marked at inference | ft6 training data for the contextual ones (mine steam-пара, noun-чета, imperative-учи contexts); global overrides for синъ́т; suppression list for weak words (L9); native ruling on all |
| **D. Single-variant stress errors** | heard by ear; invisible to the judge (one lexicon variant → no choice) | All-variant judge lexicon (§P5), then recount |

**New instrument discovered:** *unanimous disobedience audits the
front-end.* The judge grades TTS obedience to the mark; when all 8
candidates contradict a mark whose prior reading is linguistically right,
the experiment has caught an **Accentor** error instead. The pipeline
audits itself in both directions — fold this into P4 data generation:
8/8-disobedience sentences are candidate Accentor-error mines, and the
judged "19/27" is an *obedience* score, not a correctness score; true
quality accounting requires the marks be native-vetted first.

## 3. Prerequisites (start now, all parallelizable with the crawl)

- [ ] **CUDA 12.x Kaldi rebuild** — required for milestone-A TDNN stages 6+
      (`kaldi_milestone_a.sh` hard-gates on `cuda-compiled`).
- [ ] **Native rulings** (native reviewer):
      1. по́-добре́ — double stress on по-/най- comparatives? (unlocks the
         0.64 % OOV-recovery lexicon rule, F10)
      2. Correct prosody for още / ако / между in the border sentences —
         is the Accentor wrong (suppress) or right (keep)?
      3. Confirm the class-C Accentor errors (пара́@s02, чета́@s05,
         у́чи@s17, Си́нът@s16, unmarked чета@s04) and the class-B
         memberships (прави́, ходи́, сгреши́, пеша́, въ́лна) by ear in
         `bon_eval/picked/`.
      4. синъ́т — confirm, then add to `bg_stress_overrides.json`
         (definite form is context-free, safe as a global override).
- [ ] **Crawl to ≥400 h** parliament (auto-stops at 200 GB floor;
      `crawl-status` to monitor).

## 4. Execution phases

### P1 — Milestone-A acoustic model (`kaldi_milestone_a.sh`)
Stages 0–5 on the CPU build (volume gate, full-crawl segmentation,
combined GMM retrain → tri4b, cleanup); stages 6–8 after the CUDA rebuild
(CE non-chain TDNN — 10 ms frames preserved for harvest timestamps —
align + F5 acceptance: може/всички/съюз counts must flip canonical).

**Recognized byproduct — a full Bulgarian ASR (off the critical path,
recorded 2026-07-15):** P1 produces two of the three components of a
complete ASR system: the TDNN acoustic model (trained on ~860 h of
transcribed BG: audiobooks + parliament) and the 115 K-word pronunciation
lexicon. The only missing piece is an n-gram language model — buildable
from chitanka text — after which LM + lexicon + TDNN compile into an HCLG
decoding graph = a working Bulgarian recognizer. Days of work on top of
milestone-A, not months. Uses, in priority order:
1. **Standalone release/paper**: ~860 h of aligned, cleaned BG speech is
   likely the largest such corpus assembled; "BG parliament ASR + corpus
   recipe" mirrors the Finnish-parliament reference paper (2203.14876)
   and is publishable alongside the Accentor work.
2. **Data flywheel**: transcribe UNLABELED Bulgarian audio to grow the
   corpus — with ASR transcripts quarantined from stress-label harvesting
   (same circularity discipline as synthetic audio).
3. In-pipeline value is modest (Whisper already covers the CER gate); a
   domain-tuned ASR only wins on new unlabeled in-domain audio.
Nothing in P1–P6 depends on this — build it only after the cycle lands.

**Architecture note — Zipformer/icefall for the ASR, TDNN stays for the
aligner (recorded 2026-07-16).** The milestone-A **TDNN is the right tool
for label manufacturing** and should NOT be swapped: forced alignment is
transcript-constrained (F2 saturation — a stronger acoustic model barely
moves alignment quality), and the TDNN's *bounded* context gives crisper
10 ms phone boundaries than a long-memory LSTM or a Transformer — boundary
precision is the whole game for the stress harvest. But the ASR byproduct
is **free decoding**, where acoustic-model power does matter, and there the
modern choice is a **Zipformer in icefall/k2** (CTC or pruned RNN-T). Nice
coherence: **icefall is k2-fsa — the same group behind OmniVoice**, so a
"modern k2 stack" (Zipformer ASR + OmniVoice TTS) is self-consistent. The
milestone-A assets transfer directly: the 740 h cleaned+aligned corpus and
the 115 K lexicon are exactly icefall's inputs; only an n-gram/neural LM
must be added. Effort is a framework port (k2 paradigm), justified only if
the ASR is pursued as its own deliverable — never as a mid-flight swap of
the working Kaldi label pipeline. (Transformer/LSTM were both weighed for
the aligner and rejected: Transformer-CTC can't do the pronunciation-
variant FST fork; LSTM smears boundaries. The Accentor itself already uses
Transformer-style FFT blocks — architecture is chosen per task.)

**Feature upgrade decision (do at P1, free while retraining anyway):**
add Kaldi pitch features (`steps/make_mfcc_pitch.sh`, +3 dims: F0/voicing/
ΔF0) — Bulgarian stress is realized via pitch movement that plain MFCC
discards, so the aligner currently discriminates а́/а on reduction +
duration alone. Use hi-res 40-dim MFCC (+pitch) for the TDNN stage (DCT
truncation to 13 exists for GMMs only). Any feature change invalidates
all models — which is why this rides the milestone-A retrain and must not
be done piecemeal before it. SSL-based aligners were considered and
rejected: the lexicon-FST pronunciation fork IS the method; CTC aligners
can't choose stress variants. Optional later: WavLM stress classifier
over aligner-localized vowel spans as a second judge.

### P2 — Harvest v3 + Accentor ft6
1. Harvest audiobooks + parliament with the TDNN aligner (expect >1.4 M
   tokens; parliament adds multi-speaker evidence for class-B minorities).
2. Targeted vetting: every class-B word (пара́, чета́, прави́, ходи́,
   сгреши́, пеша́ …) gets a `vet_stress_cli.py --per-variant` pass; verdicts
   append to `vetting_results.jsonl` (veto is cumulative).
3. `build_accentor_finetune.py` with the veto; **new**: emit an
   inference-suppression list from tier-2 (fix weak-word marking at the
   source).
4. Targeted context mining for the confirmed class-C homograph errors:
   steam-па́ра, noun-че́та (adjective+чета patterns), imperative-учи́ —
   these need training *contexts*, not overrides.
5. **BgGPT in the front-end loop (adopted 2026-07-14)** — homograph
   resolution is a text-understanding problem; INSAIT's BgGPT (Gemma-based,
   BG-extended tokenizer) has the semantics our 19.3 M char-level Accentor
   lacks. Concrete model: **`INSAIT-Institute/BgGPT-Gemma-3-27B-IT-GGUF`
   `Q4_K_M`** on local llama.cpp (llama-server, OpenAI-compatible API).
   Serving reality: ~16–17 GB file → CPU-only on the 24-core/60 GB box
   (or partial `-ngl` offload when the GPU is idle), ~2–4 tok/s → design
   for FEW-TOKEN constrained outputs (emit the homograph choice, not
   rewritten text); LoRA at 27 B is off the table locally, so the design
   is pure few-shot + distillation (its outputs train the Accentor, its
   weights never change). Sanity-check first that combining U+0301
   survives tokenizer round-trip. Two roles:
   - **Teacher**: few-shot prompted with harvest-labeled examples, marks
     large chitanka text in overnight CPU batches → distilled into the
     fast Accentor; Kaldi labels remain ground truth, BgGPT amplifies
     their reach into contexts the harvest never saw.
   - **Arbiter**: third opinion on Accentor-uncertain sentences and on
     8/8-disobedience flags from the bon audit (few-token verdicts —
     CPU latency is irrelevant here).
   **Validation gate — run 2026-07-15, two-phase result:**
   - Direct stress-notation prompting **FAILED 15/28** (`bggpt_validate.py`):
     BgGPT picks one fixed reading per word type, context-blind, sometimes
     the wrong one (моли́, ко́са for hair, си́нът). Root cause: BG
     orthography never writes stress, so no text-trained LLM has a
     context→U+0301 mapping.
   - Sense-disambiguation prompting **PASSED 12/12** on the very same
     items (`bggpt_validate2.py`): asked "аорист или повелително?" /
     "дружина или глагол?" with glossed (а)/(б) options, it was flawless.
   **Adopted design (two-step):** BgGPT decides the SENSE, the lexicon
   supplies the MARK — the model never sees stress notation. New curation
   artifact required: a per-homograph **sense inventory** (gloss pairs as
   in bggpt_validate2.py) covering tier-3 + P5-discovered homographs;
   one-time, native-checkable. Teacher pipeline: chitanka sentence →
   BgGPT sense (few-token а/б) → lexicon mark → Accentor training row.
   **Rejected as TTS backbone**: swapping Qwen3→BgGPT discards the audio
   weights (TTS + in-context cloning live in the trained weights, not the
   architecture) and ~2.6 B doesn't full-fine-tune on the 16 GB card —
   same verdict as from-scratch reproduction, see §P4 continued-pretrain
   contingency instead.
6. Train ft6; select on battery + wild reads (never val-SER alone, L8) +
   the 8/8-disobedience audit from the bon harness as a new gate.

### P3 — TTS refresh (F9 contract)
`mark_tts_txts.py` with ft6 over the 448 h token dirs (txts only — audio
tokens reused), then a 3000-step refresh initialized from
`omnivoice_bg_full/checkpoint-6500`. ~40 min. Re-run the border A/B.

**Contingency — BG continued pretraining (escalation if P4 RAFT
saturates):** init from the MULTILINGUAL base (cloning ability intact,
NOT our narrator fine-tune), continue pretraining on ALL BG audio —
audiobooks + parliament (multi-speaker is a feature at this layer, not
dilution) — with every transcript ft6-marked, ~50–100 K steps (1–2 days
on the 16 GB card), then narrator fine-tune on top. Attacks the thin-BG-
prior root cause (prior dominance) at the level RAFT only polishes.
From-scratch BG-only reproduction and a BgGPT backbone swap were both
considered and rejected: cloning is emergent from ~100 K h multi-speaker
pretraining that ~900 h of BG audio cannot re-create, and the TTS ability
lives in the trained weights, not the architecture.

### P4 — Preference ladder (fixes class B; rungs per guide stage 8)
1. **Rung 0 now**: serve-time best-of-N in `serve.py` behind a flag,
   triggered only when the marked text contains a judgeable boundary word
   (bounded latency; judge = oracle, so N=4–8 recovers class A
   immediately). Base-independent — ships before milestone A.
2. Scale `bon_generate.py` beyond the border set — **chitanka as the text
   source (adopted 2026-07-14)**: mine chitanka for sentences containing
   class-B words and rare homograph contexts (че́та-noun, imperative-ходи́
   …) that the natural 400 h never provided; K=8, overnight generation.
   The new information in this loop is TEXT/context diversity, not
   acoustics — raw self-training on own output is model collapse; the
   composite judge (CER gate → stress rank) is what turns it into RAFT.
   Hard rules: synthetic audio NEVER enters the Kaldi label harvest
   (circularity trap above), and every synthetic clip carries its judge
   verdict in the manifest (bad batches must be traceable and removable).
3. Judge with `bon_judge.sh`; keep judge-winning candidates.
4. **Rung 1 (RAFT, recommended target)**: fine-tune from the P3 checkpoint
   on the winners with the ORDINARY training objective — reuses the
   existing trainer, no new machinery. Rung 2 (Diffusion-DPO with a
   masked-CE proxy) only if RAFT saturates.

**Composite judge design (adopted 2026-07-14, wire in during P4):**
the Kaldi stress-aligner and a plain-ASR round-trip have exactly
complementary blind spots, so judging is a two-stage composite:

| Stage | Component | Catches | Blind to |
|---|---|---|---|
| 1. GATE | Whisper round-trip CER vs input text (threshold ~10 %) | dropped/repeated/hallucinated/wrong words, garble — cases forced alignment force-fits (the `-:1` rows) | stress: BG orthography doesn't write it, so па́ра/пара́ transcribe identically |
| 2. RANK | Kaldi stress score (current `bon_score.py`) among gate survivors | which stress variant was actually rendered | wrong-word audio (it must align *something*) |

Apply the same gate→rank order in `bon_score.py` (`--cer-gate`) and the
serve-time best-of-N (rung 0), so a stress-perfect but word-mangled
candidate can never win. Whisper loading already exists in OmniVoice
(`load_asr`). Speaker-sim / UTMOS stay optional tiebreaks per the guide's
stage-8 reward table.

**Circularity trap (recorded so nobody falls in):** do NOT train a
stress-aware free-decoding ASR on our marked *transcripts* — those marks
are Accentor predictions, so such a judge would inherit and then
"confirm" the Accentor's own errors. Stress ground truth must keep coming
from the audio side (Kaldi labels, P5 free-stress sweep, milestone-A
TDNN), never from the text side.

### P5 — Judge coverage + lexicon recall audit (fixes class D measurement)
Build a **judge-only all-variant lexicon**: every polysyllabic word gets
all vowel-stress variants (the serving lexicon stays canonical). Two uses
of the same artifact:
1. Re-run the bon judge with it so single-variant words become
   adjudicable; class D gets a measured rate instead of anecdotes, and
   P4's RAFT winner set can include those errors too.
2. **Lexicon recall audit**: re-align the 400 h corpus with it. Wherever
   the aligner repeatedly/confidently picks a stress position NOT in the
   serving lexicon, a missing variant has been discovered empirically
   (missing variants are otherwise invisible — forced alignment force-fits
   the only option and reports success). Complementary detectors:
   grammar-paradigm enumeration (Grammar tom III + `homograph_verbs.json`:
   diff grammar-predicted variant sets against the lexicon per
   morphological class) and a native triage of the top-100
   frequency-ranked suspects. Recall is the only direction to chase —
   spurious variants are already caught by vetting + canonical-share
   tiers; missing ones silently corrupt labels.
   Memory guard: the 2.5 M-word L.fst OOM'd the 60 GB box once — build
   the all-variant lexicon on CORPUS VOCAB only (~115 K words).
3. **Lexicon regeneration from audio (adopted 2026-07-14)** — extend the
   audit into rebuilding the menu from evidence. Aggregate the free-sweep
   choices per (word, variant): count, share, context diversity (distinct
   sentences — F2: count without diversity lies). Classify:
   - both variants well-attested across diverse contexts → **confirmed
     homograph**: add to lexicon v2 AND to the Accentor context-mining
     list (each discovery ships with vettable timestamps);
   - one variant ~100 % and it CONTRADICTS the current canonical entry →
     wrong lexicon entry, audio outvotes the dictionary;
   - never attested → no evidence, not disproof: dictionary-backed
     variants STAY (recall orientation survives the rebuild); only
     G2P-overgenerated variants with no dictionary backing and no
     attestation get demoted.
   Traps to filter: (a) prosodic weakening masquerading as a second
   reading (като́ lesson) — homograph variants correlate with context/POS,
   prosodic ones with phrase position/rate; ambiguous → native triage;
   (b) aligner noise on short/reduced vowels — apply the tier-1
   count + canonical-share thresholds unchanged.
   This closes the loop (menu → alignment → labels → better menu);
   expect convergence in 1–2 iterations. Sequencing: run the regeneration
   AFTER the P1 retrain (better acoustic model = cleaner free-sweep), and
   feed lexicon v2 into the P2 harvest.

### P6 — Final acceptance & ship
- Border set: judge-scored ≥25/27 with serve-time settings.
- Class-D rate below an agreed threshold on a 100-sentence wild sample.
- TrainGen listening sign-off (native reviewer) on production sentences.
- Serve: best TTS + ft6 + suppression list + overrides; update
  `preslaff/omnivoice-bg` on HF; final paper-notebook re-execution with
  milestone-A tables.

## 5. Order of operations

P1 gates P2 gates P3 (each needs the previous). P4.1 (serve-time
best-of-N) and P5 (judge lexicon) are independent — buildable **now**,
before the crawl finishes. The RAFT data generation (P4.2–3) can also be
prototyped early against the existing bon_eval labels, but per the guide's
sequencing discipline the RAFT *training* runs only on the post-P3 base.

## 6. Steady state after milestone A: the model is a fixed label factory

Milestone A is a **one-time** build of the acoustic model; feeding it more
data afterwards is **incremental**, not a restart. The distinction rests on
finding F2 — alignment models saturate at a few hundred hours (the 63 h GMM
aligned the full 400 h cleanly, log-lik −45.8), so once tri4b + the annotation
TDNN exist they are a *fixed label factory*: more parliament audio buys more
homograph **tokens/contexts** (pushing more F4-wall classes over), not a better
aligner.

**Two regimes, opposite incrementality:**

- **More labels (the routine case) — no retrain.** New crawl sessions run
  through the *existing* models only: `harvest_new_sessions.sh` diffs the
  manifest against already-harvested session ids, then segments (tri2b_comb
  seed) → cleans (tri4b) → hires MFCC → TDNN-aligns (tdnn_a) → harvests the
  new sessions to a tagged, **disjoint** `stress_labels_incr_<tag>.jsonl` that
  stacks with `stress_labels_milestone_a.jsonl`. Then: native spot-vet the new
  boundary tokens → rebuild the Accentor set over all `stress_labels_*.jsonl`
  → retrain the ~40 min Accentor → (F9) re-mark + refresh the TTS. The
  acoustic model is never touched.

- **A better aligner (milestone decision) — full retrain.** Only worth it at
  a deliberate "milestone B" (e.g. another ~400 h banked, or a genuinely new
  domain the fixed model aligns poorly). Plain "more parliament" never
  triggers this.

**Caveat.** The incremental path assumes same-domain audio (parliament speech,
same stenogram register). A different source (radio, other register) may align
worse and argue for a retrain — but that is a domain judgement, not a volume
one. `harvest_new_sessions.sh` is drafted (stage-resumable) but UNTESTED until
milestone A finishes and new sessions are banked; watch its first run
(stage-0 session diff, stage-1 `prep_kaldi_data --long-set`).
