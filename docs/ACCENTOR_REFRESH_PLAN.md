# Accentor-Refresh Cycle — full-parliament harvest + imperative text → fine-tune

**Axis:** improves **mark correctness** (does the front-end choose въ́лна vs
вълна́, ку́пи vs купи́ for *this* sentence) — **not** acoustic rendering (that's
the TTS fine-tune / the imperative ft8 work). This is the biggest remaining
lever on homograph disambiguation.

**Two complementary sources feed it:**
1. **Full-parliament stress harvest** — hundreds of hours of real human speech
   in a different register; the aligner reads which stress variant the audio
   chose per ambiguous word. Fixes context-dependent homographs
   (въ́лна/вълна́, гово́ри/говори́…).
2. **Curated imperative marked-text (NEW)** — the same sentences we built for
   ft8 (BgTTS set + human recordings), used **text-only**. Real speech *starves
   the imperative reading* (`build_accentor_finetune.py` already flags this:
   "говори́!/гово́ри … parliament/narration starves the imperative reading"), so
   the harvest alone can't teach it. Our curated marks fill that exact gap.

**The payoff — closing the loop:** the Accentor's imperative failure (marks
ку́пи, not купи́) is *why* we needed pins and the ft8 acoustic detour. Teach the
Accentor to *mark* imperatives (this cycle) **and** ft8 to *render* marks (done)
→ correct imperatives **end-to-end, no pins**. The imperative set does double
duty: audio→ft8, text→Accentor.

**Gating:** run after ft8 is validated/promoted. Independent cycle; parliament
audio stays OUT of the TTS (register mismatch would dull the cloned voice).

---

## Stage 1 — Incremental parliament harvest (the label factory)

Use the existing incremental harvester, **not** a pilot re-run. Milestone A's
models are a *fixed* label factory (finding F2: alignment saturates at a few
hundred h — the 63 h GMM aligned 400 h cleanly), so new sessions just pass
through `tri2b_comb → tri4b → tdnn_a`:

```bash
KALDI_ROOT=~/kaldi bash harvest_new_sessions.sh          # auto-diffs new sessions
# -> data/kaldi/stress_labels_incr_<date>.jsonl  (DISJOINT from prior sets)
```

- It diffs the crawl manifest vs already-harvested session ids and processes
  only the new ones (73 GB downloaded; prior harvest = `stress_labels_parliament{,2}`).
- **Gotchas (recorded):** full-vocab `L.fst` × high `nj` OOM'd a 60 GB box —
  keep `seg_nj=8`, check `journalctl` for the oom-killer before blaming a crash.
  If the harvest's phone-map mismatch warning >10%, re-run `get_prons` against
  the current lang first.
- Watch the **segmentation yield** (retained h / total); low yield = transcript↔
  audio drift in those sessions.

---

## Stage 2 — Build the imperative label set (NEW, text-only)

Convert the curated imperative sentences into the harvest label format (one row
per imperative verb token: `{word, marked, sentence}`). Use **all** verbs
including прави/ходи — for the Accentor we use only the *marks* (hand-vetted,
correct); the polluted *audio* is irrelevant here.

```bash
# from the ft8 bank (verb, verb_marked, plain sentence) + the human record set
./venv/bin/python - <<'PY'
import json, re, pathlib
rows=[]
# synthetic set: every verb x frame (marks are curated ground truth)
for r in map(json.loads, open("exp/imperatives/bank.jsonl")):
    rows.append({"word": r["verb"], "marked": r["verb_marked"],
                 "sentence": r["plain"], "source": "imperative_synth"})
# human record sentences (пpави/ходи, the resistant class)
for r in map(json.loads, open("data/bg/imperatives_human/train.jsonl")):
    marked=r["text"]; plain=marked.replace("́","")
    # verb = first token; its marked form carries the acute
    w=plain.split()[0].lower(); wm=marked.split()[0].lower()
    rows.append({"word": w, "marked": wm, "sentence": plain.lower(),
                 "source": "imperative_human"})
pathlib.Path("data/kaldi/stress_labels_imperatives.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in rows)+"\n")
print(len(rows), "imperative label rows")
PY
```

**Balance is automatic:** `build_accentor_finetune.py` mixes chitanka + the
parliament harvest, which carry the *aorist/declarative* readings of the same
words (ку́пи, гово́ри). So the Accentor sees купи́ (imperative, our labels) AND
ку́пи (aorist, corpus) → learns the **contextual distinction**, not a blanket
"купи→купи́" (which would break the aorist — the over-correction we must avoid).

*Refinement if it over-fits the templates:* the synthetic frames are limited
(22); if eval shows the Accentor keying on the verb rather than imperative
context, generate more varied imperative frames (text-only — cheap, no audio).

---

## Stage 3 — Combine + fine-tune the Accentor

```bash
./venv/bin/python build_accentor_finetune.py \
    --labels data/kaldi/stress_labels_milestone_a.jsonl \
    --labels data/kaldi/stress_labels_parliament2.jsonl \
    --labels data/kaldi/stress_labels_incr_<date>.jsonl \
    --labels data/kaldi/stress_labels_imperatives.jsonl \
    --mix-chitanka data/chitanka/marked.jsonl --mix-ratio 5 \
    --out data/accentor_finetune_ft7.jsonl

./venv/bin/python train_accentor.py \
    --data data/accentor_finetune_ft7.jsonl \
    --init exp/accentor_ft6b/best.pt \
    --out exp/accentor_ft7 --steps 40000
```

- Fine-tune **from ft6b** (`--init`) — preserve what it learned, don't retrain
  from scratch.
- Oversample the rare homograph + imperative tokens (the harvest label rows are
  a tiny fraction of the chitanka majority) — same minority-oversampling that
  helped calibration before.

---

## Stage 4 — Evaluate on the axis that matters

```bash
./venv/bin/python eval_accentor.py     --ckpt exp/accentor_ft7/best.pt   # in-domain
./venv/bin/python eval_accentor_ood.py --ckpt exp/accentor_ft7/best.pt   # OOD (real test)
# + homograph battery (config/bg/homograph_battery*.json)
# + a NEW imperative battery: does it now mark купи́/обади́/прави́/ходи́ in
#   imperative context AND ку́пи/оба́ди in aorist context?
```

**Success =**
- homograph battery + OOD improve over ft6b (parliament payoff), AND
- **imperatives now marked correctly in context, aorist intact** (the loop-close),
- with no regression on plain-word stress.

The imperative battery is the headline for *this* project: if it passes, the
served pipeline produces купи́/прави́/ходи́ from the Accentor alone — **the pins
in `bg_stress_overrides.json` become unnecessary** for these words.

---

## Stage 5 — Promote

```bash
ACCENTOR=exp/accentor_ft7/best.pt bash serve_bg.sh
```

Update `read_book.py --accentor` and the served checkpoint together (offline
dogfood ↔ LAN server in lockstep). Then thin `bg_stress_overrides.json` — drop
the imperative pins the Accentor now handles natively; keep only proper-name
pins (Чопари́нов etc.) the model can't infer.

---

## Honest bounds & sequencing

- **Helps:** homograph mark-correctness (parliament) + imperative marking
  (curated text) → fewer wrong-stress words on general prose, and pins retired.
- **Does NOT help:** ч/щ brightness, naturalness, voice quality (acoustic — the
  TTS fine-tune owns those).
- **Byproduct:** the segmented+aligned parliament corpus is also the ASR
  (Zipformer/icefall) training set — scope separately.

Sequence: ft8 promoted → Stage 1 harvest (CPU-heavy Kaldi, hours-to-overnight)
→ Stage 2 imperative labels (minutes) → Stage 3 fine-tune (~1 h) → Stage 4 eval
→ Stage 5 promote + retire pins. Stage 2's native input is already done (the ft8
verb lexicon you vetted).

---

## Progress log — ft7 refresh cycle (2026-07-30)

Stages 1–4 executed. Summary and findings.

**Stage 1 (harvest) — DONE.** 81 new sessions (~288 h effective) through the
fixed Kaldi factory → `stress_labels_incr_20260728.jsonl`: **549,189 tokens,
14,681 types, 2,673 both-variant types**, 38 phone-map mismatches (0.023 %),
disjoint from milestone-A. Two ops lessons in `docs/HARVEST_CRASH_CAPTURE.md`:
the 07-28 freeze was a platform hang on the auto-installed `7.0.0-28` kernel
(NOT OOM — re-run clean on `6.17.0-40`); and incremental cleanup needs an
`align_fmllr` pass first — tri4b's stored `trans.*` are training-speaker
transforms with zero overlap on new sessions (all 24 decode jobs exited 1).

**Native vetting.** 518 audio verdicts (`vetting_results.jsonl`);
`build_vetting_shortlist.py` ranks both-variant tokens by label-noise impact.
Confirmed цели/води/работи as true homographs (→ tier-3); the alignment-noise
splits (колеги́, всички́, вече́…) are already handled by the veto / share<0.9 gate.

**Stages 3–4 — the ft7 sub-cycle (init from ft6b throughout):**

| run | battery v1 | OOD SER | change |
|---|---|---|---|
| ft6b (prior prod) | 15/25 | 0.005 % | baseline |
| ft7 — corpus refresh + tier-3 цели/води/работи | 14/25 | 0.005 % | net-neutral (+па́ра −говори́/чета) |
| ft7b — + starved set (brute, all 164 verbs) | 17/25 | 0.011 % | fixed купи/моли/чета-noun; **final-stress bias** (въ́лна regressed) |
| ft7c — `--homograph-only` + mine_homographs + F3 curated носи/учи/ходи | 22/25 | 0.005 % | imperfective imperatives flipped; въ́лна recovered; bias gone |
| **ft7d — + context-diversity (abstract/comparative/ad imperatives)** | **22/25** | **0.005 %** | ALL imperatives flip on both batteries; residuals move to see-sawing lexicals |

**Generalization eval (NEW) — five-way on unseen data:**

| run | battery v1 | battery v2 (unseen ctx) | wild SER (unseen reg) |
|---|---|---|---|
| ft7b | 17/25 | 16/25 | 0.119 % |
| ft7c | 22/25 | 21/25 | 0.089 % |
| **ft7d** | **22/25** | **22/25** | **0.059 %** |

`config/bg/homograph_battery_v2.json` (25 sentences, 15 homographs, 6 registers,
disjoint from v1 + training) and `fetch_eval_wild.py` (~33 k words unseen
Wikipedia). ft7d is best on both unseen axes; the +5 gain reproduces on unseen
contexts ⇒ learned rule, not overfit.

**Conclusion — SHIPPED ft7d (2026-07-30).** Five checkpoints establish the data
ceiling: the model fits to 0.023 % SER and learns each reading when given adequate
balanced+diverse data (⇒ **data-limited, not architecture-limited**), but the
remaining lexical homographs **само/вълна/чета́ see-saw ±1–2 with training
variance** and pushing imperatives harder starts inverting 3sg (у́чи). ft7d is the
balance point: **all homograph imperatives resolve contextually** (говори́/носи́/
прави́/купи́/учи́/ходи́ pass v1 AND v2), best generalization. Residuals само/вълна/
чета́ are **accepted, unpinnable** (context-dependent → default to the common
reading, right ~99 % of the time; a static pin would break the majority).

**Promotion + pin cull (Stage 5):** `serve_bg.sh` ACCENTOR → `exp/accentor_ft7d/
best.pt`. Data-driven cull of `bg_stress_overrides.json` (79 → ~31): **49 pins
redundant** (ft7d reproduces the mark standalone — CTA -ете imperatives, lexicon
words, ако/или suppression), **20 kept** (model still wrong: стартирайте, вашия,
можете…), **9 homograph pins verify-in-context** (perfective CTA imperatives
добави́/запази́/вдигни́ default to aorist standalone — keep unless context-tested).
Keep чопари́нов (proper name — the one class the model can't infer). Applied
2026-07-30: `bg_stress_overrides.json` **79 → 31 entries**.

**End-to-end audio eval (2026-07-30, reklama voice) — closes the marking track.**
21 hard cases through the served TTS. Of the 10 audio failures, **8 were
RENDERING, not marking**: ft7d marked прави́/носи́/ходи́/купи́/говори́/цели́/води́/
добаве́те all correctly, but the fixed voice did not *render* the ending stress
(учи́ *did* render — voice- and word-specific). Only 2 were mark errors
(само́/чета́), both context-dependent see-saw homographs whose majority reading
passed ⇒ unpinnable. **Nothing to add to the overrides**: render errors are
mark-already-correct (a pin is a no-op); mark errors are not statically pinnable.
**The imperative ceiling is therefore 100 % acoustic** — the exact target of the
volunteer many-voice drive → ft9. The Accentor front-end (marking) is **done**;
the open lever is acoustic, not front-end work.

**Multi-voice reframe + PAUSE (2026-07-30).** Re-tested the failing imperatives
across many reference voices: **almost all voices fail the SAME words**
(прави́/носи́/ходи́/купи́/говори́), учи́ a rare exception. So it is NOT voice-specific
— it's a **model-level word-prior bias** in the production e3_sparse TTS (the mark
doesn't override the learned пра́ви/но́си 3sg prior, uniformly across voices; the
reference voice sets timbre, not this stress prior). This is actually a *cleaner*
target: one uniform word-prior, so teaching прави́ helps every voice at once —
exactly what the many-voice **ft9** produces. Interim levers: **BgTTS-38M renders
these correctly from plain text** (`engine:"bgtts"`) = stopgap for imperative/ad
content today; `guidance_scale`↑ is a cheap per-request test. **The imperative
RENDERING work is now PAUSED** — pivoted to the ASR byproduct (see
[BG_ASR_PLAN.md](BG_ASR_PLAN.md)); ft9 remains the durable fix when the volunteer
dataset matures.

**Findings this cycle** (→ paper §7):
1. **The Accentor is data-limited, not architecture-limited** — it fits to
   0.026 % SER and learns every reading the moment adequate balanced data appears;
   capacity/receptive-field are not the bottleneck.
2. **Brute-force final-stress injection biases globally** — marking all 164 verbs'
   imperatives (ft7b) tipped въ́лна→вълна́ and doubled OOD SER; `--homograph-only`
   removed it.
3. **Context beats volume at the imperfective-imperative wall** — носи́/учи́/ходи́
   flipped at ~250 *diverse* contexts (curated + natural mined) + balanced 3sg,
   where 176 templated frames alone had failed in every prior checkpoint.
4. **Generalization must be measured on unseen contexts AND unseen register** —
   battery v1 can't tell learning from memorization; v2 + wild-SER can.
