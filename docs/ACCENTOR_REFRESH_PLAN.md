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
