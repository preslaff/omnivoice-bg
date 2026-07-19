# Accentor-Refresh Cycle — full-parliament stress harvest → Accentor fine-tune

**Axis:** this improves **mark correctness** (does the front-end choose въ́лна vs
вълна́ for *this* sentence), **not acoustic rendering** (that's the TTS
fine-tune / the imperative work). It's the single biggest remaining lever on
homograph disambiguation — the errors the rechko dict structurally can't fix
and BgTTS/OmniVoice can't either, because they're a *labelling* problem, not a
*saying* problem.

**Why parliament:** hundreds of hours of **real human speech** in a *different
register* from the audiobook corpus. The Kaldi aligner reads which stress
variant the audio actually chose per ambiguous word (`harvest_stress_labels.py`)
— contextual ground truth the model is otherwise starved of. 73 GB is
downloaded (`data/bg_parliament`); the pilot harvested only a subset
(`stress_labels_parliament{,2}.jsonl`).

**Gating:** run this *after* ft8 is validated and promoted. It's an independent
cycle — do not fold parliament audio into the TTS (wrong register, would pull
the cloned voice toward podium prosody).

**Prerequisites (already satisfied):** `kaldi_pilot.sh` through stage 3
(`exp/kaldi_pilot/exp/{tri2b,tri3b}` exist); `KALDI_ROOT=~/kaldi`.

---

## Stage 1 — Scale the segmentation + harvest to the full corpus

The pilot ran `kaldi_parliament_pilot.sh` on a subset. Now run it over **all**
downloaded sessions. It self-stages (segment → fMLLR-align → harvest):

```bash
KALDI_ROOT=~/kaldi bash kaldi_parliament_pilot.sh      # stages 0-3, all sessions
```

- **stage 0-1:** `segment_long_utterances.sh` (biased-LM decode, seed=tri2b)
  turns 2 h session recordings into utterances → `data/parl_seg`. Watch the
  **segmentation yield** it prints (retained h / total) — the pilot's viability
  bar was ≥~50%; below that, transcripts and audio are drifting.
- **stage 2:** fMLLR-align the segments with tri3b → `exp/tri3b_parl_ali`.
- **stage 3:** `get_prons.sh` → then `harvest_stress_labels.py` reads the
  chosen stress variant per ambiguous token:
  ```bash
  ./venv/bin/python harvest_stress_labels.py \
      --pilot exp/kaldi_pilot --ali exp/tri3b_parl_ali --data data/parl_seg \
      --out data/kaldi/stress_labels_parliament_full.jsonl
  ```

**Gotcha (recorded):** the full-vocab `L.fst` × high `nj` OOM'd a 60 GB box
before — the parliament lang uses a **vocab-limited combined dict** (stage 0).
Keep `seg_nj` modest (8), and if OOM: `journalctl` for the oom-killer first,
don't assume a crash.

**Watch the harvest's mismatch warning:** if phone-map mismatch >10%, the
`prons.*.gz` were written against a different `words.txt`/`phones.txt` than the
lang — re-run `get_prons.sh` against the *current* lang before trusting labels.

---

## Stage 2 — Vet the harvested labels (quality gate)

Forced alignment is noisy; not every chosen variant is right. The A/B lesson
from the imperative cycle applies: **the aligner seeds, the ear gates.**

```bash
./venv/bin/python vet_stress_labels.py \
    --labels data/kaldi/stress_labels_parliament_full.jsonl ...   # sample by word
# spot-listen at (audio_path, t0) for the high-value homographs; the harvest
# already emits t0/dur per token for exactly this.
```

- Prioritise **words with BOTH variants observed** (the harvest prints the top
  by evidence) — those are the genuine contextual homographs, the whole point.
- Discard word types where the split looks like alignment noise (e.g. one
  variant <5% with no plausible context).
- Native spot-check a stratified sample per word type; the ear-verified set is
  what feeds training.

---

## Stage 3 — Fold into the Accentor training set + fine-tune

Merge the vetted parliament labels with the existing label sources
(`stress_labels_{milestone_a,ab400,parliament,parliament2}.jsonl`, text rows,
and the `adapt_teacher_rows.py` BgGPT rows) into the marked-sentence format
`train_accentor.py` consumes. Then **fine-tune from the current production
Accentor** (don't retrain from scratch — preserve what ft6b learned):

```bash
./venv/bin/python train_accentor.py \
    --data data/chitanka/marked_plus_parliament_full.jsonl \
    --init exp/accentor_ft6b/best.pt \
    --out exp/accentor_ft7 --steps 40000
```

- **Oversample the homograph tokens** (they're rare vs plain words) — same
  minority-oversampling that helped calibration before.
- Keep the plain-word majority so the model doesn't forget non-homograph G2P.

---

## Stage 4 — Evaluate on the axis that matters

```bash
./venv/bin/python eval_accentor.py     --ckpt exp/accentor_ft7/best.pt   # in-domain
./venv/bin/python eval_accentor_ood.py --ckpt exp/accentor_ft7/best.pt   # OOD (the real test)
# + the homograph battery (config/bg/homograph_battery*.json)
```

**Success = the homograph battery and OOD accuracy improve** over ft6b, with no
regression on plain-word stress. The battery is the headline number: the pilot
showed the lexicon feeds the wrong mark in 6/8 homograph cases; the target is to
close those with contextual labels. Parliament's *domain diversity* is the
lever — it should help most on homographs whose disambiguating context the
audiobook corpus never contained (political/formal vocabulary).

---

## Stage 5 — Promote

If the battery + OOD improve and plain-word stress holds:

```bash
# serve_bg.sh / serve.py --accentor
ACCENTOR=exp/accentor_ft7/best.pt bash serve_bg.sh
```

Update `read_book.py --accentor` default and the served checkpoint together so
offline dogfood and the LAN server stay in lockstep.

---

## Expected impact & honest bounds

- **Where it helps:** homograph mark-correctness on varied text — measurably, on
  the battery/OOD. This is the axis parliament uniquely feeds.
- **Where it does NOT help:** imperative rendering, ч/щ brightness, naturalness
  (all acoustic — the TTS fine-tune owns those). Don't expect ft8-style wins
  here; expect fewer wrong-stress homographs on general prose.
- **Byproduct, not this cycle:** the same segmented+aligned parliament corpus is
  the training set for the **ASR deliverable** (Zipformer/icefall) — bank it,
  scope it separately.

## Effort / sequencing

1. ft8 validated & promoted  ← current cycle
2. Stage 1 harvest (GPU-light, CPU-heavy Kaldi; hours-to-overnight on full corpus)
3. Stage 2 vetting (native time — the gating human step)
4. Stage 3 fine-tune (Accentor is small; ~1 h) → Stage 4 eval → Stage 5 promote
