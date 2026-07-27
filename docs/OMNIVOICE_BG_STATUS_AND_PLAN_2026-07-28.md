# OmniVoice-BG — Status & Forward Plan

*Snapshot: 2026-07-28.* Companion docs: `ACCENTOR_REFRESH_PLAN.md` (Track A
detail), `IMPERATIVE_DATASET.md` (imperative data), `BG_FINAL_EFFORT_PLAN.md`
(milestone-A history), `deploy/README.md` (collector deployment).

## Context

We paused after the ft7/ft8 imperative experiments hit a **voice-generalization
ceiling** — imperative stress is learned *per-voice*, not as a voice-independent
text→stress rule, so прави́/ходи́/купи́ render correctly only with the voice they
trained in. Meanwhile the Parliament crawl kept banking sessions. This plan
fixes the exact new-data count, states where every track stands, and sets two
independent forward tracks: **harvest the new Parliament audio → refresh the
Accentor** (marking), and **collect many volunteer voices → break the imperative
ceiling** (acoustic).

## 1. Where we stand

| Component | Shipped / best | State |
|---|---|---|
| TTS (production) | `exp/omnivoice_bg_e3_sparse/checkpoint-3000` | bright T5 on sparse marks; served via `serve_bg.sh` |
| Accentor (production) | `exp/accentor_ft6b/last.pt` | battery 16/25, OOD SER ~0.005% |
| Stress pins | `config/bg/bg_stress_overrides.json` (79) | still needed — imperatives + proper names |
| Kaldi label factory | `exp/kaldi_pilot/exp/{tri2b_comb,tri4b,nnet3/tdnn_a}` | **built** (milestone A) — fixed factory |
| Imperative acoustic | `exp/omnivoice_bg_ft{7,8}/checkpoint-3000` | **not promoted** — voice ceiling |
| Parliament corpus | 200 sessions / **707.7 h** downloaded | **new harvest running** (2026-07-28) |

**The ceiling, precisely:** from few clips per (verb, voice) the model memorizes
rather than abstracts. обади fires for BgTTS voices but not the user's;
прави́/ходи́ (human, one voice) fire for the user's voice but not the reklama
clone. Root cause = **too few speakers per imperative** → Track B.

## 2. New Parliament data (as of 2026-07-28)

Diff of `data/bg_parliament/manifest.jsonl` vs harvested session ids:

| Bucket | Sessions | Hours |
|---|---|---|
| Downloaded total | 200 | 707.7 |
| Already harvested (milestone-A labels) | 119 | 408.8 |
| **NEW — unharvested** | **81** | **298.9** |
| ↳ effective (≥3 min, ≥5 char/s) | 72 | 287.8 |

**≈288 h of new effective audio**, processed by the existing factory with zero
retraining (F2: the aligner is saturated). **Harvest is running now.**

## 3. Track A — Accentor-refresh (marking; retire pins)

Orthogonal to acoustics — fixes which stress the front-end *marks*. Full detail
in `ACCENTOR_REFRESH_PLAN.md`. Steps (user runs the big commands):

1. **Harvest** — `KALDI_ROOT=~/kaldi bash harvest_new_sessions.sh` *(running)*.
   → `stress_labels_incr_<date>.jsonl`, disjoint from milestone-A. Watch
   `seg_nj=8`, `journalctl` for OOM, phone-map mismatch <10%, segmentation yield.
2. **Imperative label set (text-only)** — fold curated imperative marks (synth
   bank + human set) so the Accentor learns to mark купи́/прави́ in imperative
   context while corpus keeps ку́пи/пра́ви aorist.
3. **Rebuild + fine-tune** — `build_accentor_finetune.py` over all
   `stress_labels_*.jsonl` + chitanka → `train_accentor.py --init
   exp/accentor_ft6b/best.pt --out exp/accentor_ft7`.
4. **Evaluate** — `eval_accentor{,_ood}.py` + homograph battery + a new
   imperative battery (imperative vs aorist). Success = homograph/OOD improve,
   imperatives marked in context, aorist intact, no plain-word regression.
5. **Promote + thin pins** — update `serve_bg.sh` ACCENTOR; drop imperative pins
   the Accentor now handles; keep proper-name pins.

## 4. Track B — Volunteer many-voice recording (acoustic; break the ceiling)

Collect the same marked imperatives from **many volunteers**; real speaker
diversity is what the ft8 post-mortem identified as missing. **Tooling built and
smoke-tested (2026-07-28):**

- **`omnivoice/cli/collect.py`** (`omnivoice-collect`) — self-contained FastAPI
  mic-recording web app. No GPU / no ML deps (fastapi + uvicorn + multipart +
  ffmpeg). Serves marked prompts from `imperative_verbs.txt`×`imperative_frames.txt`,
  **ending-stress class first**, per-speaker frame variety; duration gate;
  resumes per speaker; `/health` + `/stats`. Illustrative examples shown on the
  landing page.
- **`convert_volunteer.py`** — decode → 24 kHz mono → full QC (RMS floor +
  clipping + duration) → peak-normalise → marked `train.jsonl` + prints the GPU
  tokenize command (**no `--brighten`**).
- **`config/bg/data_config_bg_ft9.json`** — base bg + BgTTS synth + human +
  **volunteer set** (repeat tuned to volume); reuses `train_config_bg_ft7.json`.

Deploy: behind nginx TLS at **`omnivoice.cdsv.dev`** (`deploy/` — HTTPS is
mandatory for the mic). Collect on the VPS → rsync to the GPU box →
`convert_volunteer.py` → tokenize → train `exp/omnivoice_bg_ft9`.

**Validate ft9 across ≥3 held-out reference voices** with `sample_imperatives.py`
— voice-invariance (ending-stress for *unseen* clones) is the pass bar.

## 5. Sequencing & loop-close

- Tracks A and B are **independent and parallel**. A is orchestration-only
  (running). B's long pole is volunteer recording time.
- Crawl resume is optional/background — 288 h already saturates the aligner.
- **The imperative loop closes when both land:** Accentor *marks* купи́/прави́
  correctly (A) **and** ft9 *renders* the mark across voices (B) → pins retired
  end-to-end. Until then, BgTTS-for-ads routing (already wired in `serve.py`) is
  the pragmatic fallback for imperative-heavy copy.

## 6. Verification

- **Count:** re-run the manifest-vs-labels diff — NEW→0 after the harvest.
- **Track A:** imperative + homograph batteries + OOD SER before/after ft7;
  confirm pins droppable without `read_book.py` regressions.
- **Track B app:** `/health` + one end-to-end submit (clip in `raw/`, row in
  `submissions.jsonl`, gate passes); `convert_volunteer.py` emits 24 kHz mono.
- **Track B model:** `sample_imperatives.py` across ≥3 unseen voices — ending
  stress on прави́/ходи́/купи́ must hold for voices it never trained on.
