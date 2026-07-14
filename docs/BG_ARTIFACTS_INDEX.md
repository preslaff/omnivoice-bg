# OmniVoice-BG — Analytical Artifacts Index

One-stop map of everything the project produced: what it is, where it
lives, and what it proved. Gitignored bulk (data/, exp/, eval outputs)
exists only on the training box — back it up accordingly.

## Documents & notebooks

| Artifact | What it is |
|---|---|
| `OMNIVOICE_BG.md` | Front page: architecture of the fork, serving, TrainGen integration |
| `BG_FINETUNE_PLAN.md` | TTS fine-tune deep-dive (recipes, configs, A/B history) |
| `docs/BG_PIPELINE_GUIDE.md` | Living stage-by-stage guide of the whole label→Accentor→TTS pipeline |
| `docs/BG_LESSONS.md` | **Distilled lessons L1–L16** (the education document) |
| `docs/BG_FINAL_EFFORT_PLAN.md` | Plan for the milestone-A final effort (this cycle) |
| `docs/BG_PARLIAMENT_INGESTION.md` | Parliament crawler design (enumeration, health probe, resume) |
| `docs/BG_Accentor_Paper.ipynb` | Paper foundation: live cells reproduce every table/figure, findings F1–F11 |

## Corpora (gitignored, on-disk)

| Artifact | Contents | Proved / used for |
|---|---|---|
| `data/bg/` ("sivov") | 63 h single-narrator audiobooks, train/dev JSONL | Original TTS fine-tune + Kaldi pilot seed |
| `data/bg400/` | 195,701 rows ≈ 400 h same narrator (HF shards, wget-resumed) | Full-scale alignment + TTS scale-up |
| `data/bg_parliament/` | Parliament sessions: FLAC + stenograms + manifest (316.8 h @ 101 sessions, crawl ongoing to ≥400 h) | Milestone-A acoustic retrain; OOD eval text |
| `data/bg400full/tokens/` | Higgs-codec tokens: 448.3 h train / 4.5 h dev, ft5b-marked twins | The 413 h TTS run (`omnivoice_bg_full`) |
| `data/bg_lexicon/bg_stress.sqlite` | rechko-derived stress lexicon | Baseline (non-contextual) marking; G2P seed |

## Labels, vetting, judge outputs

| Artifact | Rows | What it is |
|---|---|---|
| `data/kaldi/stress_labels_ab400.jsonl` | 1,215,956 | **Canonical harvest** (full 400 h, tri3b_comb). Supersedes `stress_labels.jsonl` — never stack both |
| `data/kaldi/stress_labels_parliament2.jsonl` | 210,802 | Parliament harvest v2 (segmented) |
| `data/kaldi/vetting_results.jsonl` | 306 | Native per-token verdicts → ft5b veto (169 dropped) |
| `bon_eval/` | 224 wavs | Best-of-8 oracle experiment: manifest, per-candidate scores, `picked/` A/B pairs |
| `exp/kaldi_pilot/` | — | Kaldi workspace: GMM ladder (mono→tri3b_comb), lang dir, alignments, prons |

## Models

| Artifact | Selection | Status / role |
|---|---|---|
| `exp/accentor_ft5b/best.pt` (11 K steps) | wild-text reads + battery (val-SER blind, F11) | **SERVED Accentor**; battery 15/25, OOD SER 0.020 % |
| `exp/accentor_ft5b/last.pt` (20 K) | — | battery 16/25 but wild чета/купи regressions; kept for study |
| `exp/accentor_ft{1..5}/`, `accentor_text*/` | — | Generation history behind F2/F3/F4 coverage laws |
| `exp/bg_keep/checkpoint-2750` | — | Base BG TTS fine-tune (init for all later runs) |
| `exp/omnivoice_bg_ft5b/checkpoint-2750` | — | 63 h TTS + ft5b marks (marking-consistency refresh) |
| `exp/omnivoice_bg_full/checkpoint-6500` | eval loss 3.4315 + A/B by ear | **Best TTS**: 413 h + ft5b marks |
| HF `preslaff/omnivoice-bg` | — | Published model repo |

## Evaluations (chronological)

| Artifact | Question | Verdict |
|---|---|---|
| `eval_ab/`, `eval_ab_ft5b/` | stock vs fine-tunes; ft4 vs ft5b marks | Marking consistency saturated (but early A/Bs used rechko marks — see `--accentor` flag) |
| `eval_ab_border.txt` + `eval_ab_border/` | 28 minimal-pair sentences for boundary homographs | The standing wild-eval set |
| `eval_ab_full/` | 63 h vs 413 h acoustics, identical ft5b marks | 6500 best; volume saturated (F12) |
| `bon_eval/` | Does sampling contain correct renderings? | single 12/27 → judge 19/27 = oracle; prior dominance on rare classes (F13/F14) |
| `config/bg/homograph_battery*.json` | 25-item Accentor battery + sentences | Checkpoint selection; progression 11→17 across generations |

## Configs (`config/bg/`)

| File | Role |
|---|---|
| `bg_stress_overrides.json` | 67 human override entries (incl. проме́ните) — last-resort veto, served |
| `function_words_unstressed.json`, `always_final_imperatives.json` | Rule layers of the marking front-end |
| `homograph_verbs.json`, `lexicon_supplement.json` | Lexicon curation inputs |
| `{train,data}_config_bg*.json` | TTS recipes: base / `_marked` / `_combined` (4:1 mix) / `_full` (413 h) |

## Scripts by pipeline stage

- **Acquisition**: `prep_bg_parliament.py` (+ `refresh_bg_parliament.py`), `prep_bg400.py` (shard fast-forward, `--shards-dir` offline), `prep_bg_hf.py`, `prep_chitanka.py`
- **Lexicon**: `build_bg_lexicon.py`, `build_lexicon_phones.py`, `extend_lexicon_coverage.py`, `lexicon_coverage.py`, `build_stress_candidates.py`
- **Kaldi**: `prep_kaldi_data.py`, `kaldi_pilot.sh`, `kaldi_parliament_pilot.sh`, `kaldi_milestone_a.sh` (pre-written, gated on ≥400 h parliament)
- **Harvest & vetting**: `harvest_stress_labels.py`, `vet_stress_cli.py`, `vet_stress_labels.py`, `mine_homographs.py`
- **Accentor**: `build_accentor_finetune.py` (tiers + `--vet-results` veto), `train_accentor.py`, `accentor_model.py`, `eval_accentor.py`, `eval_accentor_ood.py`
- **TTS**: `tokenize_bg400_full.sh`, `mark_tts_txts.py`, `train_tts_full.sh`, `train_tts_ft5b.sh`
- **Eval & judge**: `eval_bg_ab.py` (`--accentor` = contextual marks), `bon_generate.py`, `bon_judge.sh`, `bon_score.py`
- **Serving**: `omnivoice/cli/serve.py`, `serve_bg.sh`, `bg_accent.py`
