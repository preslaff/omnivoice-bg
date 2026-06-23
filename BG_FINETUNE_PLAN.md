# Fine-tuning OmniVoice for Bulgarian — Working Plan

A step-by-step, followable plan to fine-tune `k2-fsa/OmniVoice` on Bulgarian
speech so narration quality (pronunciation, stress, prosody) improves for the
TrainGen videos. This is **Path A** from our earlier discussion: we keep the
Qwen3-0.6B backbone and the pretrained multilingual acoustic model, and
specialize it on Bulgarian data — we do **not** swap the backbone or train from
scratch.

> Companion context: voice *identity/consistency* is already solved at inference
> time by fixed-reference cloning (see `omnivoice/cli/serve.py` + TrainGen's
> `RemoteOmniVoiceBackend`). Fine-tuning is a **different lever** — it improves
> how good the Bulgarian *sounds*, not which voice is used. After fine-tuning,
> the same cloning mechanism still applies; you just point the server at the new
> checkpoint.

---

## 0. Decision gate — do we even need this?

Fine-tuning costs time + data. Only proceed if the **stock model's Bulgarian is
not good enough** with the fixed-reference + deterministic setup already in place.

- [ ] Generate ~20 representative narration sentences with the current setup.
- [ ] Listen for **real Bulgarian errors**: wrong word stress, mispronounced
      words, unnatural intonation, foreign accent leaking in.
- [ ] If errors are **transcript/punctuation** issues → fix those first (cheaper).
- [ ] If errors are **inherent to the model's Bulgarian** → fine-tuning is justified.

**Outcome of this gate:** a short list of concrete failure examples. Keep them —
they become your subjective test set in Phase 6.

---

## 1. The pipeline at a glance

```
 HF dataset raditotev/bg-audiobooks-tts (15.2h, 1 speaker, normalized)
        │  Phases 1-2: prep_bg_hf.py  (stream → WAV clips + JSONL)
        ▼
 train.jsonl / dev.jsonl   ({id, audio_path, text, language_id:"bg"})
        │  Phase 3: extract_audio_tokens  (audio → 8-codebook tokens)
        ▼
 WebDataset shards (.tar) + per-shard .jsonl + data.lst
        │  Phase 4: write train_config + data_config
        ▼
 Phase 5: accelerate launch -m omnivoice.cli.train  → checkpoints
        │  Phase 6: evaluate (listen + WER)
        ▼
 Phase 7: serve.py --model <checkpoint>   → TrainGen uses it
```

Each phase below maps to one section.

---

## 2. Phases 1–2 — Data (DONE via the HF dataset)

We use a ready, public-domain dataset that already covers collection,
segmentation, transcription, and normalization:

**[`raditotev/bg-audiobooks-tts`](https://huggingface.co/datasets/raditotev/bg-audiobooks-tts)**

| Property | Value |
|---|---|
| Size | 10,627 clips, **15.2 h** |
| Speaker | **1** (Plamen Sivov) — ideal for a single consistent narrator |
| Clips | 3–15 s, utterance-sized (no segmentation needed) |
| Audio | **24 kHz mono** — matches OmniVoice, no resampling |
| Text | `text` (raw Whisper) + **`text_normalized`** (numbers/dates expanded) |
| License | **CC0-1.0** (public domain) |

This replaces the original collect → segment → transcribe → hand-correct →
normalize workflow. We ingest it straight into OmniVoice's JSONL format with
`prep_bg_hf.py` (uses `text_normalized`, `language_id: "bg"`, deterministic
~5% dev hold-out, streamed so it won't load all 2.6 GB into RAM):

```bash
./venv/bin/pip install datasets          # one-time

# Smoke test first (50 clips), then the full run:
./venv/bin/python prep_bg_hf.py --out data/bg --limit 50
./venv/bin/python prep_bg_hf.py --out data/bg
#   first run only ~5-6 h?  ->  --limit 4000
```

Produces:
- `data/bg/clips/*.wav`
- `data/bg/train.jsonl`, `data/bg/dev.jsonl` — each line:
  `{"id": "...", "audio_path": "<abs>", "text": "<normalized>", "language_id": "bg"}`

**Disk:** the WAVs add ~2–3 GB on top of the ~2.6 GB parquet cache → keep ~6 GB free.

- [x] `./venv/bin/pip install datasets` done (datasets 5.0.0).
- [x] Smoke test (`--limit 50`) produced sane clips + JSONL (47/3, 24 kHz mono PCM_16).
- [x] Full ingest done: 10,627 clips (train 10,095 / dev 532), 0 missing, 0 empty.

> **Other sources** (only if you later want more data / multi-speaker): Mozilla
> Common Voice (bg), FLEURS (bg), CoVoST, or your own recordings — each then needs
> the segment/transcribe/correct/normalize work this dataset already did for you.

---

## 4. Phase 3 — Tokenize audio into WebDataset shards

This encodes each clip into 8-codebook discrete tokens with the Higgs audio
tokenizer and packs them into `.tar` shards (`docs/data_preparation.md`).

```bash
cd ~/WORK/OmniVoice

# Train split
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl data/bg/train.jsonl \
    --tar_output_pattern data/bg/tokens/train/audios/shard-%06d.tar \
    --jsonl_output_pattern data/bg/tokens/train/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 2 --shuffle True --skip_errors \
    --min_length 1.0 --max_length 25.0

# Dev split
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl data/bg/dev.jsonl \
    --tar_output_pattern data/bg/tokens/dev/audios/shard-%06d.tar \
    --jsonl_output_pattern data/bg/tokens/dev/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 2 --shuffle True --skip_errors
```

Produces `data/bg/tokens/{train,dev}/data.lst` (the manifest training reads).
`--min_length/--max_length` drop pathological clips; `--skip_errors` keeps the
run going past a bad file (check the emitted `errors.jsonl`).

- [ ] `data/bg/tokens/train/data.lst` exists and lists shards.
- [ ] `data/bg/tokens/dev/data.lst` exists.
- [ ] Reviewed `errors.jsonl` (expect it to be empty/small).

---

## 5. Phase 4 — Configuration (tuned for one 16 GB GPU)

> **Already created** in `config/bg/` — the two files below are committed ready to
> use; this section explains them. Adjust only if Phase 5 needs memory/quality tuning.

### 5.1 Data config — `config/bg/data_config_bg.json`

```json
{
    "train": [
        { "language_id": "bg", "manifest_path": ["data/bg/tokens/train/data.lst"], "repeat": 1 }
    ],
    "dev": [
        { "language_id": "bg", "manifest_path": ["data/bg/tokens/dev/data.lst"], "repeat": 1 }
    ]
}
```

### 5.2 Train config — `config/bg/train_config_bg.json`

Start from `examples/config/train_config_finetune_sdpa.json` (SDPA is the
compatible choice on a single consumer GPU) and adjust for 16 GB:

```json
{
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "num_audio_codebook": 8,

    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.0, 0.3],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 0.8,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,

    "resume_from_checkpoint": null,
    "init_from_checkpoint": "k2-fsa/OmniVoice",

    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": 3000,
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.03,
    "warmup_steps": 0,

    "batch_tokens": 1024,
    "gradient_accumulation_steps": 8,
    "num_workers": 2,

    "mixed_precision": "bf16",
    "allow_tf32": true,
    "attn_implementation": "sdpa",
    "max_sample_tokens": 2000,
    "min_sample_tokens": 50,
    "max_batch_size": 32,

    "logging_steps": 25,
    "eval_steps": 250,
    "save_steps": 250,
    "keep_last_n_checkpoints": 5
}
```

**Why these values:**

| Field | Choice | Reason |
|---|---|---|
| `init_from_checkpoint` | `"k2-fsa/OmniVoice"` | load pretrained weights — the whole point of Path A |
| `attn_implementation` | `"sdpa"` | broad GPU compatibility; flex_attention is faster but pickier |
| `batch_tokens` | **1024** (↓ from 8192) | fits 16 GB alongside fp32 weights+grads+Adam (~9.6 GB static); 2048 OOM'd by ~20 MB |
| `gradient_accumulation_steps` | **8** | restores an effective batch ≈ 8192 tokens despite small micro-batch |
| `learning_rate` | `1e-5` | conservative for fine-tuning; raise to `5e-5` only if it underfits |
| `steps` | **3000** | a few hours of data fine-tunes fast; watch `eval/loss`, don't overtrain |
| `instruct_ratio`/`only_instruct_ratio`/`use_pinyin_ratio` | `0.0` | no voice-design/pinyin augmentation — pure cloning/auto fine-tune (matches the stock finetune recipe; irrelevant for Bulgarian) |
| `language_ratio` | `0.8` | 80% of samples carry the `bg` language tag during training |
| `keep_last_n_checkpoints` | `5` | rotate to save disk (was `-1` = keep all) |

> **Memory math (measured on the 16 GB RTX 5070 Ti):** the backbone loads in
> **fp32** for training, so static memory is fp32 weights + grads + Adam states
> ≈ **~9.6 GB** before any activations — leaving only ~6 GB. `batch_tokens=2048`
> OOM'd by ~20 MB on the second (larger) batch; **1024 × grad_accum 8** keeps the
> same effective batch (~8192) with half the activation memory and real headroom.
> Always launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (defrag).
> If `nvidia-smi` shows headroom, you can try raising `batch_tokens` back to 1536/2048
> (drop `gradient_accumulation_steps` to keep the product steady); if it OOMs, go to
> 512 × 16.

- [x] `config/bg/data_config_bg.json` created.
- [x] `config/bg/train_config_bg.json` created.

---

## 6. Phase 5 — Launch training

> **Free the GPU first:** stop the `serve.py` TTS server — you can't train and
> serve on the same 16 GB card at once.

Single GPU:

```bash
cd ~/WORK/OmniVoice
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./venv/bin/accelerate launch \
    --gpu_ids 0 --num_processes 1 \
    -m omnivoice.cli.train \
    --train_config config/bg/train_config_bg.json \
    --data_config config/bg/data_config_bg.json \
    --output_dir exp/omnivoice_bg
```

> The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prefix reduces allocator
> fragmentation — needed on the 16 GB card (the run OOM'd by ~20 MB without it).

**Metrics / telemetry.** The trainer auto-logs to **TensorBoard** (Accelerate
`log_with="tensorboard"`): `train/loss`, `train/learning_rate`, `train/grad_norm`
(every `logging_steps`) and `eval/loss` (every `eval_steps`). Event files →
`exp/omnivoice_bg/tensorboard/`. There's also a live tqdm bar and a plain-text
`exp/omnivoice_bg/train.log`. The `tensorboardX` *writer* ships with the repo, but
the *viewer* must be installed once:

```bash
./venv/bin/pip install tensorboard
```

Monitor in a second terminal (bind to 0.0.0.0 to view from another LAN machine,
e.g. your laptop at http://192.168.78.101:6006):

```bash
./venv/bin/tensorboard --logdir exp/omnivoice_bg/tensorboard --host 0.0.0.0 --port 6006
watch -n 5 nvidia-smi                                  # VRAM headroom
```

**What to watch:**
- `train/loss` should fall then flatten. `eval/loss` (every `eval_steps`) is the
  real signal — when it **stops improving or rises**, you're done / overfitting.
- `train/grad_norm` spikes → instability; if it explodes, lower `learning_rate`.
- Checkpoints land in `exp/omnivoice_bg/checkpoint-<step>/` (HF format:
  `config.json` + safetensors + tokenizer + a copy of `train_config.json`).

- [ ] Training launched; loss decreasing.
- [ ] `eval/loss` tracked; picked best checkpoint step = **____**.

---

## 7. Phase 6 — Evaluate

Two complementary checks:

1. **Subjective (primary).** Re-synthesize the Phase 0 failure sentences with the
   new checkpoint (Phase 7 shows how to load it) and compare side-by-side against
   the stock model. Did the specific stress/pronunciation errors improve?
2. **Objective WER.** Run a Bulgarian ASR (e.g. Whisper) over generated dev-set
   sentences and compare the transcription to the input text — lower WER = clearer
   speech. (The repo's `run_eval.sh` targets EN/ZH sets; for Bulgarian a simple
   Whisper-WER loop on your held-out sentences is the practical metric.)

- [ ] Subjective A/B on the Phase 0 sentences done; clear improvement?
- [ ] (Optional) WER on dev set: stock = ____ → finetuned = ____.

> If quality regressed or barely moved: lower `learning_rate` and/or reduce
> `steps` (overfitting), or collect more data (underfitting). See Troubleshooting.

---

## 8. Phase 7 — Deploy the fine-tuned model

Point the LAN server at the checkpoint directory — `from_pretrained` loads it
like any OmniVoice model (the audio tokenizer falls back to the public one
automatically, so no extra files needed):

```bash
./venv/bin/python -m omnivoice.cli.serve \
    --model exp/omnivoice_bg/checkpoint-<best> \
    --host 0.0.0.0 --port 8002 --seed 42
```

Then:
- [ ] **Regenerate the reference clips** (`assets/voices/voice_bg_calm.wav`) from
      the fine-tuned model, so the cloned voice reflects the improved Bulgarian.
- [ ] Verify TrainGen narration end-to-end with `TRAINGEN_TTS=omnivoice_remote`.
- [ ] Keep the chosen checkpoint backed up; archive the rest.

---

## 9. Hyperparameter & troubleshooting guide

| Symptom | Likely cause | Adjustment |
|---|---|---|
| CUDA OOM at start of training | batch/optimizer too big for 16 GB | lower `batch_tokens` (1536/1024), raise `gradient_accumulation_steps` to compensate |
| `eval/loss` rises while `train loss` falls | overfitting (small data, too many steps) | fewer `steps`, lower `learning_rate`, more data |
| Quality barely changed | underfitting / LR too low | raise `learning_rate` to `5e-5`, more `steps`, more data |
| Pronunciation improved but voice drifts | expected — fine-tuning ≠ identity | rely on fixed-reference cloning (Phase 7) |
| `flex_attention` errors on launch | backend unsupported in env | already using `sdpa`; keep it |
| Garbled/robotic output after training | catastrophic forgetting (LR too high / too many steps) | lower LR, fewer steps, ensure `init_from_checkpoint` set |
| Wrong stress on specific words | transcript/normalization issue in data | fix those transcripts, re-tokenize, re-train |

**Golden rules:**
- Change **one thing at a time** and re-listen.
- The dev `eval/loss` curve decides when to stop — not a fixed step count.
- Garbage transcripts in → garbage prosody out. Data quality > step count.

---

## 10. Master checklist

- [ ] **Phase 0** — confirmed stock BG quality is insufficient (have failure examples)
- [x] **Phases 1–2** — dataset chosen (raditotev/bg-audiobooks-tts, 15.2h, CC0)
- [x] **Phases 1–2 run** — ingested 10,627 clips → `data/bg/{train,dev}.jsonl` (10,095 / 532)
- [x] **Phase 3** — tokenized: train 33 shards/10,095, dev 34 shards/532, 0 errors
- [x] **Phase 4** — `config/bg/{train,data}_config_bg.json` created (validated)
- [ ] **Phase 5** — training run complete; best checkpoint chosen
- [ ] **Phase 6** — evaluated (subjective A/B + optional WER)
- [ ] **Phase 7** — deployed via `serve.py --model`; references regenerated; TrainGen verified

---

## 11. Open questions

Resolved by choosing the HF dataset:
- ~~How many hours / how many speakers?~~ → **15.2 h, single speaker** (Plamen Sivov).
- ~~Source?~~ → **raditotev/bg-audiobooks-tts** (CC0 audiobooks).
- ~~Who corrects the transcripts?~~ → **already normalized** (`text_normalized`).

Still to keep in mind:
1. **Single 16 GB GPU** — training and serving can't overlap; stop `serve.py`
   before Phase 5 and plan a maintenance window.
2. **Use all 15.2 h or subset first?** A `--limit 4000` (~5–6 h) first run is
   faster to iterate; scale to the full set if Phase 6 shows it's needed.

The rest of the plan is now mechanical: run `prep_bg_hf.py` → tokenize → launch.
