# Bulgarian ASR — icefall / Zipformer recipe

A Zipformer transducer trained on the clean 649 h BG corpus (parliament + audiobook)
built by the Kaldi-aligned pipeline. The benchmark is spontaneous **parliament** speech
(16.4 h test / 6.2 h dev, whole-session holdout, no leakage); audiobook is train-only.
See [`../docs/BG_ASR_PLAN.md`](../docs/BG_ASR_PLAN.md).

## Data prep (this repo)

```bash
./venv/bin/python detect_stale_sessions.py     # quarantine refresh/segment-mismatched sessions
./venv/bin/python make_asr_split.py            # whole-session train/dev/test split (clean only)
./venv/bin/python prep_lhotse_manifests.py     # -> data/lhotse/bg/cuts_{train,dev,test}.jsonl.gz
./venv/bin/python compute_fbank_bg.py          # -> data/fbank/bg_cuts_{train,dev,test}.jsonl.gz
./venv/bin/python train_bpe_bg.py              # -> data/lang_bpe_500/{bpe.model,tokens.txt}
```

## Scaffold the icefall recipe

`icefall` is a repo on `PYTHONPATH`, not a pip package. With `~/icefall` cloned:

```bash
ICEFALL=~/icefall bash asr/setup_icefall_bg.sh
```

This creates `~/icefall/egs/bg_parliament/ASR/` = a copy of the librispeech zipformer
scripts + our `zipformer/asr_datamodule.py` (BG cuts, MUSAN off) + symlinks to
`data/fbank` and `data/lang_bpe_500`. Idempotent — safe to re-run after data changes.

## Train (one RTX 5070 Ti, 16 GB)

```bash
export PYTHONPATH=~/icefall:$PYTHONPATH
cd ~/icefall/egs/bg_parliament/ASR

python zipformer/train.py \
  --world-size 1 --num-epochs 30 --start-epoch 1 \
  --use-fp16 1 --causal 0 \
  --exp-dir zipformer/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --max-duration 350 --num-workers 2
```

- `--max-duration 350`: conservative for 16 GB fp16. Raise (400–500) if VRAM heads-room,
  lower on CUDA OOM. Run under `tmux` — this is a multi-day from-scratch run on one GPU.
- Non-streaming (`--causal 0`) for the best offline WER benchmark.
- Stock `train.py` filters cuts to 1–20 s, so the few 20–30 s parliament segments are
  dropped from training (kept in dev/test only if ≤ the same filter at decode time).
- Monitor: `tensorboard --logdir zipformer/exp/tensorboard`.

## Decode (after training)

```bash
python zipformer/decode.py \
  --epoch 30 --avg 5 --use-averaged-model 1 \
  --exp-dir zipformer/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --max-duration 300 --decoding-method modified_beam_search
```

WER lands in `zipformer/exp/modified_beam_search/`. For n-gram-rescored numbers, rebuild
the KenLM against the *new* held-out split first (`prep_lm_text.py` + `lmplz`, see plan).

## Recover the quarantined 110 sessions (v2 data boost)

`detect_stale_sessions.py` stages `data/bg_parliament/manifest_stale.jsonl`. Re-segment
those against the current audio with the existing pipeline
(`SET=parl_refresh bash segment_asr_sessions.sh` after pointing it at that manifest),
then re-run the data-prep chain above to fold ~250–330 h back in.
