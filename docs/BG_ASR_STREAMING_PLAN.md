# GP-S — streaming (causal) Bulgarian ASR

**Status: prepared, not launched (2026-08-08).** Roadmap phase 3, promoted to the current
priority ahead of self-training round 2 and the playground rollout.

## Why this is a separate training run

GP-v2 is **offline**: its training log says `"causal": false`, so the encoder attends over the
whole utterance including future audio. That is why it reaches 4.12 %, and it is also why it
can never emit words mid-sentence. **A non-causal checkpoint cannot be converted to a streaming
one** — `export-onnx-streaming.py` requires a causal model, which is why only the offline
`export-onnx.py` path worked for the playground.

What causal training changes: chunk-wise attention with a bounded left context, and causal
padding in the convolution modules. Parameter *shapes* are unchanged; what changes is what the
model is allowed to see.

## Configuration

Exactly one variable moves versus GP-v2 — `--causal` — so the delta is attributable, following
the same discipline as the v1 → GP-v2 comparison.

| | GP-v2 (offline) | **GP-S (streaming)** |
|---|---|---|
| `--causal` | 0 | **1** |
| `--chunk-size` | *(inert)* | `16,32,64,-1` |
| `--left-context-frames` | *(inert)* | `64,128,256,-1` |
| Corpus | `bg_cuts_train_gp_shuf.jsonl.gz` (1,744.8 h) | identical |
| Architecture | Zipformer-M, 65.5 M params | identical |
| BPE / features | BPE-500, 80-dim fbank @ 16 kHz | identical |
| MUSAN, fp16, epochs, `--max-duration` | on, on, 30, 350 | identical |
| exp dir | `zipformer/exp_gp` | `zipformer/exp_gp_stream` |

Chunk size and left context are **sampled randomly per batch** from those lists. That is the
point of the lists: one checkpoint then serves multiple latency settings at decode time, chosen
per request, instead of committing to a single latency at training time. `-1` means "unbounded"
and keeps a full-context mode available in the same model.

## Launch

```bash
tmux new -s train_s
EXP=zipformer/exp_gp_stream CAUSAL=1 ENABLE_MUSAN=true \
  TRAIN_MANIFEST=bg_cuts_train_gp_shuf.jsonl.gz \
  bash ~/WORK/OmniVoice/asr/train_resumable.sh
```

`train_resumable.sh` resumes from the highest `epoch-N.pt`, so a crash or reboot costs at most
the epoch in progress. After a reboot, run the same line again.

Preconditions checked 2026-08-08: GPU idle (332 MiB / 16 GB), **362 GB free** on `/` against
~61 GB needed (`exp_gp` is that size), corpus manifest and MUSAN in place.

## Expected outcome

- **~1–2 points absolute WER worse than offline** — the published price of giving up right
  context. Expect roughly 5–6 % on the frozen parliament test against GP-v2's 4.12 %. That is a
  success, not a regression: it buys live decoding, which offline cannot do at any accuracy.
- Wall clock ≈ GP-v2's, ~1.5–2 days for 30 epochs on the RTX 5070 Ti.
- Memory: causal masking does not increase activation size, so `--max-duration 350` should hold.
  Watch the first epoch; drop to 300 if it OOMs.

## Evaluation (do not reuse the offline commands)

Streaming models must be decoded with the streaming decoder, at the latency you intend to ship:

```bash
cd ~/icefall/egs/bg_parliament/ASR
python zipformer/streaming_decode.py \
    --epoch 30 --avg 5 --use-averaged-model 1 \
    --exp-dir zipformer/exp_gp_stream --causal 1 \
    --chunk-size 32 --left-context-frames 128 \
    --decoding-method greedy_search --eval-set dev
```

Report a **latency ladder**, not one number — chunk 16 / 32 / 64 at fixed left context — since
the whole point of the run is choosing a latency/accuracy operating point. Score through the
same `bg_text_norm` normalizer so the numbers stay comparable to every other row in
`BG_ASR_BENCHMARK.md`. Keep the frozen test decode-once discipline: pick the operating point on
**dev**, then decode test once.

## Then

`export-onnx-streaming.py` produces a streaming sherpa-onnx recognizer — the artifact that
enables live captioning, dictation and genuine on-device use. The playground's record-then-
transcribe UX does not need it and should keep using the offline model, which is more accurate.

## Decisions taken

- **Train from scratch, not warm-started from GP-v2.** Parameter shapes are identical, so
  initializing from the offline checkpoint is technically possible and would likely converge
  faster. Rejected for this run: it adds a second variable to a comparison whose only purpose
  is isolating the cost of causality, and the model would have to unlearn a right-context
  habit. Worth trying as GP-S2 if the from-scratch result disappoints.
- **Same 30-epoch schedule**, even though GP-v2's loss was still descending at epoch 30, for
  the same attributability reason. Extending the schedule is a separate experiment that should
  be run on the offline model first, where it is cheaper to interpret.
