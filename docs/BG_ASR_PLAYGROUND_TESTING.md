# ASR playground — local manual test guide

How to run `omnivoice/cli/asr_playground.py` on this machine, record with a real microphone,
and check that what lands on disk is what we want. Everything here happens on the dev box;
nothing touches the VPS.

Companion docs: [BG_ASR_BENCHMARK.md](BG_ASR_BENCHMARK.md) (what the model scores),
[BG_ASR_ROADMAP.md](BG_ASR_ROADMAP.md) (why this exists).

---

## 1. Start the server

```bash
cd ~/WORK/OmniVoice
mkdir -p data/bg/asr_playground_localtest

OMNIVOICE_ASR_SECRET=localtest-secret \
./venv-asr/bin/python omnivoice/cli/asr_playground.py \
    --host 127.0.0.1 --port 8055 \
    --out-dir data/bg/asr_playground_localtest
```

Runs in the foreground so you can watch the log; `Ctrl-C` stops it. Use **`./venv-asr/bin/python`**,
not `./venv/bin/python` — the training venv has torch and a conflicting onnxruntime.

A healthy start looks like this (~1 s):

```
INFO model files verified against MODEL_CARD.json
INFO recognizer ready in 0.4s (gp-v2-e30-avg5-int8, 1 thread(s)) — rss 190 MB
INFO ASR playground on 127.0.0.1:8055 — 89 prompts, cap 15s, out=data/bg/asr_playground_localtest
INFO:     Uvicorn running on http://127.0.0.1:8055
```

If it exits instead, the message says why — a missing model file, a sha256 mismatch against
`MODEL_CARD.json` (a half-finished copy), or missing `ffmpeg`/`ffprobe`. It is meant to fail at
startup rather than in front of a volunteer.

In a second terminal:

```bash
curl -s localhost:8055/asr/health | python3 -m json.tool
```

## 2. Open the page

**http://localhost:8055/asr/**

Use `localhost` (or `127.0.0.1`) — **not** the machine's LAN IP. Browsers only allow microphone
access in a "secure context", which means HTTPS *or* localhost. From `http://192.168.x.x:8055`
the record button will fail with a permission error. (Phone testing therefore needs the VPS
over HTTPS, or an SSH tunnel to localhost.)

Chrome and Firefox both work. The browser will ask for microphone permission the first time you
press record — allow it.

## 3. The flow

**Consent gate.** Tick both boxes, type a nickname (e.g. `test1`), press *Продължи*. The
nickname is remembered in `localStorage`, so a reload skips straight back to the app. Use
**different nicknames** across runs if you want to see different eval/train split assignments.

**Two modes**, switched by the buttons at the top. Both show the model's output *first*, then
ask what you actually said — typing is what kills completion, especially on a phone.

| | Прочети фраза (read) | Говори свободно (free) |
|---|---|---|
| You do | Read the yellow phrase aloud | Say anything |
| Step 2 shows | The model's output **diffed against the prompt** | The model's output, plain |
| Step 3 box | Prefilled with **the prompt** | Empty, with a green **copy** button |
| Typing needed | **None** — just press Изпрати | None if the model got it right |
| Reference is | The prompt | Your text (a post-edit of the model's) |

**Record.** Press *● Запиши*, speak, press *■ Спри*. The counter hard-stops at 15 s on its own.
Transcription starts automatically; the result appears as `Готово · NN мс`, and the model's
text appears immediately below it — red = substitution, orange = the model dropped a word,
green = the model invented one.

**Answer and submit.** In read mode the box already holds the prompt: press *Изпрати* unless
you misread. In free mode, if the model got you exactly right press the green
*✓ Моделът разпозна вярно — копирай текста*; otherwise copy and fix just the wrong words, or
type from scratch. Then *Изпрати*.

**Why the asymmetry matters.** Read mode's reference is the prompt, which existed *before* the
model ran — so no ordering can bias it, and it is the least work of all. That is where the
frozen eval set comes from. Free-mode text is a post-edit of the model's output and is training
data. The server records which of `prompt / edited_prompt / typed / copied / edited_copy`
produced each row and sets `eval_eligible` accordingly — see §5.

**Next.** *Следващ запис* clears everything and (in read mode) fetches a new prompt.

## 4. Things worth deliberately trying

| Try this | Expect |
|---|---|
| Speak quietly / fast / with background noise | Worse transcripts — this is the point, it's what we can't measure otherwise |
| Numbers, dates, prices, names | The prompt bank leans this way because it is where BG ASR fails |
| Talk for the full 15 s | Recorder hard-stops itself; no server error |
| Say nothing at all | A short or empty transcript; submit still works |
| Press *Изпрати* twice quickly | Second one is fine; each submit is its own row |
| Stop the server (`Ctrl-C`), then record and submit | Recording still works, the page says the model didn't answer, **and the submission still saves** — the (audio, transcript) pair is the artifact; the hypothesis is decoration |

## 5. Check what landed

```bash
cd ~/WORK/OmniVoice/data/bg/asr_playground_localtest
ls -R                                   # raw/<speaker_hash>/*.webm, staging/, *.jsonl
python3 -m json.tool < <(tail -1 submissions.jsonl)
```

Fields to look at:

- `mode` — `read` or `free`.
- `human_text` vs `asr_hyp` — your text vs the model's. **Both are always kept**; `asr_hyp` is
  never overwritten by your edit.
- `human_text_source` — **the field that decides what may enter a benchmark**, recomputed
  server-side and never trusted from the form:

  | value | meaning | eval? |
  |---|---|---|
  | `prompt` | read mode, prompt left as-is | ✅ |
  | `typed` | free mode, typed without copying | ✅ |
  | `edited_prompt` | read mode, prompt corrected after seeing the model | ✗ |
  | `copied` | text is verbatim the model's output | ✗ |
  | `edited_copy` | copied, then corrected | ✗ |

- `eval_eligible` — derived from the above at write time. A `copied` row's "reference" *is*
  the model's own output, so scoring it gives 0 % WER by construction; letting those into a
  benchmark would make the number fiction.
- `matches_hyp` — text equals the hypothesis exactly (how `copied` is detected regardless of
  what the client claimed).
- `edited` — read mode only: true if you changed the prompt text.
- `ui_flow` — the UI regime that produced the row (`hyp-first-v2`), so data collected under
  different flows is never silently pooled.
- `paste_events`, `keystrokes`, `answer_ms` — effort signals.
- `split` — `eval` or `train`, assigned from a hash of the speaker (never the clip), so one
  person can never land in both.
- `consent_sha256` — the exact consent wording you agreed to.
- `audio_sha256`, `duration`, `bytes` — the clip.

Quick tally across a session:

```bash
python3 - <<'PY'
import json, collections, pathlib
rows = [json.loads(l) for l in
        pathlib.Path("submissions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"{len(rows)} rows, {sum(r['duration'] for r in rows)/60:.1f} min audio")
print("by mode:  ", collections.Counter(r["mode"] for r in rows))
print("by source:", collections.Counter(r["human_text_source"] for r in rows))
print("by split: ", collections.Counter(r["split"] for r in rows))
print("eval-eligible:", sum(r["eval_eligible"] for r in rows))
PY
```

Rough WER of your own session, through the project's scoring normalizer:

```bash
cd ~/WORK/OmniVoice
./venv-asr/bin/python - <<'PY'
import json, pathlib, sys
sys.path.insert(0, ".")
from bg_text_norm import normalize
import jiwer
rows = [json.loads(l) for l in pathlib.Path(
    "data/bg/asr_playground_localtest/submissions.jsonl"
    ).read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r["asr_ok"] and r["asr_hyp"] and r["eval_eligible"]]
refs = [normalize(r["human_text"]) for r in rows]
hyps = [normalize(r["asr_hyp"]) for r in rows]
m = jiwer.process_words(refs, hyps)
print(f"{len(rows)} clips  WER {m.wer*100:.1f}%  "
      f"[{m.insertions} ins, {m.deletions} del, {m.substitutions} sub]")
PY
```

Expect **much worse than the 4.12 % headline** — that number is in-domain parliament audio.
A close-talk mic in a quiet room might land in the single digits; a laptop mic across the room,
or fast casual speech, can easily be 20–40 %. That gap is exactly what the playground exists to
measure.

## 6. Deletion round-trip

The token shown under *Изтриване на моите записи* after your first submit:

```bash
curl -s -X POST localhost:8055/asr/delete \
     -F "speaker=test1" -F "token=<the token>"
# -> {"ok":true,"deleted_files":N}
```

Audio files are unlinked immediately; the JSONL rows stay (append-only) with a tombstone in
`deletions.jsonl`. A wrong token returns 403.

## 7. Watch resource use

`/asr/health` reports `peak_rss_mb`, `last_ms`, `rtf`, `inflight`/`queued`:

```bash
watch -n2 'curl -s localhost:8055/asr/health | python3 -m json.tool'
```

Reference numbers measured here: 190 MB after startup, 303 MB peak under 12 concurrent 14 s
clips, 201 ms for a 14 s clip on one thread (RTF 0.014). The VPS core is much slower and has no
VNNI, so expect RTF an order of magnitude worse there — that is what the Phase 1 soak measures,
not this box.

## 8. Reset between runs

```bash
rm -rf ~/WORK/OmniVoice/data/bg/asr_playground_localtest
```

`data/` is gitignored, so nothing recorded here can be committed by accident.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Record button errors immediately | Not a secure context — use `localhost`, not the LAN IP |
| `ModuleNotFoundError: fastapi` | Wrong interpreter; use `./venv-asr/bin/python` |
| Exits with a sha256 mismatch | Model files in `models/asr/gp-v2/` don't match `MODEL_CARD.json` |
| `Записът не може да бъде прочетен` | ffmpeg couldn't decode the browser's container — check `ffmpeg -version` |
| 429 `Системата е заета` | More than 3 concurrent transcribes; by design |
| Transcript is empty | Silence, or the mic captured nothing — check the playback control |
