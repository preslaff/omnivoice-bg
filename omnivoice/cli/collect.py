#!/usr/bin/env python3
# Copyright    2026  OmniVoice-BG  (Apache-2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Volunteer imperative-recording web app (OmniVoice-BG, Track B).

WHY THIS EXISTS
---------------
The ft7/ft8 experiments showed imperative stress is learned *voice-specifically*
— from a few clips per (verb, voice) the model memorises rather than abstracts,
so прави́/ходи́/купи́ render correctly only with the voice they trained in. The
root cause is too few speakers per imperative. This app collects the SAME marked
imperative sentences from MANY volunteers; the resulting speaker diversity is
what forces a voice-*independent* text→stress rule (see
docs/ACCENTOR_REFRESH_PLAN.md and the imperative-distill memory).

Each volunteer records short marked imperatives (прави́, ходи́, купи́ …) in their
own voice. Clips land in ``data/bg/imperatives_volunteer/`` and are later turned
into training tokens by ``convert_volunteer.py`` (24 kHz, QC, Higgs tokens).

Run (no GPU needed — this only collects audio):

    python -m omnivoice.cli.collect --host 0.0.0.0 --port 8003

Then point volunteers at http://<host>:8003/ . For wider-than-LAN reach put it
behind a tunnel/reverse proxy — there is NO authentication here, so keep it to a
trusted group or add auth at the proxy.

Endpoints:
    GET  /                       -> the recording page (self-contained HTML)
    GET  /health                 -> {"status","prompts","submissions", ...}
    GET  /prompt?speaker=<id>     -> next unrecorded prompt for that speaker
    POST /submit                 -> multipart (audio, speaker, prompt_id, ...);
                                    multiple takes per prompt allowed (…_t0,_t1…)
    POST /flag                   -> volunteer reports a wrong/awkward sentence;
                                    logged to flags.jsonl + skipped for that speaker
    GET  /stats                  -> per-verb / per-speaker coverage counts
"""

import argparse
import hashlib
import json
import logging
import pathlib
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

logger = logging.getLogger("omnivoice.collect")

ACUTE = "́"                      # combining acute (the stress mark)
VOWELS = set("аъоуеиюяАЪОУЕИЮЯ")
REPO = pathlib.Path(__file__).resolve().parents[2]

# Sections whose imperatives carry ENDING stress (-и́) — the resistant, high-value
# class the whole effort is about (прави́/ходи́ live in section 5, the CTA -и́ bulk
# in section 4). Prompts from these sections are surfaced FIRST to every speaker.
HIGH_VALUE_SECTIONS = {4, 5}


# ----------------------------------------------------------------------------
# Prompt bank — verbs × frames, mirroring teacher_distill.stage_expand but
# additionally tracking the section (for prioritising the resistant class).
# ----------------------------------------------------------------------------
def _strip_marks(s: str) -> str:
    return s.replace(ACUTE, "")


def _cap_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _read_verbs(path: pathlib.Path):
    """Yield (verb_marked, homograph, section) skipping comments/blanks. Tracks
    the current 'SECTION N' header so ending-stress classes can be prioritised."""
    section = 0
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"SECTION\s+(\d+)", ln)
        if m:
            section = int(m.group(1))
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        homograph = "# HOMOGRAPH" in ln
        text = ln.split("#", 1)[0].strip()
        if text:
            out.append((text, homograph, section))
    return out


def _read_frames(path: pathlib.Path):
    frames = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        f = ln.split("#", 1)[0].strip()
        if "{imp}" in f:
            frames.append(f)
    return frames


def build_bank(verbs_path: pathlib.Path, frames_path: pathlib.Path):
    """Full verb×frame combinatorial bank with per-prompt metadata."""
    verbs = _read_verbs(verbs_path)
    frames = _read_frames(frames_path)
    if not verbs or not frames:
        raise SystemExit(f"empty corpus: {len(verbs)} verbs, {len(frames)} frames")
    bank, seen = [], set()
    for verb_marked, homograph, section in verbs:
        for frame in frames:
            marked = _cap_first(frame.replace("{imp}", verb_marked))
            plain = _strip_marks(marked)
            if plain in seen:
                continue
            seen.add(plain)
            # Priority: ending-stress sections first, homographs bumped above
            # their section-mates — so even a volunteer who records only a
            # handful covers прави́/ходи́/купи́ and the other locked words.
            priority = (2 if section in HIGH_VALUE_SECTIONS else 0) + (1 if homograph else 0)
            bank.append({
                "id": f"imp_{len(bank):06d}",
                "verb": _strip_marks(verb_marked),
                "verb_marked": verb_marked,
                "homograph": homograph,
                "section": section,
                "priority": priority,
                "marked": marked,
                "plain": plain,
            })
    return bank


def speaker_order(bank, speaker: str):
    """Deterministic per-speaker prompt ordering: high-value class first, but
    verb/frame choice varied between speakers (seeded by the speaker id) so the
    same word is recorded in different carrier phrases across the crowd."""
    seed = int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16)
    def key(row):
        # stable pseudo-random tie-break per (speaker, prompt)
        h = int(hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()[:8], 16)
        return (-row["priority"], h)
    return sorted(bank, key=key)


# ----------------------------------------------------------------------------
# Front-end (self-contained; no external assets).
# ----------------------------------------------------------------------------
def _render_marked_html(marked: str) -> str:
    """Wrap the acute-marked vowel in a highlight span for legible display."""
    out, i = [], 0
    while i < len(marked):
        ch = marked[i]
        nxt = marked[i + 1] if i + 1 < len(marked) else ""
        if ch in VOWELS and nxt == ACUTE:
            out.append(f'<span class="stress">{ch}́</span>')
            i += 2
        else:
            out.append(ch if ch != "<" else "&lt;")
            i += 1
    return "".join(out)


INDEX_HTML = """<!doctype html>
<html lang="bg"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Дарете гласа си · OmniVoice-BG</title>
<link rel="icon" href="/favicon">
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 640px; margin: 0 auto; padding: 1.2rem; line-height: 1.5; }
  .brand { margin: .6rem auto .3rem; max-width: min(480px, 92%);
           background: #fff; border-radius: 18px; padding: 10px;
           box-shadow: 0 1px 6px #0002; }
  /* wide, short banner: crop the tall 16:9 art top/bottom, keep it centered */
  .brand img { display: block; width: 100%; height: 96px; object-fit: cover;
               object-position: center; border-radius: 10px; }
  h1 { font-size: 1.3rem; text-align: center; }
  .card { border: 1px solid #8883; border-radius: 14px; padding: 1.4rem 1.2rem;
          margin: 1rem 0; text-align: center; }
  .prompt { font-size: 2rem; margin: .6rem 0 1.2rem; }
  .stress { color: #d33; font-weight: 700; text-decoration: underline;
            text-underline-offset: 4px; }
  .hint { color: #888; font-size: .85rem; }
  .examples { border: 1px dashed #8884; border-radius: 12px; padding: .8rem 1rem;
              margin: 1rem 0; background: #8881; }
  .examples .ex-title { font-weight: 600; font-size: .95rem; margin-bottom: .4rem; }
  .examples ul { margin: .3rem 0; padding-left: 1.1rem; }
  .examples li { font-size: 1.15rem; margin: .3rem 0; }
  button { font-size: 1rem; padding: .6rem 1.1rem; margin: .25rem; border-radius: 10px;
           border: 1px solid #8886; background: #8881; cursor: pointer; }
  button.primary { background: #2a7; color: #fff; border-color: #2a7; }
  button.rec { background: #d33; color: #fff; border-color: #d33; }
  button.flag { background: transparent; border-color: #c93a; color: #c93;
                font-size: .85rem; padding: .35rem .8rem; }
  button:disabled { opacity: .4; cursor: default; }
  #consent input, #consent label { font-size: .95rem; }
  .row { display: flex; gap: .5rem; align-items: center; justify-content: center;
         flex-wrap: wrap; }
  input[type=text] { font-size: 1rem; padding: .45rem .6rem; border-radius: 8px;
                     border: 1px solid #8886; }
  .bar { height: 6px; background: #8882; border-radius: 3px; overflow: hidden; margin: .8rem 0; }
  .bar > div { height: 100%; background: #2a7; width: 0; transition: width .3s; }
  .msg { min-height: 1.4em; font-size: .9rem; }
  .ok { color: #2a7; } .err { color: #d33; }
  audio { width: 100%; margin-top: .6rem; }
</style></head><body>
<div class="brand"><img src="/logo" alt="OmniVoice-BG"
     onerror="this.style.display='none'"></div>
<h1>Дарете гласа си за българския синтез</h1>
<p class="hint">Прочетете изречението на глас, естествено. Ударението е
<span class="stress">оцветено</span> — наблегнете на тази сричка. Записите се
използват само за обучение на модел за реч.</p>

<div class="examples">
  <div class="ex-title">Примери — така ще изглеждат изреченията, които ще запишете:</div>
  <ul>
    <li>Прав<span class="stress">и́</span> упражнения всеки ден.</li>
    <li>Ход<span class="stress">и́</span> на разходка всяка сутрин.</li>
    <li>Куп<span class="stress">и́</span> сега на промоция.</li>
    <li>Обад<span class="stress">и́</span> се още днес.</li>
  </ul>
  <div class="hint">Прочетете всяко изречение спокойно и естествено, като наблегнете
  на <span class="stress">оцветената</span> сричка (тя носи ударението).</div>
</div>

<div id="consent" class="card">
  <div class="row">
    <label>Псевдоним: <input type="text" id="nick" placeholder="напр. mila_v" maxlength="24"></label>
  </div>
  <div class="row" style="margin:.6rem 0">
    <label>Глас:
      <select id="sex">
        <option value="">—</option>
        <option value="f">женски</option>
        <option value="m">мъжки</option>
        <option value="x">друг</option>
      </select>
    </label>
  </div>
  <label><input type="checkbox" id="agree"> Съгласен/на съм записите ми да се
  използват за обучение на модел (без лични данни освен псевдонима).</label>
  <div class="row" style="margin-top:.8rem">
    <button class="primary" id="start">Започни</button>
  </div>
</div>

<div id="rec" class="card" hidden>
  <div class="hint" id="progress">—</div>
  <div class="bar"><div id="progbar"></div></div>
  <div class="prompt" id="prompt">…</div>
  <div class="row" id="reprow">
    <button class="rec" id="record">● Запиши</button>
    <button id="stop" disabled>■ Спри</button>
  </div>
  <audio id="playback" controls hidden></audio>
  <div class="row" id="confirmrow" style="margin-top:.6rem">
    <button id="again" disabled>↺ Отново</button>
    <button class="primary" id="submit" disabled>✓ Изпрати</button>
  </div>
  <div class="row" id="postsubmit" style="margin-top:.6rem" hidden>
    <span class="hint" id="takeinfo"></span>
    <button id="againtake">↺ Запиши това пак</button>
    <button class="primary" id="next">Следващо ▶</button>
  </div>
  <div class="row" style="margin-top:.6rem">
    <button class="flag" id="flag">⚑ Изречението е грешно</button>
  </div>
  <div class="msg" id="msg"></div>
</div>

<script>
let speaker=null, sex="", current=null, mediaRec=null, chunks=[], blob=null, done=0;

function slug(s){ return s.trim().toLowerCase().replace(/[^a-z0-9а-я_]+/gi,"_").replace(/^_+|_+$/g,"").slice(0,24); }

document.getElementById("start").onclick = async () => {
  const nick = slug(document.getElementById("nick").value);
  if(!nick){ alert("Въведете псевдоним."); return; }
  if(!document.getElementById("agree").checked){ alert("Моля, отбележете съгласието."); return; }
  sex = document.getElementById("sex").value;
  speaker = nick;
  localStorage.setItem("ov_speaker", speaker);
  document.getElementById("consent").hidden = true;
  document.getElementById("rec").hidden = false;
  await nextPrompt();
};

async function nextPrompt(){
  const msg = document.getElementById("msg"); msg.textContent="";
  const r = await fetch(`/prompt?speaker=${encodeURIComponent(speaker)}`);
  const j = await r.json();
  if(j.done){
    document.getElementById("prompt").textContent = "🎉 Готово! Записахте всичко. Благодарим!";
    document.getElementById("record").disabled = true;
    return;
  }
  current = j;
  done = j.completed;
  document.getElementById("prompt").innerHTML = j.marked_html;
  document.getElementById("progress").textContent =
     `Записани от вас: ${j.completed} · остават ~${j.remaining}`;
  document.getElementById("progbar").style.width =
     Math.min(100, 100*j.completed/(j.completed+j.remaining)) + "%";
  resetControls();
}

function resetControls(){
  blob=null;
  document.getElementById("reprow").hidden=false;
  document.getElementById("confirmrow").hidden=false;
  document.getElementById("postsubmit").hidden=true;
  document.getElementById("record").disabled=false;
  document.getElementById("stop").disabled=true;
  document.getElementById("again").disabled=true;
  document.getElementById("submit").disabled=true;
  document.getElementById("flag").disabled=false;
  const pb=document.getElementById("playback"); pb.hidden=true; pb.removeAttribute("src");
}

document.getElementById("record").onclick = async () => {
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1}}); }
  catch(e){ document.getElementById("msg").innerHTML='<span class="err">Няма достъп до микрофона.</span>'; return; }
  chunks=[];
  mediaRec = new MediaRecorder(stream);
  mediaRec.ondataavailable = e => chunks.push(e.data);
  mediaRec.onstop = () => {
    blob = new Blob(chunks, {type: mediaRec.mimeType || "audio/webm"});
    const pb=document.getElementById("playback");
    pb.src = URL.createObjectURL(blob); pb.hidden=false;
    stream.getTracks().forEach(t=>t.stop());
    document.getElementById("again").disabled=false;
    document.getElementById("submit").disabled=false;
  };
  mediaRec.start();
  document.getElementById("record").disabled=true;
  document.getElementById("stop").disabled=false;
};

document.getElementById("stop").onclick = () => {
  if(mediaRec && mediaRec.state!=="inactive") mediaRec.stop();
  document.getElementById("stop").disabled=true;
};
document.getElementById("again").onclick = resetControls;

document.getElementById("submit").onclick = async () => {
  if(!blob) return;
  document.getElementById("submit").disabled=true;
  const msg=document.getElementById("msg"); msg.textContent="Изпращане…";
  const fd = new FormData();
  fd.append("audio", blob, "clip.webm");
  fd.append("speaker", speaker);
  fd.append("prompt_id", current.id);
  fd.append("sex", sex);
  fd.append("mime", blob.type||"audio/webm");
  try {
    const r = await fetch("/submit", {method:"POST", body: fd});
    const j = await r.json();
    if(j.ok){
      // success -> offer another take of THIS sentence, or move on
      const takes = (j.take||0)+1;
      document.getElementById("takeinfo").textContent =
         (takes>1 ? `Записани дубли: ${takes}. ` : "") + "✓ Записано.";
      document.getElementById("reprow").hidden=true;
      document.getElementById("confirmrow").hidden=true;
      document.getElementById("postsubmit").hidden=false;
      document.getElementById("flag").disabled=true;
      msg.innerHTML='<span class="ok">Благодарим! Може да запишете още един дубъл или да продължите.</span>';
    } else {
      msg.innerHTML='<span class="err">'+(j.reason||"отхвърлено")+' — опитайте пак.</span>';
      document.getElementById("submit").disabled=false;
    }
  } catch(e){ msg.innerHTML='<span class="err">Мрежова грешка.</span>';
              document.getElementById("submit").disabled=false; }
};

// record ANOTHER take of the same sentence
document.getElementById("againtake").onclick = () => {
  document.getElementById("msg").textContent="";
  resetControls();   // same `current`, re-arm recording
};
// move on to the next sentence
document.getElementById("next").onclick = async () => { await nextPrompt(); };

// flag the current sentence as wrong/awkward -> logged, skipped
document.getElementById("flag").onclick = async () => {
  if(!current) return;
  const reason = prompt("Какво не е наред с изречението? (по избор)") || "";
  const fd = new FormData();
  fd.append("speaker", speaker); fd.append("prompt_id", current.id); fd.append("reason", reason);
  try { await fetch("/flag", {method:"POST", body: fd});
        document.getElementById("msg").innerHTML='<span class="ok">⚑ Отбелязано, благодарим.</span>';
        await nextPrompt();
  } catch(e){ document.getElementById("msg").innerHTML='<span class="err">Мрежова грешка.</span>'; }
};
</script>
</body></html>"""


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def _image_mime(path: pathlib.Path) -> str:
    """Sniff the real image format from magic bytes (the file may be misnamed)."""
    head = path.read_bytes()[:4]
    if head[:2] == b"\xff\xd8":
        return "image/jpeg"
    if head[:4] == b"\x00\x00\x01\x00":
        return "image/x-icon"
    return "image/png"


def _ffprobe_duration(path: pathlib.Path) -> Optional[float]:
    """Clip duration in seconds via ffprobe, or None if unavailable/undecodable."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return None


def build_app(bank, out_dir: pathlib.Path, min_sec: float, max_sec: float,
              target_per_prompt: int, logo_path: Optional[pathlib.Path] = None,
              favicon_path: Optional[pathlib.Path] = None) -> FastAPI:
    app = FastAPI(title="OmniVoice-BG voice collector")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    submissions_path = out_dir / "submissions.jsonl"
    flags_path = out_dir / "flags.jsonl"
    by_id = {r["id"]: r for r in bank}
    lock = threading.Lock()

    # Warm state from any prior run so /prompt resumes and /stats is accurate.
    done_by_speaker: dict[str, set] = {}     # prompts a speaker has finished-or-skipped
    count_by_prompt: dict[str, int] = {}     # accepted takes per prompt (all speakers)
    takes_by_key: dict[tuple, int] = {}      # accepted takes per (speaker, prompt)
    if submissions_path.exists():
        for ln in submissions_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                s = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not s.get("accepted", True):
                continue
            done_by_speaker.setdefault(s["speaker"], set()).add(s["prompt_id"])
            count_by_prompt[s["prompt_id"]] = count_by_prompt.get(s["prompt_id"], 0) + 1
            key = (s["speaker"], s["prompt_id"])
            takes_by_key[key] = takes_by_key.get(key, 0) + 1
    # Flagged prompts are skipped for the speaker who flagged them.
    if flags_path.exists():
        for ln in flags_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                fl = json.loads(ln)
            except json.JSONDecodeError:
                continue
            done_by_speaker.setdefault(fl["speaker"], set()).add(fl["prompt_id"])

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "prompts": len(bank),
            "speakers": len(done_by_speaker),
            "submissions": sum(len(v) for v in done_by_speaker.values()),
            "out_dir": str(out_dir),
        }

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/logo")
    def logo():
        if logo_path is None or not logo_path.exists():
            raise HTTPException(status_code=404, detail="no logo configured")
        return FileResponse(str(logo_path), media_type=_image_mime(logo_path))

    @app.get("/favicon")
    @app.get("/favicon.ico")
    def favicon():
        # Prefer a dedicated square favicon; fall back to the header logo.
        path = favicon_path if (favicon_path and favicon_path.exists()) else logo_path
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="no favicon configured")
        return FileResponse(str(path), media_type=_image_mime(path))

    @app.get("/prompt")
    def prompt(speaker: str):
        speaker = speaker.strip().lower()
        if not speaker:
            raise HTTPException(status_code=400, detail="speaker required")
        with lock:
            done = done_by_speaker.get(speaker, set())
            order = speaker_order(bank, speaker)
            nxt = next((r for r in order if r["id"] not in done), None)
        if nxt is None:
            return {"done": True, "completed": len(done)}
        return {
            "done": False,
            "id": nxt["id"],
            "verb": nxt["verb"],
            "marked": nxt["marked"],
            "marked_html": _render_marked_html(nxt["marked"]),
            "completed": len(done_by_speaker.get(speaker, set())),
            "remaining": len(bank) - len(done_by_speaker.get(speaker, set())),
        }

    @app.get("/stats")
    def stats():
        # verbs still under the target coverage — where more volunteers help most
        per_verb: dict[str, int] = {}
        for pid, n in count_by_prompt.items():
            v = by_id[pid]["verb"] if pid in by_id else pid
            per_verb[v] = per_verb.get(v, 0) + n
        under = sorted((v for v, n in per_verb.items() if n < target_per_prompt))
        return {
            "speakers": len(done_by_speaker),
            "accepted_total": sum(count_by_prompt.values()),
            "verbs_covered": len(per_verb),
            "verbs_under_target": len(under),
            "target_per_prompt": target_per_prompt,
        }

    @app.post("/submit")
    async def submit(audio: UploadFile, speaker: str = Form(...),
                     prompt_id: str = Form(...), sex: str = Form(""),
                     mime: str = Form("audio/webm")):
        speaker = speaker.strip().lower()
        row = by_id.get(prompt_id)
        if row is None:
            raise HTTPException(status_code=400, detail="unknown prompt_id")
        data = await audio.read()
        if len(data) < 1024:                       # empty / failed recording
            return JSONResponse({"ok": False, "reason": "празен запис"})

        # Multiple takes per (speaker, prompt) are allowed — the volunteer may
        # re-record the same sentence as many times as they like. Each take gets
        # its own file (…_t0, _t1, …) so nothing is overwritten.
        ext = ".webm" if "webm" in mime else (".ogg" if "ogg" in mime else ".bin")
        spk_dir = raw_dir / speaker
        spk_dir.mkdir(parents=True, exist_ok=True)
        with lock:
            take = takes_by_key.get((speaker, prompt_id), 0)
        clip_path = spk_dir / f"{prompt_id}_t{take}{ext}"
        clip_path.write_bytes(data)

        dur = _ffprobe_duration(clip_path)
        accepted, reason = True, ""
        if dur is not None:
            if dur < min_sec:
                accepted, reason = False, "твърде кратко"
            elif dur > max_sec:
                accepted, reason = False, "твърде дълго"
        if not accepted:
            clip_path.unlink(missing_ok=True)      # don't keep rejected audio

        rec = {
            "speaker": speaker, "sex": sex, "prompt_id": prompt_id, "take": take,
            "verb": row["verb"], "verb_marked": row["verb_marked"],
            "marked": row["marked"], "plain": row["plain"],
            "homograph": row["homograph"], "section": row["section"],
            "audio_path": str(clip_path) if accepted else None,
            "mime": mime, "bytes": len(data),
            "duration": round(dur, 3) if dur is not None else None,
            "accepted": accepted, "reason": reason,
            "ts": int(time.time()),
        }
        with lock:
            with open(submissions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if accepted:
                done_by_speaker.setdefault(speaker, set()).add(prompt_id)
                count_by_prompt[prompt_id] = count_by_prompt.get(prompt_id, 0) + 1
                takes_by_key[(speaker, prompt_id)] = take + 1
        return {"ok": accepted, "reason": reason, "take": take,
                "duration": rec["duration"], "completed": len(done_by_speaker.get(speaker, set()))}

    @app.post("/flag")
    async def flag(speaker: str = Form(...), prompt_id: str = Form(...),
                   reason: str = Form("")):
        """Volunteer reports the current sentence as wrong/awkward. Logged for the
        native gate and skipped for this speaker."""
        speaker = speaker.strip().lower()
        row = by_id.get(prompt_id)
        if row is None:
            raise HTTPException(status_code=400, detail="unknown prompt_id")
        rec = {"speaker": speaker, "prompt_id": prompt_id, "verb": row["verb"],
               "marked": row["marked"], "plain": row["plain"],
               "reason": reason.strip()[:200], "ts": int(time.time())}
        with lock:
            with open(flags_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done_by_speaker.setdefault(speaker, set()).add(prompt_id)  # skip it
        return {"ok": True}

    return app


def main():
    p = argparse.ArgumentParser(description="OmniVoice-BG volunteer voice collector")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 to accept LAN clients")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--verbs", default=str(REPO / "config/bg/imperative_verbs.txt"))
    p.add_argument("--frames", default=str(REPO / "config/bg/imperative_frames_clean.txt"),
                   help="CURATED universal frames (default) so volunteers never hit "
                        "an awkward verb×frame combo; pass imperative_frames.txt for "
                        "the full combinatorial set")
    p.add_argument("--out-dir", default=str(REPO / "data/bg/imperatives_volunteer"))
    p.add_argument("--min-sec", type=float, default=0.4, help="reject clips shorter than this")
    p.add_argument("--max-sec", type=float, default=15.0, help="reject clips longer than this")
    p.add_argument("--target-per-prompt", type=int, default=20,
                   help="coverage target per prompt (for /stats reporting only)")
    p.add_argument("--logo", default=str(REPO / "deploy/logo.png"),
                   help="image shown in the page header ('' to disable)")
    p.add_argument("--favicon", default=str(REPO / "deploy/favicon.png"),
                   help="SQUARE image for the browser-tab favicon; falls back to "
                        "--logo if absent ('' to disable)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    bank = build_bank(pathlib.Path(args.verbs), pathlib.Path(args.frames))
    n_hi = sum(1 for r in bank if r["priority"] > 0)
    logger.info("prompt bank: %d sentences (%d high-value ending-stress) from %d verbs",
                len(bank), n_hi, len({r["verb"] for r in bank}))
    logo_path = pathlib.Path(args.logo) if args.logo else None
    if logo_path and not logo_path.exists():
        logger.warning("logo %s not found — header will be blank", logo_path)
    favicon_path = pathlib.Path(args.favicon) if args.favicon else None
    if favicon_path and not favicon_path.exists():
        logger.warning("favicon %s not found — falling back to the header logo "
                       "(wide art makes a poor tab icon; drop a square PNG there)",
                       favicon_path)
    app = build_app(bank, pathlib.Path(args.out_dir), args.min_sec, args.max_sec,
                    args.target_per_prompt, logo_path, favicon_path)
    logger.info("Collecting to %s  (NO auth — trusted network / add a proxy).", args.out_dir)
    logger.info("Serving on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
