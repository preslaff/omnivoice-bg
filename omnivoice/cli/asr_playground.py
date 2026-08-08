#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""OmniVoice-BG ASR playground — volunteers find where the recognizer breaks.

Companion to collect.py (the imperative voice collector), deliberately a SEPARATE process:
own port, own venv, own systemd unit, own data directory. On the shared 2 GB VPS that is the
only way to give ASR a memory ceiling without putting the working collector inside it —
cgroup limits are per-unit. See docs/BG_ASR_PLAYGROUND.md.

WHAT IT COLLECTS. Every submission is (audio, human transcript). That pair is the point:
GP-v2 scores 4.12% on the frozen parliament holdout and that is its ONLY honest number,
because the FLEURS/CV train splits went into training and consumed the project's only
out-of-domain eval sets. This rebuilds one from real microphones.

TWO MODES, BOTH ANCHORING-FREE:
  read  — a phrase from config/bg/asr_prompts.txt is shown, the volunteer reads it, and the
          PROMPT is the reference. No typing. The reference existed before the model ran.
  free  — the volunteer speaks freely, then types what they said into an EMPTY box. Only
          after that is the hypothesis revealed, with a word-level diff.
Never prefill the free-mode textarea with the hypothesis. Post-edited text drifts toward the
model — accepting plausible-but-wrong words, missing omissions — which is fine for training
data and fatal for a benchmark. `typed_before_reveal` records which rows are trustworthy.

MEMORY. Attention in the encoder's first stack is O(T^2) at 50 Hz, so a clip's cost grows
quadratically: 15 s ~ 27 MB, 60 s ~ 430 MB. The 15 s cap is a MEMORY control, not a UX
preference, and it is also the only timeout available — a sherpa-onnx decode is a blocking
C++ call that cannot be interrupted from Python, so bounding the input bounds the stall.

DEPENDENCIES: fastapi, uvicorn, python-multipart, sherpa-onnx. No numpy (audio is fed as
array.array), no torch, no lhotse.

  IMPORTANT — launch as a SCRIPT BY PATH:  python omnivoice/cli/asr_playground.py
  NOT `python -m omnivoice.cli.asr_playground`, which would import omnivoice/__init__.py and
  pull in torch. Same rule as collect.py; see deploy/omnivoice-asr.service.
"""
import argparse
import array
import hashlib
import hmac
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

logger = logging.getLogger("asr_playground")

REPO = pathlib.Path(__file__).resolve().parents[2]     # /opt/omnivoice on the VPS

# Split assignment is keyed on the SPEAKER, never the clip, so one person can never straddle
# train and eval. This salt is a committed constant and must never be rotated — rotating it
# silently reshuffles every future speaker against an already-frozen eval set.
SPLIT_SALT = "omnivoice-bg-asrp-v1"
EVAL_BUCKET_PCT = 25                                    # < 25 of 100 buckets -> eval

MAX_TEXT_CHARS = 2000
SPEAKER_RE = re.compile(r"[^a-z0-9а-я_]+")


def slug_speaker(s: str) -> str:
    s = SPEAKER_RE.sub("_", (s or "").strip().lower()).strip("_")
    return s[:24]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_mb() -> float:
    """Current and peak RSS from /proc — the production memory signal."""
    try:
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return -1.0


def ffprobe_duration(path: pathlib.Path) -> Optional[float]:
    """Clip duration in seconds, or None if undecodable. Same helper as collect.py."""
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


def decode_to_16k(path: pathlib.Path) -> array.array:
    """Any browser container -> 16 kHz mono float32, as array.array (no numpy on the VPS)."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace")[:200])
    a = array.array("f")
    a.frombytes(out.stdout)
    return a


def read_prompts(path: pathlib.Path):
    """[{id, text}] from the prompt bank; blank lines and #-comments ignored."""
    bank = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s and not s.startswith("#"):
            bank.append({"id": f"p{n:04d}", "text": s})
    return bank


def word_diff(ref: str, hyp: str):
    """Word-level ops for the UI: [(tag, ref_word, hyp_word)], tag in eq/sub/del/ins.

    Plain Levenshtein over words. The volunteer sees WHERE the model went wrong, which is the
    whole reward loop; scoring proper is done offline through bg_text_norm + jiwer, never here.
    """
    r, h = ref.split(), hyp.split()
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1].lower() != h[j - 1].lower()))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1].lower() != h[j - 1].lower()):
            ops.append(("eq" if r[i - 1].lower() == h[j - 1].lower() else "sub",
                        r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", r[i - 1], "")); i -= 1
        else:
            ops.append(("ins", "", h[j - 1])); j -= 1
    return list(reversed(ops))


INDEX_HTML = """<!doctype html>
<html lang="bg"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniVoice-BG · Къде греши разпознаването?</title>
<link rel="icon" href="/asr/favicon">
<style>
 :root { --ink:#1a202c; --mut:#5a6675; --line:#e2e8f0; --bg:#f7fafc; --accent:#2b6cb0;
         --ok:#276749; --bad:#c53030; --warn:#b7791f; }
 * { box-sizing:border-box; }
 body { margin:0; font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
        color:var(--ink); background:var(--bg); }
 .wrap { max-width:720px; margin:0 auto; padding:0 16px 48px; }
 .brand img { display:block; width:100%; height:88px; object-fit:cover; border-radius:0 0 10px 10px; }
 h1 { font-size:22px; margin:18px 0 6px; }
 p.lede { color:var(--mut); margin:0 0 18px; }
 .card { background:#fff; border:1px solid var(--line); border-radius:10px;
         padding:16px; margin:0 0 16px; }
 label { display:block; font-weight:600; margin:0 0 6px; }
 input[type=text], textarea { width:100%; padding:10px; font:inherit;
        border:1px solid var(--line); border-radius:8px; background:#fff; }
 textarea { min-height:96px; resize:vertical; }
 button { font:inherit; padding:11px 18px; border-radius:8px; border:1px solid transparent;
          background:var(--accent); color:#fff; cursor:pointer; }
 button.ghost { background:#fff; color:var(--accent); border-color:var(--accent); }
 button:disabled { opacity:.45; cursor:not-allowed; }
 .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
 .modes { display:flex; gap:8px; margin:0 0 14px; }
 .modes button { flex:1; background:#fff; color:var(--ink); border-color:var(--line); }
 .modes button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
 .prompt { font-size:20px; line-height:1.5; padding:14px; background:#fffbea;
           border:1px solid #f6e05e; border-radius:8px; margin:0 0 14px; }
 .muted { color:var(--mut); font-size:14px; }
 .timer { font-variant-numeric:tabular-nums; font-weight:700; }
 .hyp { padding:12px; background:#f7fafc; border:1px solid var(--line); border-radius:8px;
        font-size:17px; line-height:1.7; }
 .eq  { }
 .sub { background:#fed7d7; color:var(--bad); border-radius:3px; padding:0 2px; }
 .del { background:#feebc8; color:var(--warn); border-radius:3px; padding:0 2px;
        text-decoration:line-through; }
 .ins { background:#c6f6d5; color:var(--ok); border-radius:3px; padding:0 2px; }
 .legend span { margin-right:12px; font-size:13px; }
 .hide { display:none; }
 .err { color:var(--bad); }
 .ok { color:var(--ok); }
 footer { color:var(--mut); font-size:13px; text-align:center; padding:24px 0 0; }
 footer a { color:var(--mut); }
</style></head><body>
<div class="brand"><img src="/asr/logo" alt="OmniVoice-BG"></div>
<div class="wrap">

<h1>Помогнете ни да намерим къде греши</h1>
<p class="lede">Това е български модел за разпознаване на реч, обучен от нулата.
Работи добре на записи от парламента, но <b>ние не знаем колко добре се справя с истински
микрофони и обикновена реч</b> — точно това търсим. Всяка грешка, която откриете, е принос.</p>

<div id="consent" class="card">
  <label>Съгласие</label>
  <div id="consentText" class="muted" style="max-height:180px;overflow:auto;margin-bottom:10px"></div>
  <label class="row" style="font-weight:400">
    <input type="checkbox" id="c1"> Съгласен/на съм записът ми да бъде съхранен.</label>
  <label class="row" style="font-weight:400">
    <input type="checkbox" id="c2"> Съгласен/на съм да се използва за оценка и обучение на
    модели за българска реч.</label>
  <div class="row" style="margin-top:12px">
    <input type="text" id="speaker" placeholder="Псевдоним (напр. митко)" style="flex:1">
    <button id="start">Продължи</button>
  </div>
  <div id="cErr" class="err muted"></div>
</div>

<div id="app" class="hide">
  <div class="modes">
    <button id="mRead" class="on">Прочети фраза</button>
    <button id="mFree">Говори свободно</button>
  </div>

  <div class="card">
    <div id="promptBox">
      <label>Прочетете на глас:</label>
      <div class="prompt" id="promptText">…</div>
    </div>
    <div id="freeBox" class="hide">
      <label>Кажете каквото искате</label>
      <p class="muted">Няколко изречения на естествен български — как говорите с приятели,
      какво сте правили днес, каквото се сети. Максимум 15 секунди.</p>
    </div>
    <div class="row">
      <button id="rec">● Запиши</button>
      <span class="timer" id="timer">0.0 с / 15 с</span>
      <span class="muted" id="recMsg"></span>
    </div>
    <audio id="play" controls class="hide" style="width:100%;margin-top:12px"></audio>
  </div>

  <div id="typeCard" class="card hide">
    <label id="typeLabel">Какво казахте? Напишете го точно.</label>
    <p class="muted" id="typeHint">Пишете по памет, преди да видите какво е чул моделът —
    така сравнението е честно.</p>
    <textarea id="human" placeholder="Напишете тук…"></textarea>
    <div class="row" style="margin-top:12px">
      <button id="reveal" class="ghost">Покажи какво чу моделът</button>
      <button id="send">Изпрати</button>
      <span class="muted" id="sendMsg"></span>
    </div>
  </div>

  <div id="resCard" class="card hide">
    <label>Моделът чу:</label>
    <div class="hyp" id="hypBox"></div>
    <p class="legend" style="margin:10px 0 0">
      <span class="sub">замяна</span><span class="del">пропуснато</span>
      <span class="ins">добавено</span>
      <span class="muted" id="stat"></span></p>
  </div>

  <div class="card">
    <div class="row"><button id="next" class="ghost">Следващ запис</button>
      <span class="muted" id="counter"></span></div>
  </div>

  <details class="card"><summary class="muted">Изтриване на моите записи</summary>
    <p class="muted">Запазете кода по-долу — с него можете да поискате изтриване.</p>
    <div class="muted" id="delTok">—</div>
  </details>
</div>

<footer>OmniVoice-BG · <a href="/">записване на изречения</a> ·
  моделът работи офлайн на този сървър</footer>
</div>
<script>
const $ = s => document.querySelector(s);
let mode = "read", prompt_id = null, promptText = "", clip_id = null, blob = null;
let mediaRec = null, chunks = [], t0 = 0, timerH = null, consent = null;
let revealed = false, revealedAt = 0, typedFirst = true, pastes = 0, keys = 0, tStart = 0;
let hyp = null, asrOk = false, asrErr = null, sent = 0;
const MAXSEC = 15;

fetch("/asr/consent").then(r => r.json()).then(c => {
  consent = c; $("#consentText").textContent = c.text;
});
const saved = localStorage.getItem("ov_speaker");
if (saved) $("#speaker").value = saved;

$("#start").onclick = () => {
  const sp = $("#speaker").value.trim();
  if (!$("#c1").checked || !$("#c2").checked) { $("#cErr").textContent =
      "Моля отбележете и двете съгласия."; return; }
  if (sp.length < 2) { $("#cErr").textContent = "Моля въведете псевдоним."; return; }
  localStorage.setItem("ov_speaker", sp);
  $("#consent").classList.add("hide"); $("#app").classList.remove("hide");
  nextPrompt();
};
$("#mRead").onclick = () => setMode("read");
$("#mFree").onclick = () => setMode("free");
function setMode(m) {
  mode = m;
  $("#mRead").classList.toggle("on", m === "read");
  $("#mFree").classList.toggle("on", m === "free");
  $("#promptBox").classList.toggle("hide", m !== "read");
  $("#freeBox").classList.toggle("hide", m === "read");
  resetClip();
  if (m === "read") nextPrompt();
}
function nextPrompt() {
  fetch("/asr/prompt?speaker=" + encodeURIComponent($("#speaker").value.trim()))
    .then(r => r.json()).then(p => {
      prompt_id = p.id; promptText = p.text; $("#promptText").textContent = p.text;
    });
}
function resetClip() {
  clip_id = null; blob = null; hyp = null; asrOk = false; asrErr = null;
  revealed = false; revealedAt = 0; typedFirst = true; pastes = 0; keys = 0;
  $("#human").value = ""; $("#play").classList.add("hide");
  $("#typeCard").classList.add("hide"); $("#resCard").classList.add("hide");
  $("#recMsg").textContent = ""; $("#sendMsg").textContent = "";
  $("#timer").textContent = "0.0 с / " + MAXSEC + " с";
}
$("#next").onclick = () => { resetClip(); if (mode === "read") nextPrompt(); };

$("#rec").onclick = async () => {
  if (mediaRec && mediaRec.state === "recording") { mediaRec.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({audio:true}); }
  catch (e) { $("#recMsg").innerHTML = "<span class='err'>Няма достъп до микрофона.</span>";
              return; }
  chunks = []; mediaRec = new MediaRecorder(stream);
  mediaRec.ondataavailable = e => chunks.push(e.data);
  mediaRec.onstop = () => {
    clearInterval(timerH); stream.getTracks().forEach(t => t.stop());
    $("#rec").textContent = "● Запиши";
    blob = new Blob(chunks, {type: mediaRec.mimeType});
    const url = URL.createObjectURL(blob);
    $("#play").src = url; $("#play").classList.remove("hide");
    transcribe();
  };
  mediaRec.start(); t0 = Date.now(); $("#rec").textContent = "■ Спри";
  $("#recMsg").textContent = "";
  timerH = setInterval(() => {
    const el = (Date.now() - t0) / 1000;
    $("#timer").textContent = el.toFixed(1) + " с / " + MAXSEC + " с";
    if (el >= MAXSEC) mediaRec.stop();          // hard stop AT the cap, never a server 400
  }, 100);
};

function transcribe() {
  $("#recMsg").textContent = "Разпознаване…";
  const fd = new FormData();
  fd.append("audio", blob, "clip.webm");
  fd.append("mime", blob.type || "");
  fetch("/asr/transcribe", {method:"POST", body:fd})
    .then(async r => {
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || ("HTTP " + r.status));
      return j;
    })
    .then(j => {
      clip_id = j.clip_id; hyp = j.text; asrOk = true;
      $("#recMsg").innerHTML = "<span class='ok'>Готово · " + j.ms + " мс</span>";
      afterClip();
    })
    .catch(e => {
      // The recognizer failing must NOT cost us the recording — the (audio, transcript)
      // pair is the irreplaceable artifact and /asr/submit does not need the model.
      asrOk = false; asrErr = String(e.message || e);
      $("#recMsg").innerHTML = "<span class='err'>Моделът не отговори (" + asrErr +
        "). Записът пак може да бъде изпратен.</span>";
      afterClip();
    });
}
function afterClip() {
  $("#typeCard").classList.remove("hide");
  tStart = Date.now();
  if (mode === "read") {
    $("#typeLabel").textContent = "Прочетохте ли фразата точно?";
    $("#typeHint").textContent = "Ако сте се объркали или сте казали нещо друго, поправете " +
      "текста тук. Иначе не пипайте нищо.";
    $("#human").value = promptText;         // the PROMPT, never the model's guess
  } else {
    $("#typeLabel").textContent = "Какво казахте? Напишете го точно.";
    $("#typeHint").textContent = "Пишете по памет, преди да видите какво е чул моделът — " +
      "така сравнението е честно.";
    $("#human").value = "";                 // never prefilled with the hypothesis
  }
}
$("#human").addEventListener("paste", () => { pastes++; });
$("#human").addEventListener("keydown", () => { keys++; });

$("#reveal").onclick = () => {
  if (!hyp) { $("#sendMsg").textContent = "Няма разпознат текст."; return; }
  if (!revealed) {
    revealed = true; revealedAt = Date.now() - tStart;
    typedFirst = ($("#human").value.trim().length > 0) || mode === "read";
  }
  showDiff();
};
function showDiff() {
  const ref = $("#human").value.trim();
  fetch("/asr/diff", {method:"POST", headers:{"Content-Type":"application/json"},
                      body: JSON.stringify({ref: ref, hyp: hyp})})
    .then(r => r.json()).then(d => {
      $("#hypBox").innerHTML = d.html;
      $("#stat").textContent = " · " + d.errors + " разлики от " + d.ref_words + " думи";
      $("#resCard").classList.remove("hide");
    });
}
$("#send").onclick = () => {
  const human = $("#human").value.trim();
  if (human.length < 2) { $("#sendMsg").textContent = "Моля напишете текста."; return; }
  $("#send").disabled = true; $("#sendMsg").textContent = "Изпращане…";
  const fd = new FormData();
  fd.append("speaker", $("#speaker").value.trim());
  fd.append("mode", mode);
  fd.append("prompt_id", mode === "read" ? (prompt_id || "") : "");
  fd.append("prompt_text", mode === "read" ? promptText : "");
  fd.append("human_text", human);
  fd.append("clip_id", clip_id || "");
  fd.append("asr_ok", asrOk ? "1" : "0");
  fd.append("asr_hyp", hyp || "");
  fd.append("asr_error", asrErr || "");
  fd.append("typed_before_reveal", typedFirst ? "1" : "0");
  fd.append("revealed_at_ms", String(revealedAt));
  fd.append("paste_events", String(pastes));
  fd.append("keystrokes", String(keys));
  fd.append("consent_version", consent.version);
  fd.append("consent_sha256", consent.sha256);
  if (!clip_id && blob) fd.append("audio", blob, "clip.webm");   // ASR was down: send audio
  fetch("/asr/submit", {method:"POST", body:fd})
    .then(async r => { const j = await r.json().catch(() => ({}));
                       if (!r.ok) throw new Error(j.detail || ("HTTP " + r.status)); return j; })
    .then(j => {
      sent++; $("#counter").textContent = sent + " изпратени";
      $("#delTok").textContent = j.delete_token;
      $("#sendMsg").innerHTML = "<span class='ok'>Благодарим!</span>";
      if (!revealed && hyp) showDiff();          // always show the payoff after submitting
      $("#send").disabled = false;
    })
    .catch(e => { $("#sendMsg").innerHTML = "<span class='err'>" + e.message + "</span>";
                  $("#send").disabled = false; });
};
</script>
</body></html>"""


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def build_app(recognizer, bank, out_dir: pathlib.Path, consent: dict, model_id: str,
              max_sec: float, secret: bytes, logo_path: Optional[pathlib.Path]) -> FastAPI:
    app = FastAPI(title="OmniVoice-BG ASR playground", docs_url=None, redoc_url=None)

    raw_dir, staging_dir = out_dir / "raw", out_dir / "staging"
    raw_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    subs_path = out_dir / "submissions.jsonl"
    cons_path = out_dir / "consents.jsonl"
    del_path = out_dir / "deletions.jsonl"

    io_lock = threading.Lock()        # guards JSONL appends + counters
    infer_lock = threading.Lock()     # ONE decode at a time: bounds peak RSS, keeps RTF sane
    gate = threading.Lock()           # guards the admission counters below
    # `admitted` is incremented exactly once at admission and decremented exactly once in the
    # outer finally, on every path including errors — the ONLY counter that gates entry.
    # `inflight` is observability (how many are actually decoding); queued = admitted-inflight.
    state = {"admitted": 0, "inflight": 0, "done": 0, "speakers": set(), "audio_s": 0.0,
             "infer_s": 0.0, "last_ms": 0}
    MAX_ADMITTED = 3                  # 1 decoding + 2 waiting

    @app.middleware("http")
    async def admission(request: Request, call_next):
        """Reject overload BEFORE the multipart body is parsed.

        This must be middleware, not a check inside the route: declaring `audio: UploadFile`
        makes Starlette parse (and buffer) the whole upload before the handler runs, so an
        in-handler check would already be holding N uploads in RAM. Returning 429 here costs
        nothing and never touches the body.
        """
        if request.method == "POST" and request.url.path.rstrip("/") == "/asr/transcribe":
            with gate:
                if state["admitted"] >= MAX_ADMITTED:
                    return JSONResponse(
                        {"detail": "Системата е заета. Опитайте пак след няколко секунди."},
                        status_code=429, headers={"Retry-After": "10"})
                state["admitted"] += 1
            try:
                return await call_next(request)
            finally:
                with gate:
                    state["admitted"] -= 1
        return await call_next(request)

    seen_prompts: dict = {}           # speaker -> set(prompt_id), for coverage spreading
    if subs_path.exists():
        for ln in subs_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("prompt_id"):
                seen_prompts.setdefault(r.get("speaker", ""), set()).add(r["prompt_id"])
            state["done"] += 1
            state["speakers"].add(r.get("speaker_hash", ""))

    def speaker_hash(sp: str) -> str:
        return hashlib.sha256((SPLIT_SALT + "|" + sp).encode("utf-8")).hexdigest()[:16]

    def split_for(sp_hash: str) -> str:
        b = int(hashlib.sha256((SPLIT_SALT + sp_hash).encode()).hexdigest()[:8], 16) % 100
        return "eval" if b < EVAL_BUCKET_PCT else "train"

    def delete_token(sp_hash: str) -> str:
        return hmac.new(secret, sp_hash.encode(), hashlib.sha256).hexdigest()[:20]

    def append(path: pathlib.Path, row: dict):
        with io_lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------- read-only endpoints ----------------
    @app.get("/asr/health")
    def health():
        a, i = state["audio_s"], state["infer_s"]
        return {"status": "ok", "model": model_id, "decode": "greedy_search",
                "max_sec": max_sec, "prompts": len(bank), "submissions": state["done"],
                "speakers": len(state["speakers"]), "inflight": state["inflight"],
                "queued": max(0, state["admitted"] - state["inflight"]),
                "last_ms": state["last_ms"],
                "rtf": round(i / a, 3) if a > 0 else None, "peak_rss_mb": round(rss_mb(), 1)}

    @app.get("/asr/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/asr/consent")
    def get_consent():
        return consent

    @app.get("/asr/logo")
    def logo():
        if logo_path and logo_path.exists():
            head = logo_path.read_bytes()[:2]
            mime = "image/jpeg" if head == b"\xff\xd8" else "image/png"
            return Response(logo_path.read_bytes(), media_type=mime)
        raise HTTPException(404, "no logo")

    @app.get("/asr/favicon")
    def favicon():
        return logo()

    @app.get("/asr/prompt")
    def prompt(speaker: str = ""):
        """Next unseen prompt for this speaker; wraps around when they exhaust the bank."""
        done = seen_prompts.get(slug_speaker(speaker), set())
        pool = [p for p in bank if p["id"] not in done] or bank
        # Deterministic per (speaker, progress) rather than random: a reload shows the same
        # prompt instead of silently skipping one.
        idx = int(hashlib.sha256(f"{speaker}|{len(done)}".encode()).hexdigest()[:8], 16)
        return pool[idx % len(pool)]

    @app.get("/asr/stats")
    def stats():
        return {"submissions": state["done"], "speakers": len(state["speakers"]),
                "audio_hours": round(state["audio_s"] / 3600, 2), "prompts": len(bank)}

    @app.post("/asr/diff")
    async def diff(req: Request):
        body = await req.json()
        ops = word_diff((body.get("ref") or "")[:MAX_TEXT_CHARS],
                        (body.get("hyp") or "")[:MAX_TEXT_CHARS])
        esc = lambda s: (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        parts, errs = [], 0
        for tag, rw, hw in ops:
            if tag == "eq":
                parts.append(f'<span class="eq">{esc(hw)}</span>')
            elif tag == "sub":
                errs += 1
                parts.append(f'<span class="sub" title="казано: {esc(rw)}">{esc(hw)}</span>')
            elif tag == "del":
                errs += 1
                parts.append(f'<span class="del">{esc(rw)}</span>')
            else:
                errs += 1
                parts.append(f'<span class="ins">{esc(hw)}</span>')
        return {"html": " ".join(parts), "errors": errs,
                "ref_words": len((body.get("ref") or "").split())}

    # ---------------- transcribe (the only expensive path) ----------------
    @app.post("/asr/transcribe")
    async def transcribe(audio: UploadFile, mime: str = Form("")):
        # Admission/429 is handled by the `admission` middleware above, before the multipart
        # body is parsed. By the time we are here the request is already admitted.
        clip_id = f"asrp_{time.strftime('%Y%m%dT%H%M%S')}_{os.urandom(3).hex()}"
        tmp = staging_dir / f"{clip_id}.bin"
        try:
            tmp.write_bytes(await audio.read())
            dur = ffprobe_duration(tmp)
            if dur is None:
                raise HTTPException(400, "Записът не може да бъде прочетен.")
            if dur > max_sec + 1.0:
                raise HTTPException(400, f"Записът е {dur:.0f} с — максимумът е {max_sec:.0f} с.")
            if dur < 0.3:
                raise HTTPException(400, "Записът е твърде кратък.")

            with infer_lock:
                with gate:
                    state["inflight"] += 1
                try:
                    t0 = time.time()
                    samples = decode_to_16k(tmp)
                    s = recognizer.create_stream()
                    s.accept_waveform(16000, samples)
                    recognizer.decode_stream(s)
                    ms = int((time.time() - t0) * 1000)
                    text = s.result.text.strip()
                finally:
                    with gate:
                        state["inflight"] -= 1
            with gate:
                state["audio_s"] += dur; state["infer_s"] += ms / 1000; state["last_ms"] = ms
            logger.info("transcribe %s %.1fs -> %d chars in %d ms (rss %.0f MB)",
                        clip_id, dur, len(text), ms, rss_mb())
            return {"clip_id": clip_id, "text": text, "ms": ms, "duration": round(dur, 2)}
        except HTTPException:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            logger.exception("transcribe failed")
            raise HTTPException(503, f"Разпознаването не е налично: {exc}")

    # ---------------- submit (must work with the recognizer down) ----------------
    @app.post("/asr/submit")
    async def submit(speaker: str = Form(...), mode: str = Form("read"),
                     human_text: str = Form(...), consent_version: str = Form(...),
                     consent_sha256: str = Form(...), clip_id: str = Form(""),
                     prompt_id: str = Form(""), prompt_text: str = Form(""),
                     asr_ok: str = Form("0"), asr_hyp: str = Form(""),
                     asr_error: str = Form(""), typed_before_reveal: str = Form("1"),
                     revealed_at_ms: int = Form(0), paste_events: int = Form(0),
                     keystrokes: int = Form(0), audio: Optional[UploadFile] = None):
        sp = slug_speaker(speaker)
        if len(sp) < 2:
            raise HTTPException(400, "Невалиден псевдоним.")
        human = (human_text or "").strip()[:MAX_TEXT_CHARS]
        if len(human) < 2:
            raise HTTPException(400, "Липсва текст.")
        # Consent must match the exact bytes we are currently serving, so every row is
        # self-describing and changing the wording is a deliberate act (bump to asr-v2).
        if consent_sha256 != consent["sha256"] or consent_version != consent["version"]:
            raise HTTPException(400, "Съгласието е остаряло. Презаредете страницата.")

        sph = speaker_hash(sp)
        # Audio arrives one of two ways: staged by /asr/transcribe, or attached here when the
        # recognizer was unavailable. Either way the artifact is preserved.
        staged = staging_dir / f"{clip_id}.bin" if clip_id else None
        dest_dir = raw_dir / sph
        dest_dir.mkdir(parents=True, exist_ok=True)
        cid = clip_id or f"asrp_{time.strftime('%Y%m%dT%H%M%S')}_{os.urandom(3).hex()}"
        dest = dest_dir / f"{cid}.webm"
        if staged and staged.exists():
            staged.replace(dest)
        elif audio is not None:
            dest.write_bytes(await audio.read())
        else:
            raise HTTPException(400, "Липсва запис.")
        dur = ffprobe_duration(dest) or 0.0

        row = {
            "id": cid, "speaker": sp, "speaker_hash": sph,
            "consent_version": consent["version"], "consent_sha256": consent["sha256"],
            "mode": mode if mode in ("read", "free") else "read",
            "prompt_id": prompt_id or None, "prompt_text": prompt_text or None,
            "audio_path": str(dest.relative_to(out_dir)),
            "audio_sha256": sha256_file(dest), "bytes": dest.stat().st_size,
            "duration": round(dur, 3),
            "asr_ok": asr_ok == "1", "asr_hyp": asr_hyp or None,
            "asr_error": asr_error or None, "asr_model": model_id,
            "asr_decode": "greedy_search",
            "human_text": human,
            "edited": bool(prompt_text) and human.strip() != prompt_text.strip(),
            "typed_before_reveal": typed_before_reveal == "1",
            "revealed_at_ms": revealed_at_ms, "paste_events": paste_events,
            "keystrokes": keystrokes,
            "split": split_for(sph), "ts": int(time.time()),
        }
        append(subs_path, row)
        with gate:
            state["done"] += 1; state["speakers"].add(sph)
        if prompt_id:
            seen_prompts.setdefault(sp, set()).add(prompt_id)
        append(cons_path, {"speaker_hash": sph, "version": consent["version"],
                           "sha256": consent["sha256"], "ts": row["ts"]})
        logger.info("submit %s mode=%s split=%s dur=%.1f asr_ok=%s",
                    cid, row["mode"], row["split"], dur, row["asr_ok"])
        return {"ok": True, "id": cid, "delete_token": delete_token(sph)}

    @app.post("/asr/delete")
    async def delete(speaker: str = Form(...), token: str = Form(...)):
        sph = speaker_hash(slug_speaker(speaker))
        if not hmac.compare_digest(token.strip(), delete_token(sph)):
            raise HTTPException(403, "Невалиден код.")
        n = 0
        d = raw_dir / sph
        if d.exists():
            for f in d.iterdir():
                f.unlink(missing_ok=True); n += 1
        # Rows stay (append-only is a property worth keeping); consumers filter on tombstones.
        append(del_path, {"speaker_hash": sph, "files": n, "ts": int(time.time())})
        logger.info("delete %s -> %d files unlinked", sph, n)
        return {"ok": True, "deleted_files": n}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--model-dir", default=str(REPO / "models/asr/gp-v2"))
    ap.add_argument("--suffix", default="epoch-30-avg-5.int8")
    ap.add_argument("--out-dir", default=str(REPO / "data/bg/asr_playground"))
    ap.add_argument("--prompts", default=str(REPO / "config/bg/asr_prompts.txt"))
    ap.add_argument("--consent", default=str(REPO / "config/bg/consent_asr_v1.md"))
    ap.add_argument("--logo", default=str(REPO / "deploy/logo.png"))
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the MODEL_CARD.json sha256 check (dev only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"ERROR: {tool} not found — required (apt install ffmpeg)")

    md = pathlib.Path(args.model_dir)
    paths = {k: md / f"{k}-{args.suffix}.onnx" for k in ("encoder", "decoder", "joiner")}
    tokens = md / "tokens.txt"
    for p in list(paths.values()) + [tokens]:
        if not p.exists():
            sys.exit(f"ERROR: {p} missing — rsync the exported model first")

    # A half-finished rsync must fail at startup, not at a volunteer's first click.
    card_path = md / "MODEL_CARD.json"
    card = {}
    if card_path.exists() and not args.skip_verify:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        for name, want in (card.get("sha256") or {}).items():
            got = sha256_file(md / name)
            if got != want:
                sys.exit(f"ERROR: {name} sha256 mismatch\n  want {want}\n  got  {got}")
        logger.info("model files verified against MODEL_CARD.json")
    model_id = card.get("model_id", args.suffix)

    consent_path = pathlib.Path(args.consent)
    if not consent_path.exists():
        sys.exit(f"ERROR: {consent_path} missing")
    consent_bytes = consent_path.read_bytes()
    consent = {"version": consent_path.stem.replace("consent_", "").replace("_", "-"),
               "sha256": hashlib.sha256(consent_bytes).hexdigest(),
               "text": consent_bytes.decode("utf-8")}

    bank = read_prompts(pathlib.Path(args.prompts))
    if not bank:
        sys.exit(f"ERROR: no prompts in {args.prompts}")

    secret = os.environ.get("OMNIVOICE_ASR_SECRET", "").encode()
    if not secret:
        secret = hashlib.sha256(consent_bytes + b"|dev-fallback").digest()
        logger.warning("OMNIVOICE_ASR_SECRET unset — delete tokens are not stable across "
                       "restarts. Set it in the systemd EnvironmentFile for production.")

    import sherpa_onnx
    t0 = time.time()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(paths["encoder"]), decoder=str(paths["decoder"]),
        joiner=str(paths["joiner"]), tokens=str(tokens),
        num_threads=args.num_threads, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", provider="cpu", debug=False,
    )
    # Warm-up: pays graph-optimization cost now and proves the model decodes at STARTUP,
    # so a broken export fails here rather than in front of a volunteer.
    s = rec.create_stream()
    s.accept_waveform(16000, array.array("f", [0.0] * 16000))
    rec.decode_stream(s)
    logger.info("recognizer ready in %.1fs (%s, %d thread(s)) — rss %.0f MB",
                time.time() - t0, model_id, args.num_threads, rss_mb())

    out_dir = pathlib.Path(args.out_dir)
    app = build_app(rec, bank, out_dir, consent, model_id, args.max_sec, secret,
                    pathlib.Path(args.logo) if args.logo else None)
    logger.info("ASR playground on %s:%d — %d prompts, cap %.0fs, out=%s",
                args.host, args.port, len(bank), args.max_sec, out_dir)

    import uvicorn
    # limit_concurrency is a LAST-RESORT backstop against a connection flood, deliberately set
    # well above MAX_ADMITTED: uvicorn answers with a bare 503 and no Retry-After, so anything
    # low enough to fire during normal use would preempt the middleware's Bulgarian 429.
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                access_log=False, limit_concurrency=64)


if __name__ == "__main__":
    main()
