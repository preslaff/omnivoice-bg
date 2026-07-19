#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Teacher-distill pipeline: teach OmniVoice the imperative stress it refuses.

The imperative problem (наслади́/вдигни́/купи́) is HARD prior dominance: the
narration corpus has almost no imperatives, so OmniVoice only ever learned the
aorist/noun reading, and no inference-time mark overrides it (confirmed 0/8
even best-of-8, and unchanged in dense mode). The only cure is training data
with the correct imperative stress — which the corpus can't provide and
synthetic-from-OmniVoice can't either (it renders the same wrong prior).

BgTTS breaks the deadlock: it has a CORRECT imperative prior. So we distil:

    BgTTS (correct imperatives) -> Kaldi stress-judge -> human spot-check
                               -> marked manifest -> oversample -> OmniVoice ft

This is DISTILLATION (teach OmniVoice's prior), not label-harvest, so the
"synthetic audio never enters the Kaldi LABEL harvest" rule does not apply —
here Kaldi VERIFIES the synthetic clip rather than labelling human speech, and
your ear is the final gate. OmniVoice learns the stress; the VOICE still comes
from cloning at inference, so it won't sound like BgTTS.

Scaling "tens of thousands under supervision": you supervise the DESIGN (a few
hundred verbs in config/bg/imperative_verbs.txt, a few dozen frames in
imperative_frames.txt) and VALIDATE (spot-check ~hundreds); the machine
expands (verbs × frames), renders, and auto-judges the tens of thousands.

Stages (each resumable; run them in order, big ones by hand on a free GPU):
  expand  verbs × frames            -> exp/imperatives/bank.jsonl
  gen     BgTTS render each         -> exp/imperatives/wavs/<id>.wav (+ gen.jsonl)
  judge   Kaldi stress-check        -> annotate verdict (pass/fail/uncertain)
  review  human spot-check by verb  -> human_ok / human_no
  build   keep verified, marked     -> data/bg/imperatives/train.jsonl
          then: tokenize (--brighten) -> oversample into the ft data config

Usage:
  ./venv/bin/python teacher_distill.py expand
  ./venv/bin/python teacher_distill.py gen   --bgtts-dir models/bgtts-38m \
        --ref-audio examples/refs/     # a dir = cycle voices (stress, not voice)
  ./venv/bin/python teacher_distill.py judge --judge kaldi --kaldi-dir exp/kaldi_bg
  ./venv/bin/python teacher_distill.py review --per-verb 3     # your ear
  ./venv/bin/python teacher_distill.py build  --oversample 5
"""
import argparse
import json
import pathlib
import random
import subprocess

ACUTE = "́"
VOWELS = set("аъоуеиюя")
UP_V = "АЪОУЕИЮЯ"
UP2LOW = {u: l for u, l in zip(UP_V, "аъоуеиюя")}
LOW2UP = {l: u for l, u in zip("аъоуеиюя", UP_V)}
ROOT = pathlib.Path("exp/imperatives")


# ---------- shared helpers ----------
def strip_marks(s):
    return s.replace(ACUTE, "")


def stressed_vowel_ordinal(marked_word):
    """k-th vowel (0-based) that carries the acute, or None. Matches the
    harvest's convention so target and aligned indices are comparable."""
    k, seen = None, -1
    for i, ch in enumerate(marked_word):
        if ch in VOWELS:
            seen += 1
            if i + 1 < len(marked_word) and marked_word[i + 1] == ACUTE:
                k = seen
    return k


def read_list(path):
    """Lines minus comments/blanks; returns (text, homograph_flag) per line."""
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        homograph = "# HOMOGRAPH" in ln
        text = ln.split("#", 1)[0].strip()
        if text:
            out.append((text, homograph))
    return out


def load_jsonl(path):
    if not pathlib.Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def dump_jsonl(rows, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cap_first(s):
    return s[:1].upper() + s[1:] if s else s


# ---------- stage: expand ----------
def stage_expand(args):
    verbs = read_list(args.verbs)
    frames = [f for f, _ in read_list(args.frames) if "{imp}" in f]
    rows, seen = [], set()
    for vi, (verb_marked, homograph) in enumerate(verbs):
        for fi, frame in enumerate(frames):
            marked = cap_first(frame.replace("{imp}", verb_marked))
            plain = strip_marks(marked)
            if plain in seen:
                continue
            seen.add(plain)
            rows.append({
                "id": f"imp_{len(rows):06d}",
                "verb": strip_marks(verb_marked),
                "verb_marked": verb_marked,
                "homograph": homograph,
                "plain": plain,     # -> BgTTS
                "marked": marked,   # -> OmniVoice manifest (ground truth)
            })
    if args.max:
        rows = rows[: args.max]
    dump_jsonl(rows, ROOT / "bank.jsonl")
    n_h = sum(r["homograph"] for r in rows)
    print(f"{len(verbs)} verbs × {len(frames)} frames -> {len(rows)} unique "
          f"sentences ({n_h} from homograph verbs) -> {ROOT/'bank.jsonl'}")
    for k in (5, 8, 10):
        print(f"  at ~70% judge-pass, ×{k} oversample ≈ "
              f"{int(len(rows) * 0.7 * k):,} training exposures")


# ---------- stage: gen (BgTTS teacher) ----------
def stage_gen(args):
    import torch
    import soundfile as sf
    from omnivoice.cli.bgtts_engine import BgTTSEngine

    bank = load_jsonl(ROOT / "bank.jsonl")
    if not bank:
        raise SystemExit("run `expand` first (no bank.jsonl)")
    refs = _ref_list(args.ref_audio)
    print(f"BgTTS from {args.bgtts_dir}; {len(refs)} reference voice(s); "
          f"{args.seeds} seed(s)/sentence")
    wav_dir = ROOT / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # one engine per reference voice (speaker embedding is fixed per engine)
    engines = [BgTTSEngine(args.bgtts_dir, r, device=dev) for r in refs]
    gen_rows, done, skip = [], 0, 0
    for bi, row in enumerate(bank):
        for s in range(args.seeds):
            cid = f"{row['id']}_v{s}"
            wav = wav_dir / f"{cid}.wav"
            eng = engines[(bi + s) % len(engines)]     # cycle voices
            if not wav.exists():
                torch.manual_seed(args.seed + bi * 97 + s)
                if dev == "cuda":
                    torch.cuda.manual_seed_all(args.seed + bi * 97 + s)
                sf.write(str(wav), eng.synth(row["plain"]), 24000)
                done += 1
                if done % 50 == 0:
                    print(f"  rendered {done} (+{skip} existing)", flush=True)
            else:
                skip += 1
            gen_rows.append({**row, "cid": cid, "wav": str(wav),
                             "ref": refs[(bi + s) % len(refs)]})
    dump_jsonl(gen_rows, ROOT / "gen.jsonl")
    print(f"rendered {done}, skipped {skip} -> {len(gen_rows)} clips "
          f"({ROOT/'gen.jsonl'})")


def _ref_list(ref):
    p = pathlib.Path(ref)
    if p.is_dir():
        wavs = sorted(str(x) for x in p.glob("*.wav"))
        if not wavs:
            raise SystemExit(f"--ref-audio dir {ref!r} contains no *.wav files")
        return wavs
    if not p.exists():
        raise SystemExit(
            f"--ref-audio {ref!r} not found. A trailing '/' means a directory: "
            f"create it with a few clean reference wavs (voice variety = the "
            f"model learns stress, not a voice), or pass a single wav.")
    return [ref]


# ---------- stage: judge (Kaldi stress-check, the auto-gate) ----------
def stage_judge(args):
    rows = load_jsonl(ROOT / "gen.jsonl")
    if not rows:
        raise SystemExit("run `gen` first (no gen.jsonl)")
    if args.judge == "none":
        for r in rows:
            r["verdict"] = "unjudged"
        dump_jsonl(rows, ROOT / "judged.jsonl")
        print("judge=none: all clips marked 'unjudged' -> review gates them.")
        return

    # judge=kaldi: force-align each clip, read which vowel got stress '1',
    # compare to the target ordinal on the VERB. Reuses the harvest reader
    # (harvest_stress_labels.marked_from_phones) — the label machinery run in
    # reverse (verify, not label). See _kaldi_stress() for the align hook.
    from harvest_stress_labels import marked_from_phones
    npass = nfail = nunc = 0
    for i, r in enumerate(rows):
        target = stressed_vowel_ordinal(r["verb_marked"])
        phones = _kaldi_stress(r["wav"], r["plain"], r["verb"], args.kaldi_dir)
        if phones is None:
            r["verdict"], r["aligned"] = "uncertain", None
            nunc += 1
        else:
            aligned_marked = marked_from_phones(r["verb"], phones)
            got = stressed_vowel_ordinal(aligned_marked) if aligned_marked else None
            ok = got is not None and got == target
            r["verdict"], r["aligned"] = ("pass" if ok else "fail"), aligned_marked
            npass += ok
            nfail += (not ok)
        if (i + 1) % 200 == 0:
            print(f"  judged {i+1}/{len(rows)}", flush=True)
    dump_jsonl(rows, ROOT / "judged.jsonl")
    print(f"judge=kaldi: {npass} pass, {nfail} fail, {nunc} uncertain "
          f"-> {ROOT/'judged.jsonl'}  (spot-check pass-precision in `review`)")


def _kaldi_stress(wav, text, word, kaldi_dir):
    """INTEGRATION HOOK — force-align `wav` to `text` and return the phone list
    for `word` (e.g. ['k','u','p','i1']). Wire this to your kaldi_pilot.sh
    stage-3/4 aligner over a one-utterance dir; the '1'-suffixed vowel phone is
    the stress, which marked_from_phones() turns back into a marked word.
    Returns None if alignment fails (-> 'uncertain', sent to human review).

    Left as a hook because it depends on your local Kaldi layout; until wired,
    run the pipeline with --judge none and let your ear (review) be the gate."""
    if not kaldi_dir:
        return None
    helper = pathlib.Path(kaldi_dir) / "align_one.sh"   # you provide this
    if not helper.exists():
        return None
    try:
        out = subprocess.run(
            ["bash", str(helper), wav, text, word],
            capture_output=True, text=True, timeout=60, check=True).stdout
        return out.split()          # space-separated phones for `word`
    except Exception:               # noqa: BLE001  (align fail -> uncertain)
        return None


# ---------- stage: review (human spot-check, effort ∝ verbs) ----------
def stage_review(args):
    rows = load_jsonl(ROOT / "judged.jsonl") or load_jsonl(ROOT / "gen.jsonl")
    if not rows:
        raise SystemExit("run `gen` (and optionally `judge`) first")
    prev = {r["cid"]: r.get("human")
            for r in load_jsonl(ROOT / "review.jsonl")}   # resume
    # Stratified sample: per verb up to --per-verb; ALL homographs and all
    # judge fail/uncertain always surface (that's where BgTTS errs).
    by_verb = {}
    for r in rows:
        by_verb.setdefault(r["verb"], []).append(r)
    queue, rng = [], random.Random(args.seed)
    for verb, group in by_verb.items():
        must = [r for r in group
                if r.get("verdict") in ("fail", "uncertain") or r["homograph"]]
        rest = [r for r in group if r not in must]
        rng.shuffle(rest)
        per = args.per_verb * (2 if group[0]["homograph"] else 1)
        queue += must + rest[:max(0, per - len(must))]
    queue = [r for r in queue if prev.get(r["cid"]) not in ("ok", "no")]
    print(f"{len(rows)} clips; {len(queue)} queued for your ear "
          f"(verbs={len(by_verb)}, per-verb≈{args.per_verb}). "
          f"[y]=ok [n]=wrong [r]=replay [s]=skip [q]=quit")
    for r in rows:                         # carry prior verdicts forward
        if prev.get(r["cid"]):
            r["human"] = prev[r["cid"]]
    for r in queue:
        while True:
            _play(r["wav"], args.player)
            tag = f"[{r.get('verdict','?')}]" + (" HOMO" if r["homograph"] else "")
            ans = input(f"  {tag} {r['marked']}  (aligned={r.get('aligned','-')}) "
                        f"> ").strip().lower()
            if ans == "r":
                continue
            if ans == "q":
                _save_review(rows); return
            r["human"] = {"y": "ok", "n": "no"}.get(ans, "skip")
            break
    _save_review(rows)


def _play(wav, player):
    if player.lower() == "none":
        print(f"    (play manually) {wav}")
        return
    try:
        subprocess.run(player.split() + [wav], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"    (player '{player}' not found) {wav}")


def _save_review(rows):
    dump_jsonl(rows, ROOT / "review.jsonl")
    n_ok = sum(r.get("human") == "ok" for r in rows)
    n_no = sum(r.get("human") == "no" for r in rows)
    print(f"saved review.jsonl  (human: {n_ok} ok, {n_no} wrong)")


# ---------- stage: normalize (CAPITAL-vowel <-> combining acute) ----------
# You can't type a combining acute, so mark stress by UPPERCASING the stressed
# vowel (купИ, прАви) — the notation you already use. `normalize` converts your
# CAPITAL marks into the acute form the pipeline reads; `--to caps` goes the
# other way (turn my acutes into editable capitals for your review pass).
def _verb_to_acute(verb):
    """CAPITAL vowel wins: strip any existing acute, lowercase the capital,
    put the acute after it. Returns (new_verb, error_or_None)."""
    s = verb.replace(ACUTE, "")
    caps = [i for i, ch in enumerate(s) if ch in UP_V]
    if len(caps) > 1:
        return verb, "multiple CAPITAL vowels (mark exactly one)"
    if not caps:
        return (verb, None) if ACUTE in verb else (verb, "no stress mark")
    i = caps[0]
    return s[:i] + UP2LOW[s[i]] + ACUTE + s[i + 1:], None


def _verb_to_caps(verb):
    """Acute -> CAPITAL on the preceding vowel (for an editable review file)."""
    res = []
    for ch in verb:
        if ch == ACUTE:
            if res and res[-1] in VOWELS:
                res[-1] = LOW2UP[res[-1]]
        else:
            res.append(ch)
    return "".join(res)


def stage_normalize(args):
    path = pathlib.Path(args.file)
    lines = path.read_text(encoding="utf-8").splitlines()
    out, changed, problems, n_stress_q = [], 0, [], 0
    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith("#"):
            out.append(ln)                       # comment / blank / header
            continue
        if "# STRESS?" in ln:
            n_stress_q += 1
        cut = ln.find("#")
        head = ln if cut < 0 else ln[:cut]
        tail = "" if cut < 0 else ln[cut:]
        verb = head.strip()
        if args.to == "caps":
            new = _verb_to_caps(verb)
            err = None
        else:
            new, err = _verb_to_acute(verb)
        if err:
            problems.append((verb, err))
        changed += (new != verb)
        out.append(new + (f"  {tail}" if tail else ""))
    if args.out:
        pathlib.Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
        dest = args.out
    else:
        path.with_suffix(path.suffix + ".bak").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")   # backup
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        dest = str(path) + " (backup: *.bak)"
    print(f"{args.to}: rewrote {changed} verb line(s) -> {dest}")
    if problems:
        print(f"\n{len(problems)} line(s) need attention:")
        for v, e in problems:
            print(f"   {e:35s} {v!r}")
    if args.to == "acute":
        print(f"\n{n_stress_q} line(s) still flagged # STRESS? "
              f"(remove the flag as you confirm each — it's your checklist).")


# ---------- stage: build (verified -> marked training manifest) ----------
def stage_build(args):
    rows = load_jsonl(ROOT / "review.jsonl") or load_jsonl(ROOT / "judged.jsonl")
    if not rows:
        raise SystemExit("run gen/judge/review first")

    def keep(r):
        if r.get("human") == "no":
            return False                       # your ear vetoes
        if r.get("human") == "ok":
            return True                        # your ear approves
        return r.get("verdict") == "pass"      # else trust the judge

    kept = [r for r in rows if keep(r)]
    manifest = [{
        "id": r["cid"],
        "audio_path": str(pathlib.Path(r["wav"]).resolve()),
        "text": r["marked"],                   # MARKED = the whole point
        "language_id": "bg",
    } for r in kept]
    out = pathlib.Path(args.out)
    dump_jsonl(manifest, out)
    exposures = len(manifest) * args.oversample
    print(f"kept {len(kept)}/{len(rows)} verified clips -> {out}")
    print(f"oversample ×{args.oversample} ≈ {exposures:,} exposures in the ft mix\n")
    print("Next (run by hand on a free GPU):")
    print(f"  # 1. tokenize (brighten to match the bright dialect):")
    print(f"  ./venv/bin/python -m omnivoice.scripts.extract_audio_tokens \\")
    print(f"      --jsonl {out} --out-dir data/bg/imperatives/shards \\")
    print(f"      --brighten 6 --brighten-fc 4000")
    print(f"  # 2. add the shard to the ft data config with weight/repeat so it")
    print(f"  #    contributes ~{args.oversample}× (data/dataset.py multiplexing).")
    print(f"  # 3. finetune FROM the current production checkpoint:")
    print(f"  accelerate launch -m omnivoice.cli.train --train_config ... \\")
    print(f"      --data_config <with-imperatives> --output_dir exp/omnivoice_bg_ft7")
    print(f"  # 4. regression: re-run the imperative + narration A/B before promoting.")


# ---------- cli ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    e = sub.add_parser("expand"); e.set_defaults(fn=stage_expand)
    e.add_argument("--verbs", default="config/bg/imperative_verbs.txt")
    e.add_argument("--frames", default="config/bg/imperative_frames.txt")
    e.add_argument("--max", type=int, default=0, help="cap sentences (0=all)")

    g = sub.add_parser("gen"); g.set_defaults(fn=stage_gen)
    g.add_argument("--bgtts-dir", default="models/bgtts-38m")
    g.add_argument("--ref-audio", required=True, help="wav, or DIR to cycle voices")
    g.add_argument("--seeds", type=int, default=1, help="variants per sentence")
    g.add_argument("--seed", type=int, default=42)

    j = sub.add_parser("judge"); j.set_defaults(fn=stage_judge)
    j.add_argument("--judge", choices=["none", "kaldi"], default="none")
    j.add_argument("--kaldi-dir", default="", help="dir with align_one.sh hook")

    r = sub.add_parser("review"); r.set_defaults(fn=stage_review)
    r.add_argument("--per-verb", type=int, default=3, help="ear-checks per verb")
    r.add_argument("--player", default="ffplay -nodisp -autoexit -loglevel quiet",
                   help="'none' = just print paths")
    r.add_argument("--seed", type=int, default=42)

    b = sub.add_parser("build"); b.set_defaults(fn=stage_build)
    b.add_argument("--oversample", type=int, default=5)
    b.add_argument("--out", default="data/bg/imperatives/train.jsonl")

    n = sub.add_parser("normalize", help="CAPITAL-vowel <-> acute for review")
    n.set_defaults(fn=stage_normalize)
    n.add_argument("--file", default="config/bg/imperative_verbs.txt")
    n.add_argument("--to", choices=["acute", "caps"], default="acute",
                   help="acute = your CAPITAL marks -> pipeline form (in place, "
                        "makes a .bak); caps = my acutes -> editable CAPITALS")
    n.add_argument("--out", default=None, help="write here instead of in place")

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
