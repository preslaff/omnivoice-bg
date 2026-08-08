#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Recover pseudo-label windows that the two-teacher gate threw away, using GP-v2 as arbiter.

WHAT THIS RECOVERS. The weak-supervised axes were labelled by Whisper and gated against the
YouTube caption (an independent ASR): a window survived only if the two AGREED. Where they
disagreed there was no tiebreaker, so the audio was discarded — 67 h across ytdr/ytgi/ytbb,
plus 24 h of BNR windows rejected by language-ID on archival audio. That material is already
on disk, already segmented, and is disproportionately HARD (disagreement is why it was
dropped), which makes it worth more per hour than average training data.

WHY NOW. GP-v2 scores 4.12% on the frozen parliament holdout against Whisper large-v3's
15.37%, so it is a genuinely stronger third opinion than either original teacher. This is the
"Zipformer arbitration" tier already sketched in docs/BG_ASR_DATA_PIPELINE.md.

WHAT IT DELIBERATELY DOES NOT DO. It never touches the ACCEPTED windows. Those already passed
two-teacher agreement, audit at single-digit implied error, and native listening ("nearly
perfect" on ytbb+ytgi). Overwriting them with GP-v2 would risk replacing correct
human-caption text with the model's own errors and entrenching them — a self-training trap.
Only rejects are considered, so a bad arbitration decision can add noise but can never
DEGRADE a label that is already good.

THE DECISION RULE (3-way for YouTube, 2-way for BNR). For each rejected window we now have up
to three independent opinions: Whisper's stored text, the caption text re-sliced from the
.bg.vtt on disk, and a fresh GP-v2 decode. All compared through bg_text_norm (the single
scoring authority), so "127 лева" and "сто двадесет и седем лева" count as agreement.

    zip ≈ whisper   -> accept with WHISPER's text (whisper was right; caption was wrong)
    zip ≈ caption   -> accept with GP-V2's text   (whisper was wrong; caption only corroborates)
    neither         -> stay rejected              (genuinely ambiguous; do not guess)

Note which text becomes the label. Whisper's text and GP-v2's were both produced FROM this
window and are aligned to it; the caption is document-level text sliced by timestamp and
reliably bleeds past the boundary (measured — it routinely carries in words from the
neighbouring sentence). Writing it verbatim would put words in the label that are not in the
audio, i.e. teach hallucination. So when the caption is the one that turns out to be right, we
take GP-v2's aligned decode and let the caption serve only as corroboration. A speaking-rate
sanity band (--min-wps/--max-wps) catches whatever residual misalignment slips through, and
also catches Whisper truncation — a 9 s window labelled with four words teaches deletions.

For BNR (language-ID rejects, no captions) the test is whether GP-v2 produces confident,
long-enough Bulgarian at all, agreeing with the stored text.

RUNS ON CPU via the exported int8 ONNX, so it can run ALONGSIDE a GPU training job without
contending for the card. Use ./venv-asr, not the training venv.

    ./venv-asr/bin/python arbitrate_rejects.py --axis ytdr --limit 200   # dry run first
    ./venv-asr/bin/python arbitrate_rejects.py --axis ytdr --audit-sample 40
"""
import argparse
import html
import array
import json
import pathlib
import re
import subprocess
import sys
import time

import sherpa_onnx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bg_text_norm import normalize

REPO = pathlib.Path(__file__).resolve().parent
ICEFALL_ASR = pathlib.Path.home() / "icefall/egs/bg_parliament/ASR"

VTT_TS = re.compile(r"(\d\d):(\d\d):(\d\d)[.,](\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d)[.,](\d\d\d)")
VTT_TAG = re.compile(r"<[^>]+>")


def _append_new(words, new_words):
    """Append only the part of `new_words` that does not already end `words`.

    YouTube auto-captions scroll: each cue repeats the tail of the previous one and adds a
    few words. Collapsing merely-identical adjacent WORDS is not enough — whole phrase blocks
    repeat, which inflates the extracted caption several-fold and makes it impossible for the
    caption to ever match a short window. Overlap-merge on the longest suffix/prefix match is
    what actually reconstructs the spoken line."""
    for k in range(min(len(words), len(new_words)), 0, -1):
        if words[-k:] == new_words[:k]:
            words.extend(new_words[k:])
            return
    words.extend(new_words)


def parse_vtt(path):
    """[(t0, t1, text)] from a WebVTT file, with scroll-repetition merged out."""
    cues, t0, t1 = [], None, None
    buf = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = VTT_TS.search(line)
        if m:
            if t0 is not None and buf:
                cues.append((t0, t1, " ".join(buf)))
            g = [int(x) for x in m.groups()]
            t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            buf = []
        elif line.strip() and not line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            txt = html.unescape(VTT_TAG.sub("", line)).replace(">>", " ").strip()
            if txt:
                _append_new(buf, txt.split())
    if t0 is not None and buf:
        cues.append((t0, t1, " ".join(buf)))
    return cues


def caption_for(cues, t0, t1, pad=0.4):
    """Caption text overlapping [t0, t1], scroll-repetition merged across cue seams."""
    words = []
    for c0, c1, txt in cues:
        if c1 >= t0 - pad and c0 <= t1 + pad:
            _append_new(words, txt.split())
    return " ".join(words)


def decode_window(rec, media, t0, t1):
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}",
         "-i", str(media), "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace")[:160])
    a = array.array("f")
    a.frombytes(out.stdout)
    s = rec.create_stream()
    s.accept_waveform(16000, a)
    rec.decode_stream(s)
    return s.result.text.strip()


def wer(ref, hyp):
    """Normalized word error rate in [0, inf); 0 means identical after normalization."""
    import jiwer
    r, h = normalize(ref), normalize(hyp)
    if not r and not h:
        return 0.0
    if not r or not h:
        return 1.0
    return jiwer.process_words([r], [h]).wer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", required=True, choices=["ytdr", "ytgi", "ytbb", "bnr"])
    ap.add_argument("--model-dir", default=str(REPO / "models/asr/gp-v2"))
    ap.add_argument("--suffix", default="epoch-30-avg-5.int8")
    ap.add_argument("--tokens", default=str(REPO / "models/asr/gp-v2/tokens.txt"))
    ap.add_argument("--num-threads", type=int, default=4,
                    help="keep well under nproc if a GPU training job is running")
    ap.add_argument("--max-wer", type=float, default=0.5,
                    help="zwer at or below which GP-v2 counts as agreeing (pipeline precedent)")
    ap.add_argument("--min-words", type=int, default=3, help="skip windows shorter than this")
    ap.add_argument("--min-wps", type=float, default=0.8,
                    help="reject labels slower than this (words/sec) — catches truncation")
    ap.add_argument("--max-wps", type=float, default=6.0,
                    help="reject labels faster than this — catches boundary bleed")
    ap.add_argument("--why-prefix", default="", help="only windows whose `why` starts with this")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--audit-sample", type=int, default=30,
                    help="rows to write into an audit sheet for human listening")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    axis_dir = REPO / "data" / args.axis
    pl = axis_dir / "pseudolabels.jsonl"
    if not pl.exists():
        sys.exit(f"ERROR: {pl} missing")
    out_path = pathlib.Path(args.out) if args.out else axis_dir / "arbitrated.jsonl"
    audit_path = axis_dir / "arbitration_audit.md"

    md = pathlib.Path(args.model_dir)
    paths = {k: md / f"{k}-{args.suffix}.onnx" for k in ("encoder", "decoder", "joiner")}
    for p in paths.values():
        if not p.exists():
            sys.exit(f"ERROR: {p} missing — export the model first")
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(paths["encoder"]), decoder=str(paths["decoder"]),
        joiner=str(paths["joiner"]), tokens=args.tokens,
        num_threads=args.num_threads, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", provider="cpu",
    )

    # Collect the rejected windows up front so progress reporting can show a real ETA.
    todo = []
    for line in pl.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        media = pathlib.Path(r["path"])
        if not media.is_absolute():
            media = REPO / media
        for u in r.get("utts", []):
            if u.get("verdict") == "reject" and str(u.get("why", "")).startswith(args.why_prefix):
                todo.append((r["audio_id"], media, u))
    if args.limit:
        todo = todo[:args.limit]
    tot_h = sum((u["t1"] - u["t0"]) for _, _, u in todo) / 3600
    print(f"axis {args.axis}: {len(todo):,} rejected windows ({tot_h:.1f} h) to arbitrate\n"
          f"model {args.suffix}  threads {args.num_threads}  agree-threshold zwer<={args.max_wer}\n")

    vtt_cache = {}
    stats = {"accept_whisper": 0, "accept_caption": 0, "reject": 0, "reject_rate": 0,
             "error": 0, "short": 0}
    kept_h = 0.0
    rows, audit = [], []
    t_start = time.time()

    for i, (aid, media, u) in enumerate(todo, 1):
        t0, t1 = u["t0"], u["t1"]
        whisper_txt = (u.get("text") or "").strip()
        if len(whisper_txt.split()) < args.min_words:
            stats["short"] += 1
        else:
            try:
                zip_txt = decode_window(rec, media, t0, t1)
            except Exception as exc:
                stats["error"] += 1
                if stats["error"] <= 3:
                    print(f"  [err] {aid} {t0:.1f}: {exc}")
                zip_txt = None
            if zip_txt is not None:
                cap_txt = ""
                if args.axis != "bnr":
                    vtt = media.with_suffix("").with_suffix(".bg.vtt")
                    if not vtt.exists():
                        vtt = media.parent / f"{aid}.bg.vtt"
                    if vtt.exists():
                        if vtt not in vtt_cache:
                            vtt_cache[vtt] = parse_vtt(vtt)
                        cap_txt = caption_for(vtt_cache[vtt], t0, t1)

                w_zip_whisper = wer(whisper_txt, zip_txt) if whisper_txt else 1.0
                w_zip_cap = wer(cap_txt, zip_txt) if cap_txt else 1.0
                # Which text becomes the LABEL matters as much as whether to accept.
                # Whisper's text and GP-v2's were both produced FROM this window, so they are
                # aligned to it. The caption is document-level text sliced by timestamp and
                # reliably bleeds past the boundary — accepting it verbatim would write labels
                # containing words that are not in the audio, which teaches hallucination.
                # So the caption CORROBORATES, and GP-v2's own decode supplies the label.
                if w_zip_whisper <= args.max_wer and w_zip_whisper <= w_zip_cap:
                    verdict, text, agreed = "accept_whisper", whisper_txt, w_zip_whisper
                elif w_zip_cap <= args.max_wer:
                    verdict, text, agreed = "accept_caption", zip_txt, w_zip_cap
                else:
                    verdict, text, agreed = "reject", None, min(w_zip_whisper, w_zip_cap)
                # Speaking-rate sanity: catches residual misalignment and Whisper truncation
                # (a 9 s window labelled with four words teaches deletions).
                if text:
                    wps = len(text.split()) / max(t1 - t0, 1e-6)
                    if not (args.min_wps <= wps <= args.max_wps):
                        verdict, text = "reject_rate", None
                stats[verdict] = stats.get(verdict, 0) + 1
                if text:
                    kept_h += (t1 - t0) / 3600
                    rows.append({
                        "audio_id": aid, "path": str(media), "t0": t0, "t1": t1,
                        "text": text, "label_source": f"arbitration:{verdict}",
                        "zwer": round(agreed, 3), "arbiter": "gp-v2-e30-avg5-int8",
                        "whisper_text": whisper_txt, "caption_text": cap_txt,
                        "zip_text": zip_txt, "why_rejected": u.get("why"),
                    })
                    if len(audit) < args.audit_sample:
                        audit.append(rows[-1])

        if i % 200 == 0 or i == len(todo):
            el = time.time() - t_start
            rate = i / max(el, 1e-9)
            acc = stats["accept_whisper"] + stats["accept_caption"]
            print(f"  {i:7,}/{len(todo):,}  recovered {acc:6,} ({100*acc/max(i,1):4.1f}%) "
                  f"= {kept_h:5.1f} h  |  {rate:4.1f} win/s  ETA {(len(todo)-i)/rate/60:5.1f} min")

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if audit:
        with audit_path.open("w", encoding="utf-8") as f:
            f.write(f"# Arbitration audit — {args.axis}\n\n"
                    f"Listen to each clip and mark whether the chosen text is right. These are "
                    f"windows the two-teacher gate REJECTED and GP-v2 argued back in, so this "
                    f"sheet is the check on whether the arbiter is trustworthy on this axis.\n\n"
                    f"Extract a clip with:\n```\n"
                    f"ffmpeg -ss <t0> -t <dur> -i <path> -ar 16000 -ac 1 clip.wav\n```\n\n")
            for r in audit:
                f.write(f"## {r['audio_id']} @ {r['t0']:.1f}-{r['t1']:.1f}s "
                        f"({r['label_source']}, zwer {r['zwer']})\n"
                        f"- **chosen**: {r['text']}\n"
                        f"- whisper: {r['whisper_text']}\n"
                        f"- caption: {r['caption_text'] or '—'}\n"
                        f"- gp-v2:   {r['zip_text']}\n"
                        f"- verdict: [ ] correct  [ ] wrong\n\n")

    acc = stats["accept_whisper"] + stats["accept_caption"]
    print(f"\n{'='*68}\n"
          f"{args.axis}: recovered {acc:,} of {len(todo):,} windows = {kept_h:.1f} h\n"
          f"  backed whisper {stats['accept_whisper']:,} · corroborated by caption "
          f"{stats['accept_caption']:,} · still ambiguous {stats['reject']:,} · "
          f"failed rate check {stats['reject_rate']:,} · too short {stats['short']:,} · "
          f"errors {stats['error']:,}\n"
          f"  -> {out_path}\n  -> {audit_path} (listen before trusting this axis)\n{'='*68}")


if __name__ == "__main__":
    main()
