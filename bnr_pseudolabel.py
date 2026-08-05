#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""BNR pseudo-label pipeline: speech/music quarantine -> Whisper -> faithfulness filter -> cuts.

Step 4 of docs/BG_ASR_BNR_PLAN.md, over bnr_crawl.py's output (data/bnr/bnr_manifest.jsonl +
audio/*.mp3). Three RESUMABLE stages, run as separate invocations (segmentation is TensorFlow,
transcription is torch — keeping them in separate processes avoids GPU tussles):

  STAGE 1  segment    inaSpeechSegmenter labels each mp3 into speech/music/noise regions.
                      KEEP speech, DROP music — the musical-play quarantine (metadata can't do
                      it: descriptions are generic; plain VAD fails since singing = "voice").
                      -> data/bnr/segments.jsonl   (one line per mp3: speech regions + stats)
  STAGE 2  transcribe faster-whisper large-v3 over each speech region, Bulgarian-only, with the
                      FAITHFULNESS FILTER (filtering, never LLM rewriting):
                        * language: detected 'bg' with prob >= --lang-prob (drops foreign)
                        * avg_logprob >= --min-logprob   (low-confidence = likely mishearing)
                        * no_speech_prob <= --max-nospeech
                        * compression_ratio <= --max-compression (hallucination-loop detector)
                      -> data/bnr/pseudolabels.jsonl  (accepted + rejected-with-reason)
  STAGE 3  cuts       accepted labels -> 16 kHz wavs -> bg_text_norm refs -> 80-dim fbank
                      -> data/fbank/bg_cuts_bnr.jsonl.gz  (merge_train_corpus.py-ready)

    pip install inaSpeechSegmenter          # stage 1 (TensorFlow; CPU is fine)
    # STAGE-1 MEMORY LEAK (Keras predict-loop): the segmenter's RSS grows with every file —
    # ~47GB after ~140 hour-long files (OOM-killed 2026-08-05). For LONG-file sources (YouTube
    # episodes) run stage 1 in --limit 40 batches in a shell loop; each fresh process resets
    # the leak. Short-file sources (BNR ~3min clips) survive a single run.
    ./venv/bin/python bnr_pseudolabel.py --stage segment    [--limit 40, loop until done]
    ./venv/bin/python bnr_pseudolabel.py --stage transcribe [--limit 20]
    ./venv/bin/python bnr_pseudolabel.py --stage cuts

Thresholds are deliberately conservative (faithful-but-noisy > fluent-but-wrong); calibrate on a
small vetted set (plan "Open questions"). Forced-alignment agreement (align Whisper text back with
our own model) is a planned tightening, not in this draft.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess

BNR = pathlib.Path("data/bnr")            # workdir; overridden by --workdir (e.g. data/ytdr)
SEGMENTS = BNR / "segments.jsonl"
LABELS = BNR / "pseudolabels.jsonl"
MANIFEST = BNR / "bnr_manifest.jsonl"     # overridden by --manifest
MIN_SPEECH_SEC = 1.5          # ignore blips
MAX_UTT_SEC = 28.0            # whisper sweet-spot; longer regions are windowed by whisper itself


def _done_ids(path, key):
    ids = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                ids.add(json.loads(line)[key])
            except Exception:
                pass
    return ids


def _norm_rec(r):
    """Accept both manifests: BNR {audio_id,path} and YT {video_id,audio,subs}."""
    return {"audio_id": r.get("audio_id") or r.get("video_id"),
            "path": r.get("path") or r.get("audio"),
            "subs": r.get("subs")}


# --- YouTube auto-caption VTT: rolling duplicated cues with <c>-word tags -> clean cues ---
_TS = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d\d\d)")
_TAG = re.compile(r"<[^>]+>")


def parse_vtt(path):
    """-> [(start_sec, end_sec, text), ...] deduplicating YouTube's rolling repeats."""
    cues, prev_lines = [], []
    t0 = t1 = None
    for raw in pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if "-->" in line:
            ts = _TS.findall(line)
            if len(ts) >= 2:
                t0 = int(ts[0][0])*3600 + int(ts[0][1])*60 + int(ts[0][2]) + int(ts[0][3])/1000
                t1 = int(ts[1][0])*3600 + int(ts[1][1])*60 + int(ts[1][2]) + int(ts[1][3])/1000
            prev_lines = [c[2] for c in cues[-1:]]         # last cue's text for dedup
            continue
        if t0 is None or not line or line == "WEBVTT" or line.startswith(("Kind:", "Language:")):
            continue
        import html
        text = html.unescape(_TAG.sub("", line)).replace(">>", " ").strip()
        if text and text not in prev_lines:                # drop the rolling repeat
            cues.append((t0, t1, text))
    return cues


def cues_text(cues, t0, t1):
    """caption text whose cue-midpoint falls inside [t0, t1]."""
    return " ".join(c[2] for c in cues if t0 <= (c[0] + c[1]) / 2 < t1).strip()


# ---------------------------------------------------------------- stage 1: segment
def stage_segment(args):
    from inaSpeechSegmenter import Segmenter          # TensorFlow — import only in this stage
    seg = Segmenter()                                 # labels: speech / music / noise / noEnergy
    recs = [_norm_rec(json.loads(l)) for l in open(MANIFEST, encoding="utf-8")]
    done = _done_ids(SEGMENTS, "audio_id")
    todo = [r for r in recs if r["audio_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"stage 1 segment: {len(recs)} in manifest, {len(done)} done, {len(todo)} to do")
    with SEGMENTS.open("a", encoding="utf-8") as out:
        for i, r in enumerate(todo, 1):
            try:
                regions = seg(r["path"])              # [(label, start, end), ...]
            except Exception as e:
                print(f"  [fail] {r['audio_id']}: {e}")
                continue
            speech = [(round(s, 2), round(e, 2)) for lab, s, e in regions
                      if lab in ("speech", "male", "female") and e - s >= MIN_SPEECH_SEC]
            total = sum(e - s for _, s, e in regions) or 1.0
            music = sum(e - s for lab, s, e in regions if lab == "music")
            out.write(json.dumps({
                "audio_id": r["audio_id"], "path": r["path"], "subs": r.get("subs"),
                "speech_regions": speech,
                "speech_sec": round(sum(e - s for s, e in speech), 1),
                "music_frac": round(music / total, 3),
            }, ensure_ascii=False) + "\n")
            out.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}")
    print(f"-> {SEGMENTS}")


# ---------------------------------------------------------------- stage 2: transcribe
def stage_transcribe(args):
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device, compute_type="float16")
    rows = [json.loads(l) for l in SEGMENTS.open(encoding="utf-8")]
    done = _done_ids(LABELS, "audio_id")
    todo = [r for r in rows if r["audio_id"] not in done and r["speech_regions"]]
    if args.limit:
        todo = todo[: args.limit]
    print(f"stage 2 transcribe: {len(rows)} segmented, {len(done)} done, {len(todo)} to do")
    import soundfile as sf
    import io
    if args.agree_subs:
        from jiwer import wer as jiwer_wer
        from bg_text_norm import normalize as _tn
    with LABELS.open("a", encoding="utf-8") as out:
        for i, r in enumerate(todo, 1):
            cues = None
            if args.agree_subs and r.get("subs") and pathlib.Path(r["subs"]).exists():
                cues = parse_vtt(r["subs"])
            # decode whole mp3 once to 16k mono via ffmpeg, slice regions in memory
            p = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", r["path"],
                 "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
                capture_output=True)
            if p.returncode != 0:
                print(f"  [fail-decode] {r['audio_id']}")
                continue
            audio, sr = sf.read(io.BytesIO(p.stdout))
            utts = []
            for (s, e) in r["speech_regions"]:
                # split long regions into <= MAX_UTT_SEC windows on the region grid
                t = s
                while t < e - 0.3:
                    t2 = min(t + MAX_UTT_SEC, e)
                    clip = audio[int(t * sr): int(t2 * sr)]
                    segs, info = model.transcribe(clip, beam_size=5, word_timestamps=False,
                                                  language=None)   # let it DETECT language
                    segs = list(segs)
                    text = " ".join(sg.text for sg in segs).strip()
                    verdict, why = "accept", ""
                    alp = None
                    if info.language != "bg" or (info.language_probability or 0) < args.lang_prob:
                        verdict, why = "reject", f"lang={info.language}:{info.language_probability:.2f}"
                    elif not segs or not text:
                        verdict, why = "reject", "empty"
                    else:
                        alp = sum(sg.avg_logprob for sg in segs) / len(segs)
                        nsp = max(sg.no_speech_prob for sg in segs)
                        cmp_ = max(sg.compression_ratio for sg in segs)
                        if alp < args.min_logprob:
                            verdict, why = "reject", f"logprob={alp:.2f}"
                        elif nsp > args.max_nospeech:
                            verdict, why = "reject", f"nospeech={nsp:.2f}"
                        elif cmp_ > args.max_compression:
                            verdict, why = "reject", f"compression={cmp_:.2f}"
                    # two-teacher agreement: Whisper hyp vs the platform's own caption for
                    # this window. Independent ASR errors rarely coincide -> agreement is a
                    # higher-precision keep signal than confidence alone. No caption overlap
                    # (intro/jingle gaps) -> fall through to the confidence verdict.
                    if verdict == "accept" and cues is not None:
                        cap = cues_text(cues, t, t2)
                        if cap:
                            d = jiwer_wer(_tn(cap) or " ", _tn(text) or " ")
                            if d > args.agree_wer:
                                verdict, why = "reject", f"disagree={d:.2f}"
                            else:
                                why = f"agree={d:.2f}"
                    utts.append({"t0": round(t, 2), "t1": round(t2, 2), "text": text,
                                 "verdict": verdict, "why": why,
                                 "alp": None if alp is None else round(alp, 3)})
                    t = t2
            n_ok = sum(u["verdict"] == "accept" for u in utts)
            out.write(json.dumps({"audio_id": r["audio_id"], "path": r["path"],
                                  "utts": utts, "n_accept": n_ok}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 5 == 0:
                print(f"  {i}/{len(todo)}  (last file: {n_ok}/{len(utts)} accepted)")
    print(f"-> {LABELS}")


# ---------------------------------------------------------------- stage 3: cuts
def stage_cuts(args):
    from lhotse import (CutSet, Fbank, FbankConfig, LilcomChunkyWriter, MonoCut,
                        Recording, SupervisionSegment)
    from bg_text_norm import normalize
    import soundfile as sf
    import subprocess as sp
    import io

    out_dir = pathlib.Path("data/fbank")
    manifest = out_dir / f"{args.out_name}.jsonl.gz"
    prefix = args.out_name.replace("bg_cuts_", "")
    if manifest.exists():
        print(f"[skip] {manifest} exists — delete to rebuild")
        return
    wdir = BNR / "wav16"
    wdir.mkdir(exist_ok=True)

    rows = [json.loads(l) for l in LABELS.open(encoding="utf-8")]
    n_src = sum(1 for r in rows if r["n_accept"] > 0)
    print(f"stage 3 cuts: decoding {n_src} source files to 16k wav (cached ones skipped)…", flush=True)
    cuts, n_rej, n_done = [], 0, 0
    for r in rows:
        acc = [u for u in r["utts"] if u["verdict"] == "accept"]
        n_rej += sum(1 for u in r["utts"] if u["verdict"] != "accept")
        if not acc:
            continue
        # one 16k wav per source mp3 (decoded once), cuts slice into it
        wav = wdir / f"{r['audio_id']}.wav"
        if not wav.exists():
            p = sp.run(["ffmpeg", "-nostdin", "-v", "error", "-i", r["path"],
                        "-ac", "1", "-ar", "16000", str(wav)], capture_output=True)
            if p.returncode != 0:
                continue
        rec = Recording.from_file(wav, recording_id=r["audio_id"])
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{n_src} files ({len(cuts):,} cuts so far)", flush=True)
        for j, u in enumerate(acc):
            ref = normalize(u["text"])
            if not ref:
                continue
            dur = round(min(u["t1"], rec.duration) - u["t0"], 2)
            if dur < 1.0:
                continue
            cid = f"{prefix}_{r['audio_id']}_{j:03d}"
            cuts.append(MonoCut(
                id=cid, start=u["t0"], duration=dur, channel=0, recording=rec,
                supervisions=[SupervisionSegment(
                    id=cid, recording_id=r["audio_id"], start=0.0, duration=dur,
                    channel=0, text=ref, language="Bulgarian")]))
    print(f"stage 3 cuts: {len(cuts):,} accepted utts ({n_rej:,} rejected upstream)")
    if not cuts:
        return
    extractor = Fbank(FbankConfig(num_mel_bins=80, sampling_rate=16000))
    cs = CutSet.from_cuts(cuts)
    jobs = 1 if len(cuts) < 50 else args.jobs      # tiny batches: skip multiprocessing overhead
    cs = cs.compute_and_store_features(
        extractor=extractor, storage_path=str(out_dir / args.out_name.replace("bg_cuts_", "bg_feats_")),
        num_jobs=jobs, storage_type=LilcomChunkyWriter)
    cs.to_file(manifest)
    print(f"wrote {len(cs):,} cuts ({sum(c.duration for c in cs)/3600:.1f} h) -> {manifest}")
    print("Next: merge_train_corpus.py --train", manifest, "(+ holdout a BNR slice for eval)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["segment", "transcribe", "cuts"])
    ap.add_argument("--manifest", default=None,
                    help="input manifest (default <workdir>/bnr_manifest.jsonl); accepts BNR "
                         "{audio_id,path} or yt_harvest {video_id,audio,subs} records")
    ap.add_argument("--workdir", default="data/bnr",
                    help="working dir for segments/pseudolabels/wav16 (YT: data/ytdr)")
    ap.add_argument("--out-name", default="bg_cuts_bnr",
                    help="stage-3 manifest basename (YT: bg_cuts_ytdr)")
    ap.add_argument("--agree-subs", action="store_true",
                    help="two-teacher agreement: also require the platform caption (subs vtt) "
                         "to agree with Whisper (inter-teacher WER <= --agree-wer); windows "
                         "without caption overlap fall back to the confidence verdict")
    ap.add_argument("--agree-wer", type=float, default=0.5,
                    help="max Whisper<->caption WER to accept (both normalized); calibrate")
    ap.add_argument("--limit", type=int, default=0, help="files per run (smoke/batching)")
    ap.add_argument("--model", default="models/faster-whisper-large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4))
    # faithfulness thresholds (conservative; calibrate on a vetted sample)
    ap.add_argument("--lang-prob", type=float, default=0.8)
    ap.add_argument("--min-logprob", type=float, default=-0.7)
    ap.add_argument("--max-nospeech", type=float, default=0.5)
    ap.add_argument("--max-compression", type=float, default=2.4)
    args = ap.parse_args()
    global BNR, SEGMENTS, LABELS, MANIFEST
    BNR = pathlib.Path(args.workdir)
    SEGMENTS = BNR / "segments.jsonl"
    LABELS = BNR / "pseudolabels.jsonl"
    MANIFEST = pathlib.Path(args.manifest) if args.manifest else BNR / "bnr_manifest.jsonl"
    {"segment": stage_segment, "transcribe": stage_transcribe, "cuts": stage_cuts}[args.stage](args)


if __name__ == "__main__":
    main()
