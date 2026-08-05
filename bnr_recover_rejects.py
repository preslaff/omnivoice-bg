#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Recover `lang=bg:<threshold` pseudo-label rejects via Zipformer-agreement arbitration.

Native listening (2026-08-05) confirmed the bg-detected lang-rejects are mostly good Bulgarian
(archival audio depresses Whisper's language-ID confidence). Rather than blindly relaxing
--lang-prob, this recovers them at TWO-TEACHER quality, with no re-transcription (their Whisper
text is already stored in pseudolabels.jsonl):

  --stage export   collect rejects with why=lang=bg:p >= --min-bg-prob (default 0.5), build a
                   fbank CutSet whose refs are the stored Whisper texts (cuts reference the
                   source mp3s directly — no slicing) -> <workdir>/recover/bg_cuts_recover.jsonl.gz
                   Then YOU decode it with the v1 Zipformer (command printed).
  --stage apply    parse the decode's recogs-* file, compute per-window Zipformer<->Whisper WER,
                   and flip to "accept" ONLY windows with WER <= --agree-wer (default 0.5) —
                   verdict why becomes "recovered:zwer=X.XX". Rewrites pseudolabels.jsonl
                   (backup kept). Stage-3 `cuts` then picks the recovered windows up on rebuild.

    ./venv/bin/python bnr_recover_rejects.py --stage export
    # ... run the printed decode.py command ...
    ./venv/bin/python bnr_recover_rejects.py --stage apply --recogs <errs-run's recogs-*.txt>
"""
import argparse
import ast
import json
import pathlib
import re

BG_RE = re.compile(r"lang=bg:([\d.]+)$")


def win_id(aid, t0):
    return f"rec_{aid}_{int(round(t0 * 100)):09d}"


def stage_export(args):
    from lhotse import (CutSet, Fbank, FbankConfig, LilcomChunkyWriter, MonoCut,
                        Recording, SupervisionSegment)
    from bg_text_norm import normalize
    wd = pathlib.Path(args.workdir)
    rdir = (wd / "recover").resolve()
    rdir.mkdir(exist_ok=True)
    out = rdir / "bg_cuts_recover.jsonl.gz"
    if out.exists():
        print(f"[skip] {out} exists — delete to rebuild")
        return
    rows = [json.loads(l) for l in (wd / "pseudolabels.jsonl").open(encoding="utf-8")]
    recs, cuts, n_cand = {}, [], 0
    for r in rows:
        wins = [(u, float(m.group(1))) for u in r["utts"]
                if u["verdict"] == "reject" and (m := BG_RE.match(u.get("why", "")))]
        wins = [(u, p) for u, p in wins if p >= args.min_bg_prob]
        if not wins:
            continue
        if r["audio_id"] not in recs:
            try:
                recs[r["audio_id"]] = Recording.from_file(r["path"], recording_id=r["audio_id"])
            except Exception:
                continue
        rec = recs[r["audio_id"]]
        for u, p in wins:
            n_cand += 1
            ref = normalize(u["text"])
            dur = round(min(u["t1"], rec.duration) - u["t0"], 2)
            if not ref or dur < 1.0:
                continue
            cid = win_id(r["audio_id"], u["t0"])
            cuts.append(MonoCut(
                id=cid, start=u["t0"], duration=dur, channel=0, recording=rec,
                supervisions=[SupervisionSegment(
                    id=cid, recording_id=r["audio_id"], start=0.0, duration=dur,
                    channel=0, text=ref, language="Bulgarian")]))
    print(f"candidates (lang=bg p>={args.min_bg_prob}): {n_cand:,} -> {len(cuts):,} cuts "
          f"({sum(c.duration for c in cuts)/3600:.1f} h)")
    extractor = Fbank(FbankConfig(num_mel_bins=80, sampling_rate=16000))
    cs = CutSet.from_cuts(cuts).resample(16000).compute_and_store_features(
        extractor=extractor, storage_path=str(rdir / "feats"),
        num_jobs=args.jobs, storage_type=LilcomChunkyWriter)
    cs.to_file(out)
    print(f"wrote {len(cs):,} cuts -> {out}\n")
    print("Now decode with the v1 Zipformer:")
    print(f"  cd ~/icefall/egs/bg_parliament/ASR && export PYTHONPATH=~/icefall")
    print(f"  ~/WORK/OmniVoice/venv/bin/python zipformer/decode.py --epoch 30 --avg 5 \\")
    print(f"    --use-averaged-model 1 --eval-set {out} \\")
    print(f"    --decoding-method greedy_search --exp-dir zipformer/exp \\")
    print(f"    --bpe-model data/lang_bpe_500/bpe.model --max-duration 600")
    print("Then:  bnr_recover_rejects.py --stage apply --recogs zipformer/exp/greedy_search/recogs-bg_cuts_recover-*.txt")


def _parse_recogs(path):
    """icefall recogs: '<id>:\tref=[...]' / '<id>:\thyp=[...]' -> {id: (ref_words, hyp_words)}"""
    out = {}
    for line in pathlib.Path(path).open(encoding="utf-8"):
        m = re.match(r"^(\S+):\s*(ref|hyp)=(\[.*\])\s*$", line.strip())
        if not m:
            continue
        cid, kind, lst = m.group(1), m.group(2), ast.literal_eval(m.group(3))
        out.setdefault(cid, {})[kind] = lst
    return {k: (v.get("ref", []), v.get("hyp", [])) for k, v in out.items() if "ref" in v}


def stage_apply(args):
    import jiwer
    wd = pathlib.Path(args.workdir)
    recogs = _parse_recogs(args.recogs)
    print(f"parsed {len(recogs):,} decoded windows from {args.recogs}")
    plfile = wd / "pseudolabels.jsonl"
    backup = wd / "pseudolabels.jsonl.bak"
    if not backup.exists():
        backup.write_bytes(plfile.read_bytes())
        print(f"backup -> {backup}")
    rows = [json.loads(l) for l in plfile.open(encoding="utf-8")]
    n_rec = n_seen = 0
    rec_sec = 0.0
    for r in rows:
        for u in r["utts"]:
            if not (u["verdict"] == "reject" and BG_RE.match(u.get("why", ""))):
                continue
            cid = win_id(r["audio_id"], u["t0"])
            if cid not in recogs:
                continue
            n_seen += 1
            ref, hyp = recogs[cid]
            if not ref:
                continue
            w = jiwer.wer(" ".join(ref), " ".join(hyp) or " ")
            if w <= args.agree_wer:
                u["verdict"] = "accept"
                u["why"] = f"recovered:zwer={w:.2f}"
                n_rec += 1
                rec_sec += u["t1"] - u["t0"]
        r["n_accept"] = sum(1 for u in r["utts"] if u["verdict"] == "accept")
    with plfile.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"arbitrated {n_seen:,} windows: RECOVERED {n_rec:,} "
          f"({100*n_rec/max(n_seen,1):.0f}%) = {rec_sec/3600:.1f} h at zwer<={args.agree_wer}")
    print("Rebuild the training cuts to include them:")
    print("  rm -f data/fbank/bg_cuts_bnr.jsonl.gz && rm -rf data/fbank/bg_feats_bnr")
    print("  ./venv/bin/python bnr_pseudolabel.py --stage cuts")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["export", "apply"])
    ap.add_argument("--workdir", default="data/bnr")
    ap.add_argument("--min-bg-prob", type=float, default=0.5)
    ap.add_argument("--agree-wer", type=float, default=0.5,
                    help="max Zipformer<->Whisper WER to recover a window")
    ap.add_argument("--recogs", default=None, help="recogs-*.txt from the arbitration decode")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.stage == "apply" and not args.recogs:
        raise SystemExit("--stage apply requires --recogs <recogs-file>")
    {"export": stage_export, "apply": stage_apply}[args.stage](args)


if __name__ == "__main__":
    main()
