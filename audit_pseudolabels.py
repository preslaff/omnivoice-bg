#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Audit the Whisper pseudo-labels (BNR/YT) — three tiers, cheap -> gold.

  --report            TIER 1  automatic stats: accept/reject rates + reasons, hours,
                              chars-per-second plausibility (hallucination flag), repeated
                              n-gram loops, confidence (alp) distribution, per-category rates.
  --make-crosscheck N TIER 2  export N random ACCEPTED windows as a fbank CutSet whose refs
                              are the Whisper labels -> decode it with the v1 Zipformer
                              (decode.py --eval-set <cuts>). The reported "WER" = inter-model
                              disagreement between two INDEPENDENT ASRs = a quantitative upper
                              bound on label error, no listening needed. (Where both agree,
                              the label is almost certainly right.)
  --listening-sheet N TIER 3  export N wav clips + a markdown sheet, stratified by confidence
                              bucket (and category when available), for native spot-review —
                              same workflow as the Accentor vetting sheets.

    ./venv/bin/python audit_pseudolabels.py --report
    ./venv/bin/python audit_pseudolabels.py --make-crosscheck 500
    ./venv/bin/python audit_pseudolabels.py --listening-sheet 48
    # YT run:  add  --workdir data/ytdr --manifest data/ytdr/yt_manifest.jsonl
"""
import argparse
import collections
import json
import pathlib
import random
import re
import subprocess

random.seed(20260804)                       # reproducible samples


def load(args):
    wd = pathlib.Path(args.workdir)
    labels = [json.loads(l) for l in (wd / "pseudolabels.jsonl").open(encoding="utf-8")]
    meta = {}
    mp = pathlib.Path(args.manifest) if args.manifest else wd / "bnr_manifest.jsonl"
    if mp.exists():
        for l in mp.open(encoding="utf-8"):
            r = json.loads(l)
            meta[r.get("audio_id") or r.get("video_id")] = r
    return wd, labels, meta


def _loops(text, n=3):
    """count immediately-repeated n-grams (whisper hallucination loops)."""
    w = text.split()
    return sum(1 for i in range(len(w) - 2 * n) if w[i:i+n] == w[i+n:i+2*n])


def tier1_report(labels, meta):
    acc, rej = [], collections.Counter()
    per_cat = collections.defaultdict(lambda: [0, 0])
    for r in labels:
        cat = (meta.get(r["audio_id"]) or {}).get("category", "?")
        for u in r["utts"]:
            if u["verdict"] == "accept":
                acc.append((r["audio_id"], u, cat))
                per_cat[cat][0] += 1
            else:
                rej[u["why"].split("=")[0]] += 1
                per_cat[cat][1] += 1
    n_rej = sum(rej.values())
    tot = len(acc) + n_rej
    hrs = sum(u["t1"] - u["t0"] for _, u, _ in acc) / 3600
    print(f"windows: {tot:,} | ACCEPTED {len(acc):,} ({100*len(acc)/max(tot,1):.0f}%) "
          f"= {hrs:.1f} h | rejected {n_rej:,}")
    print("reject reasons:", dict(rej.most_common()))
    # confidence distribution of accepted
    alps = [u["alp"] for _, u, _ in acc if u.get("alp") is not None]
    if alps:
        alps.sort()
        q = lambda p: alps[int(p * (len(alps) - 1))]
        print(f"accepted avg_logprob: p10={q(.1):.2f} median={q(.5):.2f} p90={q(.9):.2f}")
    # plausibility: chars/sec (too low = silence labeled; too high = hallucination)
    cps_flags = [(a, u) for a, u, _ in acc
                 if (len(u["text"]) / max(u["t1"] - u["t0"], .1)) < 4
                 or (len(u["text"]) / max(u["t1"] - u["t0"], .1)) > 30]
    loops = [(a, u) for a, u, _ in acc if _loops(u["text"]) >= 2]
    print(f"suspicious accepted: {len(cps_flags)} chars/sec outliers, {len(loops)} n-gram loops "
          f"({100*(len(cps_flags)+len(loops))/max(len(acc),1):.1f}% of accepted)")
    if per_cat and "?" not in per_cat:
        print("per-category accept rate:")
        for c, (a, r_) in sorted(per_cat.items(), key=lambda x: -x[1][0]):
            print(f"  {c:20s} {a:6,d} acc / {r_:6,d} rej  ({100*a/max(a+r_,1):.0f}%)")
    return acc


def _slice_wav(src, t0, t1, dst):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(t0), "-to", str(t1),
                    "-i", src, "-ac", "1", "-ar", "16000", str(dst)], check=True,
                   capture_output=True)


def tier2_crosscheck(args, wd, acc):
    from lhotse import (CutSet, Fbank, FbankConfig, LilcomChunkyWriter, MonoCut,
                        Recording, SupervisionSegment)
    from bg_text_norm import normalize
    sample = random.sample(acc, min(args.make_crosscheck, len(acc)))
    cdir = wd / "audit_crosscheck"
    cdir.mkdir(exist_ok=True)
    cuts = []
    print(f"exporting {len(sample)} accepted windows for the Zipformer cross-check…")
    paths = {json.loads(l)["audio_id"]: json.loads(l)["path"]
             for l in (wd / "pseudolabels.jsonl").open(encoding="utf-8")}
    for i, (aid, u, _cat) in enumerate(sample, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(sample)}")
        wav = cdir / f"{aid}_{int(u['t0']*100):07d}.wav"
        if not wav.exists():
            try:
                _slice_wav(paths[aid], u["t0"], u["t1"], wav)
            except Exception:
                continue
        rec = Recording.from_file(wav, recording_id=wav.stem)
        ref = normalize(u["text"])
        if not ref:
            continue
        cuts.append(MonoCut(id=wav.stem, start=0.0, duration=rec.duration, channel=0,
                            recording=rec,
                            supervisions=[SupervisionSegment(
                                id=wav.stem, recording_id=wav.stem, start=0.0,
                                duration=rec.duration, channel=0, text=ref,
                                language="Bulgarian")]))
    extractor = Fbank(FbankConfig(num_mel_bins=80, sampling_rate=16000))
    cs = CutSet.from_cuts(cuts).compute_and_store_features(
        extractor=extractor, storage_path=str(cdir / "feats"),
        num_jobs=1, storage_type=LilcomChunkyWriter)
    out = cdir / "bg_cuts_bnr_audit.jsonl.gz"
    cs.to_file(out)
    print(f"wrote {len(cs)} audit cuts -> {out}\n")
    print("Now decode with the v1 Zipformer (refs = Whisper labels, so reported WER = ")
    print("inter-model DISAGREEMENT, an upper bound on label error):")
    print(f"  cd ~/icefall/egs/bg_parliament/ASR && export PYTHONPATH=~/icefall")
    print(f"  ~/WORK/OmniVoice/venv/bin/python zipformer/decode.py --epoch 30 --avg 5 \\")
    print(f"    --use-averaged-model 1 --eval-set {out.resolve()} \\")
    print(f"    --decoding-method greedy_search --exp-dir zipformer/exp \\")
    print(f"    --bpe-model data/lang_bpe_500/bpe.model --max-duration 600")
    print("Reading: <=20-25% disagreement is healthy for OOD-vs-OOD; inspect the errs-* file —")
    print("windows where BOTH models agree are near-certainly correct labels.")


def tier3_sheet(args, wd, acc):
    paths = {json.loads(l)["audio_id"]: json.loads(l)["path"]
             for l in (wd / "pseudolabels.jsonl").open(encoding="utf-8")}
    scored = [x for x in acc if x[1].get("alp") is not None]
    scored.sort(key=lambda x: x[1]["alp"])
    n = min(args.listening_sheet, len(scored))
    # stratify: worst / middle / best confidence thirds
    thirds = [scored[:len(scored)//3], scored[len(scored)//3:2*len(scored)//3],
              scored[2*len(scored)//3:]]
    sample = []
    for t in thirds:
        sample += random.sample(t, min(n // 3, len(t)))
    sdir = wd / "audit_listen"
    sdir.mkdir(exist_ok=True)
    lines = ["# Pseudo-label listening sheet\n",
             "Listen to each clip, mark: ok / minor (small wording errors) / bad (wrong text).\n"]
    for i, (aid, u, cat) in enumerate(sample, 1):
        wav = sdir / f"{i:03d}_{aid[:8]}.wav"
        try:
            _slice_wav(paths[aid], u["t0"], u["t1"], wav)
        except Exception:
            continue
        lines.append(f"\n## {i:03d}  [{cat}]  alp={u['alp']}  ({u['t1']-u['t0']:.0f}s)\n"
                     f"`{wav.name}`\n\n> {u['text']}\n\n- [ ] ok  - [ ] minor  - [ ] bad\n")
    sheet = sdir / "listening_sheet.md"
    sheet.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {len(sample)} clips + {sheet}")
    print("Verdict guide: 'bad' rate on the WORST-confidence third calibrates --min-logprob;")
    print("if even that third is mostly ok, thresholds can be relaxed to keep more data.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/bnr")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--make-crosscheck", type=int, default=0, metavar="N")
    ap.add_argument("--listening-sheet", type=int, default=0, metavar="N")
    args = ap.parse_args()
    wd, labels, meta = load(args)
    acc = tier1_report(labels, meta)          # report always runs (it's cheap)
    if args.make_crosscheck:
        tier2_crosscheck(args, wd, acc)
    if args.listening_sheet:
        tier3_sheet(args, wd, acc)


if __name__ == "__main__":
    main()
