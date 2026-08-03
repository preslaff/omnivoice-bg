#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Whisper external-comparison row for the BG ASR benchmark.

Transcribes the FROZEN test cuts with faster-whisper (large-v3, zero-shot) and scores it
against the same references the Zipformer was scored on — a legible "custom 649h model vs
off-the-shelf Whisper" number on identical data, identical normalizer.

NORMALIZATION PARITY. Whisper emits cased/punctuated text; references are lowercase /
no-punct / stress-stripped. BOTH go through ONE normalizer (lowercase, strip U+0301, strip
punctuation) so Whisper is not penalized for casing/punctuation it produced correctly.

NUMBERS — known residual caveat. The test references spell numbers out as words (0% contain
digits) and `num2words` has NO Bulgarian support, so we cannot reliably map Whisper's digit
output to the reference's spelled form (Bulgarian needs gender agreement: два/две). Wherever
a number is spoken and Whisper emits a digit, that costs ~1 error — so the reported Whisper
WER is a slight UPPER bound. Numbers are a minority of parliament tokens; the run prints how
many pure-digit tokens Whisper emitted so the magnitude is visible. Tightening this (a
gender-aware BG speller, or forcing Whisper to spell via suppress_tokens) is future work.

    pip install faster-whisper jiwer
    ./venv/bin/python eval_whisper_bg.py --limit 50      # smoke test on 50 cuts
    ./venv/bin/python eval_whisper_bg.py                 # full frozen test set (~up to ~1h)
"""
import argparse
import re
import pathlib

ACUTE = "́"
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)      # keeps Cyrillic letters, digits, spaces
_ISDIGIT = re.compile(r"^\d+$")


def normalize(s):
    """Shared normalizer — applied IDENTICALLY to hyp and ref (parity is the whole point)."""
    s = s.replace(ACUTE, "").lower()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(t for t in s.split() if t != "<unk>")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuts", default="data/fbank/bg_cuts_test.jsonl.gz",
                    help="the FROZEN test cuts (same set the Zipformer was scored on)")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="bg")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--out-dir", default="exp/whisper_eval")
    ap.add_argument("--limit", type=int, default=0, help="only decode the first N cuts (smoke)")
    args = ap.parse_args()

    from lhotse import CutSet
    from faster_whisper import WhisperModel
    import jiwer

    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cuts = CutSet.from_file(args.cuts)
    ids = list(cuts.ids)
    if args.limit:
        ids = ids[: args.limit]
    print(f"scoring {len(ids):,} cuts from {args.cuts} with Whisper {args.model}")

    # parity self-check: the normalizer must not mangle already-clean references
    sample = [cuts[i].supervisions[0].text for i in ids[:200]]
    changed = sum(1 for t in sample if normalize(t) != t)
    print(f"parity check: {changed}/{len(sample)} sampled refs change under normalize "
          f"(expected ~0 — refs are already normalized)")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    refs, hyps, hyp_digit_tokens = [], [], 0
    recog = out / f"recogs-test-whisper-{args.model}.txt"
    with recog.open("w", encoding="utf-8") as w:
        for n, cid in enumerate(ids, 1):
            c = cuts[cid]
            audio = c.load_audio()[0]            # (samples,) float32 @ 16 kHz
            segments, _ = model.transcribe(audio, language=args.language,
                                           beam_size=args.beam_size)
            raw = " ".join(seg.text for seg in segments)
            ref = normalize(c.supervisions[0].text)
            hyp = normalize(raw)
            hyp_digit_tokens += sum(1 for t in hyp.split() if _ISDIGIT.match(t))
            refs.append(ref); hyps.append(hyp)
            w.write(f"{cid}:\tref={ref}\n{cid}:\thyp={hyp}\n")
            if n % 100 == 0:
                print(f"  {n}/{len(ids)}  running WER={jiwer.wer(refs, hyps)*100:.2f}%")

    m = jiwer.process_words(refs, hyps)
    wer = m.wer * 100
    ref_words = m.hits + m.substitutions + m.deletions
    summary = out / f"wer-summary-test-whisper-{args.model}.txt"
    summary.write_text(
        f"system\tWER\tsub\tdel\tins\thits\thyp_digit_tokens\n"
        f"whisper-{args.model}\t{wer:.2f}\t{m.substitutions}\t{m.deletions}\t"
        f"{m.insertions}\t{m.hits}\t{hyp_digit_tokens}\n", encoding="utf-8")
    print(f"\nWhisper {args.model}  test WER = {wer:.2f}%  "
          f"(sub {m.substitutions} / del {m.deletions} / ins {m.insertions} / {ref_words} ref words)")
    print(f"  number caveat: Whisper emitted {hyp_digit_tokens} pure-digit tokens vs 0 in refs "
          f"(~{hyp_digit_tokens} of the errors are the spelled-vs-digit gap, an upper bound)")
    print(f"  transcripts -> {recog}")
    print(f"  summary     -> {summary}")
    print("Compare vs the Zipformer's 5.29% (fast_beam_search_nbest_LG) — SAME test set, SAME normalizer.")


if __name__ == "__main__":
    main()
