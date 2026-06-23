#!/usr/bin/env python3
"""Phase 6 A/B: synthesize the same Bulgarian sentences with multiple OmniVoice
checkpoints (and stock) using ONE fixed reference voice + deterministic decoding,
so you can compare by ear. See BG_FINETUNE_PLAN.md §7.

Models are loaded one at a time (16 GB-safe). Output files are named
`<idx>__<tag>.wav` so the same sentence across models sorts adjacently.

Example:
  ./venv/bin/python eval_bg_ab.py \
      --models k2-fsa/OmniVoice=stock \
               exp/bg_keep/checkpoint-2750=ft2750 \
               exp/omnivoice_bg/checkpoint-3000=ft3000 \
      --ref-from-dataset \
      --out eval_ab

Provide your own test sentences (one per line, UTF-8) with `--sentences mine.txt`
— ideally include the lines that had the wrong-emphasis problem. Provide a
specific reference voice with `--ref-audio x.wav --ref-text "..."`, or let
`--ref-from-dataset` auto-pick a clean clip from the training data.
"""

import argparse
import json
import pathlib

# TrainGen-style instructional sentences (incl. numbers/dates to probe normalization).
DEFAULT_SENTENCES = [
    "Натиснете бутона за добавяне на нов партньор.",
    "Изберете категория от падащото меню.",
    "Въведете името на фирмата в полето по-горе.",
    "Профилът беше публикуван успешно.",
    "За да продължите, кликнете върху бутона Запази.",
    "Системата записа промените автоматично.",
    "Срещата е насрочена за 14 март в 9 часа.",
    "Въведете сумата от 1250 лева в полето.",
]


def pick_ref_from_dataset(jsonl, min_s=4.0, max_s=9.0):
    import soundfile as sf

    for line in open(jsonl, encoding="utf-8"):
        r = json.loads(line)
        try:
            info = sf.info(r["audio_path"])
        except Exception:  # noqa: BLE001
            continue
        if min_s <= info.duration <= max_s and len(r["text"]) > 25:
            return r["audio_path"], r["text"]
    raise SystemExit(f"No suitable {min_s}-{max_s}s reference clip found in {jsonl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="space-separated path=tag, e.g. k2-fsa/OmniVoice=stock")
    ap.add_argument("--sentences", default=None, help="UTF-8 file, one sentence per line")
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--ref-from-dataset", action="store_true",
                    help="auto-pick a reference clip from --train-jsonl")
    ap.add_argument("--train-jsonl", default="data/bg/train.jsonl")
    ap.add_argument("--out", default="eval_ab")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--accent", action="store_true",
                    help="apply Bulgarian stress marking (matches the production server)")
    ap.add_argument("--lexicon", default="data/bg_lexicon/bg_stress.sqlite")
    ap.add_argument("--overrides", default="config/bg/bg_stress_overrides.json")
    args = ap.parse_args()

    import torch
    import soundfile as sf
    from omnivoice import OmniVoice

    # Sentences
    if args.sentences:
        sentences = [l.strip() for l in open(args.sentences, encoding="utf-8") if l.strip()]
    else:
        sentences = DEFAULT_SENTENCES

    # Optional stress marking, so the A/B reflects the deployed pipeline
    if args.accent:
        from bg_accent import BgAccenter
        acc = BgAccenter(args.lexicon, args.overrides)
        sentences = [acc.accent_text(s) for s in sentences]

    # Reference voice
    if args.ref_audio:
        ref_audio, ref_text = args.ref_audio, args.ref_text
    elif args.ref_from_dataset:
        ref_audio, ref_text = pick_ref_from_dataset(args.train_jsonl)
    else:
        raise SystemExit("Provide --ref-audio (+ --ref-text) or --ref-from-dataset")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Index of what each number means
    with open(out / "sentences.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            f.write(f"{i:02d}\t{s}\n")
    print(f"Reference: {ref_audio}\n  ref_text: {ref_text[:80]}")
    print(f"{len(sentences)} sentences x {len(args.models)} models -> {out}/")

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    for spec in args.models:
        path, _, tag = spec.partition("=")
        tag = tag or path.replace("/", "_")
        print(f"\n=== {tag}  ({path}) ===")
        model = OmniVoice.from_pretrained(path, device_map=dev, dtype=dtype)
        prompt = model.create_voice_clone_prompt(ref_audio, ref_text=ref_text)
        for i, s in enumerate(sentences):
            torch.manual_seed(args.seed)
            if dev.startswith("cuda"):
                torch.cuda.manual_seed_all(args.seed)
            audio = model.generate(
                text=s,
                voice_clone_prompt=prompt,
                num_step=args.num_step,
                position_temperature=0.0,
                class_temperature=0.0,
            )
            sf.write(str(out / f"{i:02d}__{tag}.wav"), audio[0], 24000)
            print(f"  [{i:02d}] {s[:50]}")
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\nDone. Listen in {out}/ — files sort as <idx>__<tag>.wav so each "
          f"sentence's variants are adjacent. Index: {out}/sentences.txt")


if __name__ == "__main__":
    main()
