#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Sample MANY renditions of a few imperative sentences to judge the RATE at
which the model produces the correct ending stress — read_book decodes
deterministically (one take), which hides whether a verb is partly-learned
(some takes right) or not-learned (all wrong). Includes an обади control the
model already renders correctly, to calibrate the ear.

Usage:
  ./venv/bin/python sample_imperatives.py --model exp/omnivoice_bg_ft8/checkpoint-3000
  # compare a baseline in the same run:
  ./venv/bin/python sample_imperatives.py --model exp/omnivoice_bg_ft7/checkpoint-3000 --out reading/ft7_samples
"""
import argparse
import pathlib

# pre-marked (verb-only, sparse style); --marking is bypassed (we feed marks).
SENTENCES = {
    "pravi":  ["Прави́ упражнения всеки ден.",
               "Не се колебай, прави́ го сега."],
    "hodi":   ["Ходи́ на разходка всяка сутрин.",
               "Ходи́ уверено по своя път."],
    "obadi_CONTROL": ["Обади́ се още днес."],   # ft8 gets this right -> calibrator
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="exp/omnivoice_bg_ft8/checkpoint-3000")
    ap.add_argument("--ref-audio", default="examples/reklama_home_max.wav")
    ap.add_argument("--ref-text-file", default="examples/reklama_home_max.txt")
    ap.add_argument("--n", type=int, default=8, help="renditions per sentence")
    ap.add_argument("--pos-temp", type=float, default=0.6)
    ap.add_argument("--cls-temp", type=float, default=0.6)
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--out", default="reading/ft8_samples")
    args = ap.parse_args()

    import torch
    import soundfile as sf
    from omnivoice import OmniVoice
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32
    model = OmniVoice.from_pretrained(args.model, device_map=dev, dtype=dtype)
    ref_text = open(args.ref_text_file).read().strip()
    prompt = model.create_voice_clone_prompt(args.ref_audio, ref_text=ref_text)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for verb, sents in SENTENCES.items():
        for si, s in enumerate(sents):
            for k in range(args.n):
                torch.manual_seed(1000 + k)
                if dev.startswith("cuda"):
                    torch.cuda.manual_seed_all(1000 + k)
                wav = model.generate(text=s, voice_clone_prompt=prompt,
                                     num_step=args.num_step,
                                     position_temperature=args.pos_temp,
                                     class_temperature=args.cls_temp)[0]
                sf.write(str(out / f"{verb}_s{si}_k{k}.wav"), wav, 24000)
            print(f"  {verb} s{si}: {args.n} takes -> {out}", flush=True)
    print(f"\ndone -> {out}/  (listen per verb: how many of {args.n} takes are "
          f"ending-stress? обади_CONTROL should be ~all right)")


if __name__ == "__main__":
    main()
