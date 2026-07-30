#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Train a SentencePiece BPE model on the Bulgarian ASR train transcripts (icefall prep).

Produces data/lang_bpe_500/ in icefall's expected layout:
  * transcript_words.txt — one train supervision per line (SentencePiece input)
  * bpe.model / bpe.vocab — the trained model (unigram, matching icefall's default)
  * tokens.txt            — `<piece> <id>` per line; id 0=<blk>, 1=<sos/eos>, 2=<unk>,
    then the learned subwords (the modeling units the Zipformer predicts)

Transcripts are already normalized upstream (lowercase, no punctuation, stress/U+0301
and <unk> stripped), so character coverage is 1.0 over the small Cyrillic charset.

    ./venv/bin/python train_bpe_bg.py                 # vocab 500
    ./venv/bin/python train_bpe_bg.py --vocab-size 1000
"""
import argparse
import pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-cuts", default="data/lhotse/bg/cuts_train.jsonl.gz")
    ap.add_argument("--vocab-size", type=int, default=500)
    ap.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    ap.add_argument("--lang-dir", default=None,
                    help="default: data/lang_bpe_<vocab-size>")
    args = ap.parse_args()

    import sentencepiece as spm
    from lhotse import CutSet

    lang = pathlib.Path(args.lang_dir or f"data/lang_bpe_{args.vocab_size}")
    lang.mkdir(parents=True, exist_ok=True)
    txt = lang / "transcript_words.txt"

    n = 0
    with open(txt, "w", encoding="utf-8") as w:
        for cut in CutSet.from_file(args.train_cuts):
            for sup in cut.supervisions:
                t = (sup.text or "").strip()
                if t:
                    w.write(t + "\n")
                    n += 1
    print(f"wrote {n:,} transcript lines -> {txt}")

    # icefall control-symbol convention: 0=<blk>, 1=<sos/eos>, 2=<unk>
    user_defined_symbols = ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)
    prefix = lang / "bpe"
    print(f"training {args.model_type} vocab={args.vocab_size} -> {prefix}.model")
    spm.SentencePieceTrainer.train(
        input=str(txt),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        model_prefix=str(prefix),
        character_coverage=1.0,
        input_sentence_size=100000000,
        shuffle_input_sentence=True,
        user_defined_symbols=user_defined_symbols,
        unk_id=unk_id,
        bos_id=-1,
        eos_id=-1,
    )

    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    tokens = lang / "tokens.txt"
    with open(tokens, "w", encoding="utf-8") as w:
        for i in range(sp.vocab_size()):
            w.write(f"{sp.id_to_piece(i)} {i}\n")
    print(f"wrote {sp.vocab_size()} tokens -> {tokens}")
    # quick sanity: round-trip a Bulgarian sample
    sample = "министерският съвет прие решение"
    print("  sample:", sample)
    print("  pieces:", " ".join(sp.encode(sample, out_type=str)))


if __name__ == "__main__":
    main()
