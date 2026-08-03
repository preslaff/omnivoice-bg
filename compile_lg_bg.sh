#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Compile LG.pt for the BG Zipformer's `fast_beam_search_nbest_LG` decoding — the
# n-gram-only LM path (no neural LM). Turns the leakage-safe word n-gram from
# build_kenlm_bg.sh into a decoding graph:
#
#   words.txt (LM vocab)              # symbol table covering every ARPA word
#     + bpe.model  --prepare_lang_bpe--> L_disambig.pt   (word -> BPE-token lexicon FST)
#   bg_word_Ngram.arpa --kaldilm-----> data/lm/G_N_gram.fst.txt
#   L_disambig.pt |> G  --compile_lg--> data/lang_bpe_500/LG.pt
#
# Then: decode.py --eval-set dev --decoding-method fast_beam_search_nbest_LG
#       --ngram-lm-scale <tune on dev>.
#
# Reuses icefall's librispeech local/ scripts unchanged; writes only into the (symlinked)
# data/lang_bpe_500 + data/lm. The trained model + aligner are untouched.
#
# PREREQUISITE:  pip install kaldilm     (ARPA -> G.fst; not yet in the venv)
# Usage:  bash compile_lg_bg.sh                 # 3-gram
#         NGRAM=4 bash compile_lg_bg.sh         # if you built a 4-gram
set -euo pipefail

ICEFALL=${ICEFALL:-$HOME/icefall}
REC="$ICEFALL/egs/bg_parliament/ASR"
LOCAL="$ICEFALL/egs/librispeech/ASR/local"
OMNI=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$OMNI/venv/bin/python}
NGRAM=${NGRAM:-3}
LANG=data/lang_bpe_500
LM="G_${NGRAM}_gram"
ARPA="data/lm/bg_word_${NGRAM}gram.arpa"

cd "$REC"
export PYTHONPATH="$ICEFALL:${PYTHONPATH:-}"
ln -sfn "$LOCAL" local                       # icefall helpers (prepare_lang_bpe, compile_lg)
ln -sfn "$OMNI/data/lm" data/lm              # arpa in, G/LG out (alongside data/fbank, lang_bpe_500)

# --- prerequisites ---
[ -f "$ARPA" ] || { echo "ERROR: $ARPA missing — run build_kenlm_bg.sh (NGRAM=$NGRAM) first"; exit 1; }
[ -f "$LANG/bpe.model" ] || { echo "ERROR: $LANG/bpe.model missing"; exit 1; }
"$PY" -c "import kaldilm" 2>/dev/null || {
  echo "ERROR: kaldilm not installed (needed for ARPA -> G.fst). Run:"
  echo "    $PY -m pip install kaldilm"; exit 1; }

# --- stage 0: words.txt = symbol table over the LM vocabulary (guarantees G coverage) ---
if [ ! -f "$LANG/words.txt" ]; then
  echo "=== stage 0: build $LANG/words.txt from the LM vocab ==="
  "$PY" - "$LANG" data/lm/lm_text.txt "$LANG/transcript_words.txt" <<'PY'
import sys, pathlib
lang, lm_text, tw = sys.argv[1:4]
vocab = set()
with open(lm_text, encoding="utf-8") as f:
    for line in f:
        vocab.update(line.split())               # every word type the n-gram can emit
p = pathlib.Path(tw)                              # BPE-training words too (belt-and-suspenders)
if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.split():
            vocab.add(line.split()[0])
special = {"<eps>", "<UNK>", "#0", "<s>", "</s>", ""}
words = sorted(w for w in vocab if w not in special)
out = pathlib.Path(lang) / "words.txt"
with out.open("w", encoding="utf-8") as w:
    i = 0
    for sym in ("<eps>", "<UNK>"):               # 0=<eps>, 1=<UNK> (oov lexicon entry)
        w.write(f"{sym} {i}\n"); i += 1
    for wd in words:
        w.write(f"{wd} {i}\n"); i += 1
    for sym in ("#0", "<s>", "</s>"):            # disambig + LM boundary symbols last
        w.write(f"{sym} {i}\n"); i += 1
print(f"  words.txt: {i} symbols -> {out}")
PY
else
  echo "=== stage 0: $LANG/words.txt exists ($(wc -l <"$LANG/words.txt") symbols) — reuse ==="
fi

# --- stage 1: BPE lexicon -> L_disambig.pt ---
if [ ! -f "$LANG/L_disambig.pt" ]; then
  echo "=== stage 1: prepare_lang_bpe -> L_disambig.pt ==="
  "$PY" local/prepare_lang_bpe.py --lang-dir "$LANG"
else
  echo "=== stage 1: $LANG/L_disambig.pt exists — reuse ==="
fi

# --- stage 2: ARPA -> G.fst.txt (word-level, disambig #0) ---
if [ ! -f "data/lm/${LM}.fst.txt" ]; then
  echo "=== stage 2: kaldilm -> data/lm/${LM}.fst.txt ==="
  "$PY" -m kaldilm --read-symbol-table="$LANG/words.txt" \
      --disambig-symbol='#0' --max-order="$NGRAM" "$ARPA" > "data/lm/${LM}.fst.txt"
else
  echo "=== stage 2: data/lm/${LM}.fst.txt exists — reuse ==="
fi

# --- stage 3: compose L|>G, determinize -> LG.pt ---
echo "=== stage 3: compile_lg -> $LANG/LG.pt ==="
rm -f "$LANG/LG.pt"                              # compile_lg skips if present; force fresh
"$PY" local/compile_lg.py --lang-dir "$LANG" --lm "$LM"

echo
echo "-> $LANG/LG.pt ready.  Tune ngram-lm-scale on DEV:"
echo "   for s in 0.01 0.05 0.1 0.2 0.3; do"
echo "     $PY zipformer/decode.py --epoch 30 --avg 5 --use-averaged-model 1 \\"
echo "       --eval-set dev --decoding-method fast_beam_search_nbest_LG \\"
echo "       --ngram-lm-scale \$s --exp-dir zipformer/exp \\"
echo "       --bpe-model $LANG/bpe.model --max-duration 600 ; done"
echo "   pick the best-WER scale, then run it ONCE on --eval-set test."
