#!/usr/bin/env bash
# Copyright 2026 OmniVoice-BG (Apache-2.0)
#
# Build the Bulgarian ASR word-level n-gram (KenLM), LEAKAGE-SAFE: the LM text is
# regenerated fresh from the CURRENT frozen split so dev/test transcripts are excluded.
#
# WHY REBUILD: the pre-existing data/lm/bg_3gram.* was built 2026-07-31 00:37, ~54 min
# BEFORE the dev/test split was frozen (01:31) — so it may contain held-out transcripts
# and would inflate WER. This always rebuilds lm_text.txt first (prep_lm_text.py excludes
# data/lhotse/bg/split/{dev,test}_recordings.txt), so the n-gram can never see the holdout.
#
# Output (data/lm/):
#   lm_text.txt              normalized, dev/test-excluded training text (rebuilt)
#   bg_word_${ORDER}gram.arpa  ARPA n-gram
#   bg_word_${ORDER}gram.bin   KenLM binary (fast load)
#
# The word ARPA is the shared source for BOTH downstream consumers:
#   * Kaldi TDNN-F chain baseline  -> arpa2fst -> G.fst   (v1 plan step 4)
#   * transducer fast_beam_search_nbest_LG -> G.fst |> compose L -> LG.pt  (compile_lg, next)
# (The neural-LM shallow-fusion / LODR path is separate — it needs an RNNLM, not this.)
#
# Usage:  bash build_kenlm_bg.sh                 # 3-gram, pruned
#         ORDER=4 PRUNE="0 0 1 1" bash build_kenlm_bg.sh
set -euo pipefail

repo=$(cd "$(dirname "$0")" && pwd)
cd "$repo"

PY=${PY:-$repo/venv/bin/python}
KENLM_BIN=${KENLM_BIN:-$HOME/kenlm/build/bin}
ORDER=${ORDER:-3}
PRUNE=${PRUNE:-0 0 1}                    # no prune on uni/bi, prune singletons at 3+ (size)
OUT=${OUT:-data/lm}
TEXT="$OUT/lm_text.txt"
ARPA="$OUT/bg_word_${ORDER}gram.arpa"
BIN="$OUT/bg_word_${ORDER}gram.bin"

[ -x "$KENLM_BIN/lmplz" ] || { echo "ERROR: lmplz not at $KENLM_BIN (set KENLM_BIN)"; exit 1; }
mkdir -p "$OUT"

# --- stage 0: rebuild the LM text from the CURRENT split (leakage guard) ---
echo "=== stage 0: regenerate dev/test-excluded LM text ==="
"$PY" prep_lm_text.py --out "$TEXT"
lines=$(wc -l < "$TEXT"); words=$(wc -w < "$TEXT")
printf "  lm_text: %'d lines, %'d words\n" "$lines" "$words"

# Leakage assertion. The real guarantee is the id-based exclusion in prep_lm_text.py
# (drops every utterance whose SESSION is held out). This is a belt-and-suspenders check
# on top of it. NOTE: a line byte-identical to a held-out utterance is NOT leakage if that
# same phrase is also spoken in a TRAIN session (e.g. "има ли реплики") — its presence in
# the LM comes from the train occurrence, which is correct. Real leakage = content UNIQUE
# to held-out sessions appearing in lm_text. We fail only on that.
echo "  verifying no held-out-UNIQUE transcript leaked into lm_text…"
"$PY" - "$TEXT" <<'PY'
import sys, pathlib
ACUTE="́"
def norm(s): return " ".join(t for t in s.replace(ACUTE,"").split() if t!="<unk>")
split=pathlib.Path("data/lhotse/bg/split")
held=set()
for f in ("dev_recordings.txt","test_recordings.txt"):
    p=split/f
    if p.exists(): held|=set(p.read_text().split())
held_lines, train_lines = set(), set()
for d in ("exp/kaldi_pilot/data/train_cleaned","exp/kaldi_pilot/data/parl_new_cleaned",
          "exp/kaldi_pilot/data/parl_new2_cleaned"):
    t=pathlib.Path(d)/"text"
    if not t.exists(): continue
    for line in t.read_text(encoding="utf-8").splitlines():
        p=line.split(None,1)
        if len(p)<2: continue
        rec=p[0].split("-",1)[0]; n=norm(p[1])
        if not n: continue
        (held_lines if rec in held else train_lines).add(n)
held_only = held_lines - train_lines          # content unique to the held-out sessions
corpus = set(open(sys.argv[1],encoding="utf-8").read().splitlines())
common = (held_lines & train_lines) & corpus  # phrases shared with train — expected, OK
leak = held_only & corpus                     # unique held-out content in the LM — a leak
print(f"    held-out lines {len(held_lines):,} "
      f"({len(held_only):,} unique to holdout) | "
      f"shared-phrase overlap {len(common):,} (expected) | UNIQUE leak {len(leak)}")
if leak:
    print("    LEAK EXAMPLES:", list(leak)[:3]); sys.exit(2)
print("    OK: zero held-out-unique content in lm_text.")
PY

# --- stage 1: estimate the n-gram ---
echo "=== stage 1: lmplz -o $ORDER (prune: $PRUNE) ==="
"$KENLM_BIN/lmplz" -o "$ORDER" --prune $PRUNE --text "$TEXT" --arpa "$ARPA"

# --- stage 2: binarize ---
echo "=== stage 2: build_binary ==="
"$KENLM_BIN/build_binary" trie "$ARPA" "$BIN"

echo
echo "vocab (types in lm_text): $(tr ' ' '\n' < "$TEXT" | sort -u | wc -l)"
echo "-> $ARPA"
echo "-> $BIN"
echo
echo "NEXT to actually use it on the transducer (n-gram-only, no neural LM):"
echo "  compile LG.pt for fast_beam_search_nbest_LG:"
echo "    build BPE lexicon L_disambig.fst (icefall local/prepare_lang_bpe.py),"
echo "    kaldilm $ARPA -> G.fst, compose L|>G -> data/lang_bpe_500/LG.pt,"
echo "    then decode.py --eval-set dev --decoding-method fast_beam_search_nbest_LG"
echo "    --ngram-lm-scale <tune on dev>."
