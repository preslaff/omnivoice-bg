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
# TAG names a corpus variant so builds never clobber each other: "" = v1 (parliament +
# AB400 Kaldi text), "_gp" = the GP corpus (all axes, from the GP train cut manifest).
TAG=${TAG:-}
TEXT="$OUT/lm_text${TAG}.txt"
ARPA="$OUT/bg_word_${ORDER}gram${TAG}.arpa"
BIN="$OUT/bg_word_${ORDER}gram${TAG}.bin"
# PREP_ARGS are passed to prep_lm_text.py. Set PREP=0 to reuse an existing $TEXT
# (e.g. one already written by a bespoke prep_lm_text.py invocation).
PREP=${PREP:-1}
PREP_ARGS=${PREP_ARGS:-}

[ -x "$KENLM_BIN/lmplz" ] || { echo "ERROR: lmplz not at $KENLM_BIN (set KENLM_BIN)"; exit 1; }
mkdir -p "$OUT"

# --- stage 0: rebuild the LM text from the CURRENT split (leakage guard) ---
if [ "$PREP" = "1" ]; then
  echo "=== stage 0: regenerate dev/test-excluded LM text ==="
  "$PY" prep_lm_text.py --out "$TEXT" $PREP_ARGS
else
  echo "=== stage 0: PREP=0 — reusing existing $TEXT ==="
  [ -f "$TEXT" ] || { echo "ERROR: $TEXT missing"; exit 1; }
fi
lines=$(wc -l < "$TEXT"); words=$(wc -w < "$TEXT")
printf "  lm_text: %'d lines, %'d words\n" "$lines" "$words"

# Leakage assertion. The real guarantee is the id-based exclusion in prep_lm_text.py
# (drops every utterance whose SESSION is held out); these are belt-and-suspenders checks
# on top of it, and which one is valid depends on where the text came from:
#
#   LEAK_CHECK=phrase (v1, Kaldi --dirs)  A line byte-identical to a held-out utterance is
#     NOT leakage if that same phrase is also spoken in a TRAIN session ("има ли реплики").
#     Real leakage = content UNIQUE to held-out sessions. We fail only on that.
#   LEAK_CHECK=ids (GP, --cuts)  The phrase test does not apply: its "train" reference is
#     the three Kaldi dirs, which cover only part of the GP corpus, so text from a recovered
#     session or another axis (YouTube/BNR) reads as holdout-unique when it is nothing of the
#     sort. Cut manifests carry ids, so we assert the real property directly: zero cut-id and
#     zero recording-id intersection between the LM source and every eval manifest.
LEAK_CHECK=${LEAK_CHECK:-phrase}
if [ "$LEAK_CHECK" = "ids" ]; then
  echo "  verifying no eval cut/recording id enters the LM source ($LEAK_SRC)…"
  "$PY" - "$LEAK_SRC" $LEAK_EVAL <<'PY'
import sys, gzip, json
def ids(p):
    cuts, recs = set(), set()
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cuts.add(c["id"])
            for s in c.get("supervisions", []):
                recs.add(s.get("recording_id", ""))
    return cuts, recs
src_cuts, src_recs = ids(sys.argv[1])
bad = 0
for p in sys.argv[2:]:
    e_cuts, e_recs = ids(p)
    ic, ir = src_cuts & e_cuts, src_recs & e_recs
    print(f"    {p.split('/')[-1]:28s} cut-id overlap {len(ic)}  recording-id overlap {len(ir)}")
    bad += len(ic) + len(ir)
if bad:
    print("    LEAK: eval material is inside the LM source."); sys.exit(2)
print(f"    OK: {len(src_cuts):,} source cuts / {len(src_recs):,} recordings, zero eval overlap.")
PY
else
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
fi

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
