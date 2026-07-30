#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Build the "starved-reading" text label set for the Accentor fine-tune.

The ft7 head-to-head (2026-07-29) showed the homograph battery fails almost
exclusively on MINORITY readings that declarative parliament/chitanka speech
starves:
  * verb IMPERATIVES  — говори́/моли́/носи́/купи́/учи́/ходи́ (final stress) vs the
    abundant 3sg present (root stress) the corpus is saturated with;
  * a handful of LEXICAL homographs whose rare sense almost never occurs —
    че́та (noun 'detachment') vs чета́ (verb 'I read'); ко́са (adj 'slanted') vs
    коса́ (hair); само́ (neuter 'alone') vs са́мо (particle 'only'); ку́пи
    (aorist 'bought') vs купи́ (imperative).

Corpus data can't fix these — the reading isn't in it. So we supply explicit
TEXT labels. Each row is a ONE-TOKEN teaching signal in train_accentor's
marked.jsonl schema: the target word is marked (supervised); every other word is
masked (excluded from loss), so we never inject a wrong label on the context.

To avoid the inversion trap (fine-tune #2: piling on the minority flips the
majority), every class is BALANCED — the starved reading AND a few majority-
reading rows of the same word, so the model learns the boundary, not a new bias.

Sources:
  * imperatives  — exp/imperatives/bank.jsonl (native-vetted verb×frame marks),
    supervising the imperative verb, masking the frame. Generalizes the rule
    across all 164 verbs, not just the 8 battery ones.
  * 3sg balance + lexical homographs — the curated tables below (hand-authored,
    native-reviewable, deliberately DISJOINT from the eval battery sentences).

Fold into the ft7b build via --extra-marked:
    ./venv/bin/python build_starved_readings.py --out data/kaldi/stress_labels_starved.jsonl
    ./venv/bin/python build_accentor_finetune.py ... \
        --extra-marked data/kaldi/stress_labels_starved.jsonl \
        --out data/accentor_finetune_ft7b.jsonl
"""
import argparse
import json
import pathlib

from select_chitanka_sentences import WORD_RE

ACUTE = "́"


def mark_surface(orig, marked_lc):
    """Insert the acute into `orig` at the position it sits in `marked_lc`
    (same helper as mine_homographs.apply_mark)."""
    i = marked_lc.find(ACUTE)
    if i <= 0 or i - 1 > len(orig):
        return orig
    return orig[:i] + ACUTE + orig[i:]


def one_token_row(rid, sentence, target_lc, marked_lc):
    """Supervise ONLY the target word (labeled with marked_lc); mask every other
    word. Returns a marked.jsonl row, or None if the target isn't found."""
    sentence = sentence.replace(ACUTE, "")   # text must be plain; mark comes from marked_lc
    low = sentence.lower()
    spans = [m.span() for m in WORD_RE.finditer(low)]
    tgt = next(((s, e) for (s, e) in spans if low[s:e] == target_lc), None)
    if tgt is None:
        return None
    pieces, mask, pos = [], [], 0
    for (s, e) in spans:
        pieces.append(sentence[pos:s])
        if (s, e) == tgt:
            pieces.append(mark_surface(sentence[s:e], marked_lc))
        else:
            pieces.append(sentence[s:e])
            mask.append([s, e])
        pos = e
    pieces.append(sentence[pos:])
    return {"id": rid, "text": sentence, "marked": "".join(pieces),
            "mask_words": mask, "n_words": len(spans), "n_marked": 1}


# --- 3sg (root) balance for the imperative verbs -----------------------------
# Natural declarative frames with a subject / subordinator cue that forces the
# 3sg present reading. {v} is filled with the verb's ROOT-stress form. These keep
# the majority reading supervised so the imperative rows below don't invert it.
SG3_FRAMES = [
    "Той често {v} по същия начин.",
    "Знаем, че тя {v} внимателно всеки път.",
    "Всеки ден човекът {v} без да бърза.",
    "Когато {v}, гласът му е спокоен.",
    "Тя рядко {v} на публично място.",
    "Съседът ни {v} тихо, за да не пречи.",
]
# ROOT-stress forms per verb (from config/bg/homograph_verbs.json 'sg3').
SG3_VERBS = {
    "говори": "гово́ри", "моли": "мо́ли", "носи": "но́си",
    "учи": "у́чи", "ходи": "хо́ди", "прави": "пра́ви",
}

# --- Imperative verbs MISSING from the bank (not in imperative_verbs.txt) -----
# говори/моли are battery imperatives but absent from exp/imperatives/bank.jsonl,
# so they get no imperative supervision from source 1. Curated here with clear
# imperative cues (particle before: хайде/моля те/недей/ела), disjoint from the
# battery. word -> (imperative mark, [PLAIN sentences]).
IMPERATIVE_EXTRA = {
    "говори": ("говори́", [
        "Хайде, говори по-бавно и ясно.",
        "Моля те, говори по-открито с мен.",
        "Недей да мълчиш, говори най-после.",
        "Ела и говори с мен за проблема.",
        "Говори с него, преди да е станало късно.",
        # ft7d: comparative "говори по-X" contexts (battery-v1 failure shape).
        "Говори по-високо, за да те чуят и отзад.",
        "Говори по-тихо, детето спи в съседната стая.",
        "Говори по-ясно, думите ти се преплитат.",
        "Говори по-смело, не се притеснявай от никого.",
        "Говори направо, без излишни увъртания.",
    ]),
    "моли": ("моли́", [
        "Хайде, моли за втори шанс.",
        "Недей да се предаваш, моли за помощ.",
        "Ако имаш нужда, моли без свян.",
        "Обърни се и моли за прошка.",
        "Моли се за тях, щом наистина вярваш.",
    ]),
    # носи/учи/ходи: chitanka yields only 7-13 natural imperatives each (F3
    # starvation, confirmed by mine_homographs 2026-07-30), so brute mining can't
    # teach them. Curated diverse imperative contexts (particle-cued), disjoint
    # from the battery, to match говори's natural coverage.
    "носи": ("носи́", [
        "Хайде, носи кашоните внимателно по стълбите.",
        "Моля те, носи чадър, навън вали.",
        "Недей да мързелуваш, носи си сам раницата.",
        "Носи топли дрехи, планината е студена.",
        "Винаги носи документите си със себе си.",
        "Ела и носи този сандък до склада.",
        "Носи знамето гордо пред колоната.",
        "Стига се оплаква, носи каквото ти възложиха.",
        # ft7d: ABSTRACT-object imperatives (ad/slogan register — battery-v2 fail).
        "Носи усмивка навсякъде, където отиваш.",
        "Носи радост на хората около теб.",
        "Носи гордо българското име по света.",
        "Носи надежда дори в най-тъмните дни.",
        "Носи доброто в сърцето си винаги.",
        "Носи увереност във всяка своя стъпка.",
    ]),
    "учи": ("учи́", [
        "Хайде, учи усърдно за предстоящия изпит.",
        "Моля те, учи редовно, за да напреднеш.",
        "Недей да отлагаш, учи отрано.",
        "Учи от грешките си и продължавай напред.",
        "Учи търпеливо, знанието идва с времето.",
        "Стига игри, седни и учи си уроците.",
        "Учи чужди езици, ще ти помогнат в живота.",
        "Учи прилежно и ще постигнеш целите си.",
    ]),
    "ходи": ("ходи́", [
        "Хайде, ходи по-живо, автобусът ей сега тръгва.",
        "Моля те, ходи внимателно по хлъзгавия път.",
        "Недей да тичаш, ходи спокойно.",
        "Ходи всеки ден пеша, полезно е за здравето.",
        "Ходи изправен, не се прегърбвай.",
        "Ходи след мен и не се отделяй от групата.",
        "Стига седя, ходи да се раздвижиш малко.",
        "Ходи на тренировки редовно, не пропускай.",
    ]),
    # прави/купи passed battery v1 but were context-sensitive on v2 (прави in
    # abstract-object ad contexts, купи in the concert/"още днес" context). ft7d
    # adds those exact context shapes. прави is imperfective (прави́/пра́ви);
    # sentences avoid a second прави token so one-token supervision stays clean.
    "прави": ("прави́", [
        "Прави добро и не очаквай нищо в замяна.",
        "Прави разликата с малки ежедневни стъпки.",
        "Прави избора си смело и без колебание.",
        "Хайде, прави нещо, не стой със скръстени ръце.",
        "Прави планове, но бъди готов да ги промениш.",
        "Прави компромиси, когато мъдростта го налага.",
        "Недей да отлагаш, прави го веднага.",
        "Прави света по-добър със своите постъпки.",
    ]),
    "купи": ("купи́", [
        "Купи билета за концерта, преди да са свършили.",
        "Купи си малко време за почивка този уикенд.",
        "Хайде, купи най-после онази книга, която искаше.",
        "Купи качество, а не количество.",
        "Купи местни продукти и подкрепи фермерите.",
        "Не се колебай, купи подаръка още сега.",
    ]),
}

# --- Lexical homographs: word -> [(PLAIN sentence, marked target surface), ...]
# Sentences are PLAIN (no acute); the mark comes only from the 2nd element. The
# target may be inflected (че́тата) — target surface is derived by stripping the
# acute from the marked form. Both senses per word (starved minority FIRST, then
# majority balance). Hand-authored, native-reviewable, DISJOINT from the battery.
LEXICAL = {
    "чета": [
        # noun че́та (detachment) — passes in ft7b; keep a few to hold it.
        ("Въстаническата чета се укри дълбоко в гората.", "че́та"),
        ("Легендарна чета прекоси реката посред нощ.", "че́та"),
        ("Всяка бунтовна чета имаше свой войвода.", "че́та"),
        ("Малка партизанска чета нападна складовете.", "че́та"),
        # verb чета́ (I read) — FAILED in ft7b (over-corrected to че́та); rebalance
        # with MORE diverse 1sg contexts.
        ("Обичам да чета на глас, когато съм сам.", "чета́"),
        ("Сутрин чета вестника с чаша кафе.", "чета́"),
        ("Аз чета бавно, но запомням всичко.", "чета́"),
        ("Докато пътувам с влака, чета по някоя книга.", "чета́"),
        ("Преди сън винаги чета поне няколко страници.", "чета́"),
        ("Не ме прекъсвай, чета важен документ.", "чета́"),
        ("Аз чета между редовете на това писмо.", "чета́"),
        ("Когато чета поезия, забравям за времето.", "чета́"),
        ("Рядко чета вестници, предпочитам романи.", "чета́"),
        ("В момента чета биография на композитор.", "чета́"),
    ],
    "коса": [
        # adj ко́са (slanted) — FAILED in ft7b; more diverse contexts.
        ("Между двете имена сложи коса черта.", "ко́са"),
        ("Начертай една коса линия през квадрата.", "ко́са"),
        ("В този шрифт буквите имат лека коса чертичка.", "ко́са"),
        ("Покривът има стръмна коса повърхност.", "ко́са"),
        ("На картата границата е обозначена с коса линия.", "ко́са"),
        ("В таблицата раздели стойностите с коса черта.", "ко́са"),
        ("Начертай коса ос през центъра на фигурата.", "ко́са"),
        ("Шевът минава по коса линия през плата.", "ко́са"),
        ("Стълбата стоеше под коса наклон към стената.", "ко́са"),
        ("Сенките падаха под коса посока следобед.", "ко́са"),
        # noun коса́ (hair) — majority, keep a few.
        ("Тя прибра дългата коса на опашка.", "коса́"),
        ("Дългата му коса стигаше до раменете.", "коса́"),
        ("Вятърът развяваше нейната коса край морето.", "коса́"),
    ],
    "само": [
        # neuter само́ (alone) — FAILED in ft7b; more diverse contexts.
        ("Момчето се почувства само сред непознатите.", "само́"),
        ("Кучето остана само на двора цяла нощ.", "само́"),
        ("Детето не бива да пътува само толкова далеч.", "само́"),
        ("Селото стоеше само насред полето.", "само́"),
        ("Малкото коте остана само в големия апартамент.", "само́"),
        ("Не оставяй бебето само близо до водата.", "само́"),
        ("Дървото растеше само на билото на хълма.", "само́"),
        ("Момичето се прибра само по тъмно.", "само́"),
        ("Старецът живееше само в къщата на края на селото.", "само́"),
        ("Детето се научи да се облича само.", "само́"),
        # particle са́мо (only) — majority, keep a few.
        ("Искам само минута от вниманието ти.", "са́мо"),
        ("Това е само началото на дълъг път.", "са́мо"),
        ("Останаха само трима души в залата.", "са́мо"),
        ("Трябва ни само още малко търпение.", "са́мо"),
    ],
    "купи": [
        ("Вчера баща ми купи домати от пазара.", "ку́пи"),
        ("Той купи нова кола миналата седмица.", "ку́пи"),
        ("Кой купи всичките билети преди концерта?", "ку́пи"),
        ("Съседката купи хляб и се прибра.", "ку́пи"),
        ("Моля те, купи малко плодове на връщане.", "купи́"),
        ("Купи си топли дрехи, навън застудява.", "купи́"),
        ("Ако минаваш покрай аптеката, купи лекарствата.", "купи́"),
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default="exp/imperatives/bank.jsonl",
                    help="verb×frame imperative sentences (plain + marked verb)")
    ap.add_argument("--out", default="data/kaldi/stress_labels_starved.jsonl")
    ap.add_argument("--cap-per-verb", type=int, default=40,
                    help="max imperative rows per verb from the bank")
    ap.add_argument("--repeat-imperative", type=int, default=8,
                    help="duplicate each imperative row N times. --extra-marked is "
                         "NOT touched by the tier-3 oversampler, so density must be "
                         "baked in; ~0.1%% of the corpus is too dilute to move the "
                         "battery. TUNE: if imperatives still fail, raise it.")
    ap.add_argument("--repeat-lexical", type=int, default=6,
                    help="duplicate each lexical/3sg row N times (only ~60 unique "
                         "sentences; keeps the rare readings visible without "
                         "drowning them — prefer MORE sentences over more repeats)")
    ap.add_argument("--homograph-only", action="store_true",
                    help="emit bank imperatives ONLY for homograph verbs "
                         "(носи/учи/ходи/…). The ~156 non-homograph imperatives are "
                         "unambiguous — the lexicon already marks them — so they only "
                         "add a global final-stress bias (ft7b regressed въ́лна and "
                         "doubled OOD SER). Turn on for ft7c.")
    ap.add_argument("--homograph-verbs", default="config/bg/homograph_verbs.json",
                    help="verb set for --homograph-only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, stats = [], {}
    seen_per_verb = {}

    homo_set = set()
    if args.homograph_only:
        homo_set = set(json.load(open(args.homograph_verbs, encoding="utf-8"))["homographs"])
        print(f"homograph-only: bank imperatives restricted to {sorted(homo_set)}")

    # 1) Imperatives from the bank (one-token: verb supervised, frame masked).
    bank_path = pathlib.Path(args.bank)
    if bank_path.exists():
        for line in bank_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            b = json.loads(line)
            verb, vm = b["verb"], b["verb_marked"]
            if args.homograph_only and verb not in homo_set:
                continue
            if seen_per_verb.get(verb, 0) >= args.cap_per_verb:
                continue
            r = one_token_row(f"imp_{b['id']}", b["plain"], verb, vm)
            if r:
                for k in range(args.repeat_imperative):
                    rows.append({**r, "id": f"{r['id']}_r{k}"})
                seen_per_verb[verb] = seen_per_verb.get(verb, 0) + 1
                stats["imperative"] = stats.get("imperative", 0) + 1
    else:
        print(f"WARN: {bank_path} not found — imperative bulk skipped")

    # 1b) Imperative verbs absent from the bank (говори/моли) — curated.
    for verb, (vm, sents) in IMPERATIVE_EXTRA.items():
        for i, sent in enumerate(sents):
            r = one_token_row(f"impx_{verb}_{i}", sent, verb, vm)
            if r is None:
                print(f"WARN: {verb!r} not found in: {sent}")
                continue
            for k in range(args.repeat_imperative):
                rows.append({**r, "id": f"{r['id']}_r{k}"})
            stats["imperative"] = stats.get("imperative", 0) + 1
            seen_per_verb[verb] = seen_per_verb.get(verb, 0) + 1

    # 2) 3sg (root) balance for the imperative verbs.
    for verb, sg3 in SG3_VERBS.items():
        for i, frame in enumerate(SG3_FRAMES):
            sent = frame.format(v=sg3.replace(ACUTE, ""))
            r = one_token_row(f"sg3_{verb}_{i}", sent, verb, sg3)
            if r:
                for k in range(args.repeat_lexical):
                    rows.append({**r, "id": f"{r['id']}_r{k}"})
                stats["sg3_balance"] = stats.get("sg3_balance", 0) + 1

    # 3) Lexical homographs (both senses, balanced). Target surface is the marked
    #    form with the acute stripped (so inflected forms like ко́сата still match).
    for word, examples in LEXICAL.items():
        for i, (sent, marked) in enumerate(examples):
            target = marked.replace(ACUTE, "")
            r = one_token_row(f"lex_{word}_{i}", sent, target, marked)
            if r is None:
                print(f"WARN: target {word!r} not found in: {sent}")
                continue
            for k in range(args.repeat_lexical):
                rows.append({**r, "id": f"{r['id']}_r{k}"})
            stats["lexical"] = stats.get("lexical", 0) + 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows):,} rows -> {out}")
    print(f"  imperative (bank)   : {stats.get('imperative', 0):,} rows "
          f"({len(seen_per_verb)} verbs)")
    print(f"  3sg balance         : {stats.get('sg3_balance', 0)} distinct "
          f"x{args.repeat_lexical}")
    print(f"  lexical homographs  : {stats.get('lexical', 0)} distinct "
          f"x{args.repeat_lexical}  (чета/коса/само/купи, both senses)")
    print("\nsample rows:")
    for r in (rows[:2] + rows[-6:]):
        print(f"  {r['marked']}")


if __name__ == "__main__":
    main()
