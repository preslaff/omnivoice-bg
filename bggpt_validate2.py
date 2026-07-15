#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""BgGPT gate, reformulated: word-sense/grammar disambiguation instead of
stress notation. Hypothesis: v1's 15/28 fail was notational (text never
encodes stress, so U+0301 maps to nothing the LLM knows), while the
underlying sense/POS decision is text-native. Each item asks for (а)/(б);
the marked form is OUR lexicon mapping of the chosen sense, not BgGPT's.

Items = exactly the v1 misses. Usage: ./venv/bin/python bggpt_validate2.py
"""
import argparse
import json
import urllib.request

# (sentence, word, option_a, option_b, correct_letter, marked_if_correct)
ITEMS = [
    ("Говори по-високо, защото не те чувам!", "говори",
     "сегашно време, 3 л. ед.ч. (той говори)", "повелително наклонение (заповед)", "б", "говори́"),
    ("От тенджерата се вдигаше гъста пара.", "пара",
     "водна пара, изпарение", "монета, дребни пари", "а", "па́ра"),
    ("Хайдушката чета премина тайно през планината.", "чета",
     "въоръжена група, дружина (съществително)", "глаголът 'чета' (чета книга)", "а", "че́та"),
    ("Пуловерът е изплетен от мека овча вълна.", "вълна",
     "материал от овца, прежда", "морска вълна, вълнение", "а", "въ́лна"),
    ("Детето остана само вкъщи цяла вечер.", "само",
     "наречие 'единствено, просто'", "самичко, без придружител", "б", "само́"),
    ("Носи багажа до колата, моля те.", "носи",
     "сегашно време, 3 л. ед.ч. (тя носи)", "повелително наклонение (заповед)", "б", "носи́"),
    ("Прави каквото ти казвам и няма да съжаляваш.", "прави",
     "сегашно време, 3 л. ед.ч. (той прави)", "повелително наклонение (заповед)", "б", "прави́"),
    ("Вчера тя купи хляб и мляко от магазина.", "купи",
     "минало свършено време (аорист)", "повелително наклонение (заповед)", "а", "ку́пи"),
    ("Синът им учи медицина в София.", "учи",
     "сегашно време, 3 л. ед.ч. (той учи)", "повелително наклонение (заповед)", "а", "у́чи"),
    ("Ходи по-бързо, че ще закъснеем!", "ходи",
     "сегашно време, 3 л. ед.ч. (тя ходи)", "повелително наклонение (заповед)", "б", "ходи́"),
    ("Той моли за прошка всеки път, когато сгреши.", "моли",
     "сегашно време, 3 л. ед.ч. (той моли)", "повелително наклонение (заповед)", "а", "мо́ли"),
    ("Тя реши дългата си коса пред огледалото.", "коса",
     "коса на главата, косъм", "наклонена, полегата (прилагателно)", "а", "коса́"),
]

SYSTEM = (
    "Ти си експерт по българска граматика и лексикология. Дадено е "
    "изречение и дума от него с две възможни тълкувания. Определи кое "
    "тълкуване е вярно В ТОВА изречение. Отговори САМО с буквата: а или б."
)


def ask(url, sentence, word, opt_a, opt_b):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Изречение: {sentence}\nДума: „{word}“\n"
            f"а) {opt_a}\nб) {opt_b}"},
    ]
    body = json.dumps({"messages": messages, "temperature": 0,
                       "max_tokens": 6}).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"].strip().strip(".)（(").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8080")
    args = ap.parse_args()

    n_ok = 0
    print(f"{'word':<10} {'correct':<8} {'BgGPT':<8} -> marked form")
    for sentence, word, a, b, correct, marked in ITEMS:
        got = ask(args.url, sentence, word, a, b)[:1]
        ok = got == correct
        n_ok += ok
        print(f"{word:<10} {correct:<8} {got:<8} "
              f"{'OK  -> ' + marked if ok else 'MISS [' + sentence[:40] + ']'}",
              flush=True)
    print(f"\nsense-disambiguation: {n_ok}/{len(ITEMS)} "
          f"(v1 stress-notation got 0/{len(ITEMS)} on these same items)")


if __name__ == "__main__":
    main()
