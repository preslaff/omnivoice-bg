#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Validation gate for BgGPT as homograph teacher/arbiter (plan §P2.5).

Few-shots BgGPT (llama-server, OpenAI-compatible, localhost:8080) on the
border-set homographs + the sentences where the bon experiment exposed
Accentor errors. Auto-scores the clear-cut items; prints the disputed ones
(tier-2 weak words, ко́са черта) for native review. Adopt the
teacher/arbiter design only if the confident set is (near-)perfect.

Usage:  ./venv/bin/python bggpt_validate.py [--url http://localhost:8080]
"""
import argparse
import json
import urllib.request

FEWSHOT = [
    ("Той няма никаква вина за случилото се.", "ви́на / вина́", "вина́"),
    ("Дегустирахме отлични червени вина от Мелник.", "ви́на / вина́", "ви́на"),
    ("Соли супата, ако ти се струва безвкусна.", "со́ли / соли́", "соли́"),
]

# (sentence, word, variants, expected, confident)
# expected=None -> no auto-score, print for native review.
ITEMS = [
    ("Той говори тихо по телефона всяка вечер.", "говори", "го́вори / гово́ри / говори́", "гово́ри", True),
    ("Говори по-високо, защото не те чувам!", "говори", "го́вори / гово́ри / говори́", "говори́", True),
    ("От тенджерата се вдигаше гъста пара.", "пара", "па́ра / пара́", "па́ра", True),
    ("Той нямаше пукната пара в джоба си.", "пара", "па́ра / пара́", "пара́", True),
    ("Всяка вечер чета по една глава от книгата.", "чета", "че́та / чета́", "чета́", True),
    ("Хайдушката чета премина тайно през планината.", "чета", "че́та / чета́", "че́та", True),
    ("Пуловерът е изплетен от мека овча вълна.", "вълна", "въ́лна / вълна́", "въ́лна", True),
    ("Огромна вълна заля скалите на брега.", "вълна", "въ́лна / вълна́", "вълна́", True),
    ("Искам само една минута от времето ти.", "само", "са́мо / само́", "са́мо", True),
    ("Детето остана само вкъщи цяла вечер.", "само", "са́мо / само́", "само́", True),
    ("Тя носи тежка чанта всеки ден.", "носи", "но́си / носи́", "но́си", True),
    ("Носи багажа до колата, моля те.", "носи", "но́си / носи́", "носи́", True),
    ("Той прави всичко възможно, за да помогне.", "прави", "пра́ви / прави́", "пра́ви", True),
    ("Прави каквото ти казвам и няма да съжаляваш.", "прави", "пра́ви / прави́", "прави́", True),
    ("Вчера тя купи хляб и мляко от магазина.", "купи", "ку́пи / купи́", "ку́пи", True),
    ("Купи билети за концерта още днес.", "купи", "ку́пи / купи́", "купи́", True),
    ("Синът им учи медицина в София.", "учи", "у́чи / учи́", "у́чи", True),
    ("Синът им учи медицина в София.", "синът", "си́нът / синъ́т", "синъ́т", True),
    ("Учи всеки ден, ако искаш да изкараш изпита.", "учи", "у́чи / учи́", "учи́", True),
    ("Тя ходи на работа пеша всяка сутрин.", "ходи", "хо́ди / ходи́", "хо́ди", True),
    ("Тя ходи на работа пеша всяка сутрин.", "пеша", "пе́ша / пеша́", "пеша́", True),
    ("Ходи по-бързо, че ще закъснеем!", "ходи", "хо́ди / ходи́", "ходи́", True),
    ("Той моли за прошка всеки път, когато сгреши.", "моли", "мо́ли / моли́", "мо́ли", True),
    ("Той моли за прошка всеки път, когато сгреши.", "сгреши", "сгре́ши / сгреши́", "сгреши́", True),
    ("Тя реши дългата си коса пред огледалото.", "коса", "ко́са / коса́", "коса́", True),
    ("Купих боя за стените.", "боя", "бо́я / боя́", "боя́", True),
    ("Не се боя от нищо.", "боя", "бо́я / боя́", "боя́", True),
    ("Запишете промените, преди да затворите приложението.", "промените", "про́мените / проме́ните / промени́те", "проме́ните", True),
    # ---- native-review items (no auto-score) ----
    ("Напиши адреса с коса черта между двете имена.", "коса", "ко́са / коса́", None, False),
    ("Напиши адреса с коса черта между двете имена.", "между", "ме́жду / между́", None, False),
    ("Купи билети за концерта още днес.", "още", "о́ще / още́", None, False),
    ("Учи всеки ден, ако искаш да изкараш изпита.", "ако", "а́ко / ако́", None, False),
]

SYSTEM = (
    "Ти си експерт по българска фонетика и лексикално ударение. Дадено е "
    "изречение и дума от него с няколко възможни ударения. Определи "
    "правилното ударение на думата В ТОВА изречение. Отговори САМО с "
    "думата с поставен знак за ударение — нищо друго."
)


def ask(url, sentence, word, variants):
    messages = [{"role": "system", "content": SYSTEM}]
    for s, v, a in FEWSHOT:
        messages.append({"role": "user",
                         "content": f"Изречение: {s}\nДума: {v}"})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user",
                     "content": f"Изречение: {sentence}\nДума: {variants}"})
    body = json.dumps({"messages": messages, "temperature": 0,
                       "max_tokens": 12}).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"].strip().strip(".!")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8080")
    args = ap.parse_args()

    n_ok = n_scored = 0
    review = []
    print(f"{'word':<12} {'expected':<12} {'BgGPT':<14} verdict")
    for sentence, word, variants, expected, confident in ITEMS:
        got = ask(args.url, sentence, word, variants)
        if expected is not None:
            ok = got == expected
            n_scored += 1
            n_ok += ok
            mark = "OK" if ok else f"MISS   [{sentence[:44]}]"
            print(f"{word:<12} {expected:<12} {got:<14} {mark}", flush=True)
        else:
            review.append((word, got, sentence))
            print(f"{word:<12} {'(native?)':<12} {got:<14} review", flush=True)

    print(f"\nconfident set: {n_ok}/{n_scored} correct")
    print("\nfor native review (BgGPT's opinion on the disputed items):")
    for word, got, sentence in review:
        print(f"  {word}: BgGPT says {got}   [{sentence}]")
    print("\nGate (plan §P2.5): adopt teacher/arbiter design only if the "
          "confident set is (near-)perfect.")


if __name__ == "__main__":
    main()
