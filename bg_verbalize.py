#!/usr/bin/env python3
# Copyright 2026 OmniVoice-BG (Apache-2.0)
"""Bulgarian text normalization (verbalizer): expand numbers, currency,
percents, dates and times to WORDS before TTS.

Why: the TTS verbalizes digits internally, but drops parts of long numbers
(a neural deletion error). Expanding to explicit words moves that step out of
the unreliable neural-internal path into controllable text — the same
make-it-explicit philosophy as the stress front-end. Runs BEFORE the Accentor
(which then marks the resulting Bulgarian words normally).

Scope: 0–999,999 cardinals with gender, ordinals (dates), лв/лева currency,
%, DD month [YYYY] dates, HH:MM times, phone-style digit runs. Native review
of the test table (`python bg_verbalize.py --selftest`) is the acceptance gate.
"""
import re

# --- cardinal building blocks ---
UNITS = {  # gender variants for 1,2 (m/f/n); others invariant
    0: "нула",
    1: {"m": "един", "f": "една", "n": "едно"},
    2: {"m": "два", "f": "две", "n": "две"},
    3: "три", 4: "четири", 5: "пет", 6: "шест", 7: "седем",
    8: "осем", 9: "девет",
}
TEENS = ["десет", "единадесет", "дванадесет", "тринадесет", "четиринадесет",
         "петнадесет", "шестнадесет", "седемнадесет", "осемнадесет",
         "деветнадесет"]
TENS = {2: "двадесет", 3: "тридесет", 4: "четиридесет", 5: "петдесет",
        6: "шестдесет", 7: "седемдесет", 8: "осемдесет", 9: "деветдесет"}
HUNDREDS = {1: "сто", 2: "двеста", 3: "триста", 4: "четиристотин",
            5: "петстотин", 6: "шестстотин", 7: "седемстотин",
            8: "осемстотин", 9: "деветстотин"}
MONTHS = ["", "януари", "февруари", "март", "април", "май", "юни", "юли",
          "август", "септември", "октомври", "ноември", "декември"]
# ordinals 1..31 (masculine; days are read as ordinals: "14 март" -> "четиринадесети")
ORD = {1: "първи", 2: "втори", 3: "трети", 4: "четвърти", 5: "пети",
       6: "шести", 7: "седми", 8: "осми", 9: "девети", 10: "десети",
       11: "единадесети", 12: "дванадесети", 13: "тринадесети",
       14: "четиринадесети", 15: "петнадесети", 16: "шестнадесети",
       17: "седемнадесети", 18: "осемнадесети", 19: "деветнадесети",
       20: "двадесети", 30: "тридесети"}


def _unit(u, gender):
    v = UNITS[u]
    return v[gender] if isinstance(v, dict) else v


def _three(n, gender="m"):
    """0..999 -> list of word tokens with internal 'и'."""
    if n == 0:
        return []
    parts = []
    h, rest = divmod(n, 100)
    if h:
        parts.append(HUNDREDS[h])
    if rest:
        if rest < 10:
            parts.append(_unit(rest, gender))
        elif rest < 20:
            parts.append(TEENS[rest - 10])
        else:
            t, u = divmod(rest, 10)
            parts.append(TENS[t])
            if u:
                parts.append(_unit(u, gender))
    if len(parts) >= 2:
        parts.insert(-1, "и")
    return parts


def verbalize_int(n, gender="m"):
    """0..999_999 -> Bulgarian words. gender applies to the final unit."""
    if n == 0:
        return UNITS[0]
    if n < 0:
        return "минус " + verbalize_int(-n, gender)
    th, rest = divmod(n, 1000)
    words = []
    if th:
        if th == 1:
            words.append("хиляда")
        else:
            words += _three(th, "f") + ["хиляди"]   # хиляда is feminine
    rest_words = _three(rest, gender)
    if th and rest_words and "и" not in rest_words:
        words.append("и")                             # хиляда И пет / И двеста
    words += rest_words
    return " ".join(words)


def verbalize_ordinal(n):
    """1..31 day-of-month ordinal (masculine)."""
    if n in ORD:
        return ORD[n]
    if 21 <= n <= 39:                    # 21..31: tens-word + ordinal unit
        t, u = divmod(n, 10)
        return f"{TENS[t]} и {ORD[u]}" if u else ORD.get(n, str(n))
    return str(n)


def _digits(s):
    return " ".join(_unit(int(c), "n") if c != "0" else "нула" for c in s)


# --- context handlers (regex replacements over running text) ---
def _currency(m):
    n = int(m.group(1))
    return f"{verbalize_int(n, 'm')} лева"          # лев is masculine


def _percent(m):
    return f"{verbalize_int(int(m.group(1)), 'm')} процента"


def _time(m):
    h, mn = int(m.group(1)), int(m.group(2))
    hw = verbalize_int(h, "m")
    if mn == 0:
        return f"{hw} часа"
    return f"{hw} и {verbalize_int(mn, 'f')}"        # минути feminine


def _date(m):
    d, mon = int(m.group(1)), m.group(2)
    out = f"{verbalize_ordinal(d)} {mon}"
    if m.group(3):                                  # optional year
        out += f" {verbalize_int(int(m.group(3)), 'f')} година"
    return out


def _phone(m):
    return _digits(m.group(0).replace(" ", ""))


def _bare_int(m):
    return verbalize_int(int(m.group(0)))


def normalize(text):
    """Expand numbers/currency/%/dates/times/phones in Bulgarian text."""
    # order matters: most specific first
    t = text
    t = re.sub(r"\b(\d{1,6})\s*(?:лв\.?|лева)\b", _currency, t)
    t = re.sub(r"\b(\d{1,3})\s*%", _percent, t)
    t = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", _time, t)
    t = re.sub(r"\b(\d{1,2})\s+(" + "|".join(MONTHS[1:]) + r")"
               r"(?:\s+(\d{4})\s*(?:година|г\.?)?)?", _date, t)
    t = re.sub(r"\b\d[\d ]{6,}\d\b", _phone, t)      # long grouped digit runs = phone
    t = re.sub(r"\b\d{1,6}\b", _bare_int, t)         # any remaining integer
    t = re.sub(r"\bг\.\b", "година", t)              # stray "г." after a year
    return t


SELFTEST = [
    ("Градинска люлка триместна само за 129 лева.",),
    ("Офертата важи до 31 декември 2026 година.",),
    ("Обади се на 0888 123 456 още днес.",),
    ("Отстъпка от 20% за всички клиенти.",),
    ("Срещата е насрочена за 14 март в 9 часа.",),
    ("Цената падна от 1250 на 899 лева.",),
    ("Магазинът работи от 8:30 до 21:00.",),
    ("Той се роди на 5 май 1987 г.",),
    ("Купи 2 и вземи третия безплатно.",),
    ("Цена: 21 лева. Виж и 105, 200, 1000, 2000, 1005.",),
]

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        for (s,) in SELFTEST:
            print("IN :", s)
            print("OUT:", normalize(s))
            print()
        print("Native check: verify 'и' placement, gender (лева=м), ordinals, "
              "and that NO digits remain.")
    else:
        print(normalize(sys.stdin.read()))
