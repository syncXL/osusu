"""Fon text normalisation used by the official score (standard library only).

Three writing spaces, from richest to poorest:

    raw ──normalize()─────► canonical form: NFD, lower case, look-alike characters fixed
        ──tonal_target()──► what is pronounced: letters, spaces, tones; no punctuation, no digits
        ──strip_tones()───► the same letters, with the tones removed from the vowels

The official score (`score.py`) works in the `tonal_target()` space, applied to the reference AND to the submission.
`eval_space(t) = strip_tones(tonal_target(t))` is the tone-free space, given for diagnostics.

Documented choices:
- NFD internally: `ɔ́` and `ɛ́` have no precomposed form; in NFC the corpus would be mixed.
- Elision apostrophes are removed (`ɖ'é` -> `ɖé`), so writing them or not makes no difference.
- Digits are removed: the convention is to transcribe the spoken language; the few references that still
  contain digits are kept as they are.
- A tone is only removed by `strip_tones()` when it sits on a vowel: on a consonant (`ń`) it is part of the letter.
"""
from __future__ import annotations

import re
import unicodedata

# Fon tones, as combining marks (NFD)
ACUTE, CARON, GRAVE, CIRCUMFLEX, MACRON = "́", "̌", "̀", "̂", "̄"
TONES = frozenset({ACUTE, CARON, GRAVE, CIRCUMFLEX, MACRON})
# Diacritics with no tonal value in Fon, found in the corpus: removed
SPURIOUS = frozenset({"̧", "̈", "̣"})     # cedilla, diaeresis, dot below
LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz") | {"ɔ", "ɖ", "ɛ"}
DIGITS = frozenset("0123456789")
VOWELS = frozenset("aeiou") | {"ɔ", "ɛ"}
MARKUP = frozenset("<>+*")
LOOK_ALIKES = {                  # wrong character -> correct letter or sign
    "ε": "ɛ", "Ε": "ɛ",                      # Greek epsilon
    "ð": "ɖ", "đ": "ɖ", "ɗ": "ɖ",            # barred or hooked d
    "‚": ",", "“": '"', "”": '"', "‘": "'", "’": "'",
    "œ": "oe",
}
_SPACES = re.compile(r" {2,}")


def normalize(text: str) -> str:
    """Canonical form: NFD, lower case, look-alike characters fixed. Idempotent."""
    t = unicodedata.normalize("NFD", text).lower()
    t = unicodedata.normalize("NFD", t)          # lower-casing can recompose characters
    for wrong, right in LOOK_ALIKES.items():
        t = t.replace(wrong, right)
    return unicodedata.normalize("NFD", t)


def tonal_target(text: str, digits: bool = False) -> str:
    """What is pronounced: letters, spaces and tones. Everything else is removed.
    `digits=True` (training variant only): each digit 0-9 is kept as a separate word ("21h" -> "2 1 h");
    `rejoin_digits` rebuilds the numbers after decoding. The official score always uses `digits=False`."""
    out: list[str] = []
    for c in normalize(text):
        if c in TONES:
            # a tone with no letter before it (typed after a space) is not pronounced
            if out and out[-1] != " ":
                out.append(c)
        elif digits and c in DIGITS:
            out.extend((" ", c, " "))           # one digit = one "word"
        elif c in LETTERS or c == " ":
            out.append(c)
        elif c.isdigit() or c in SPURIOUS or c in MARKUP \
                or unicodedata.category(c).startswith("P"):
            continue
        elif c.isspace():
            out.append(" ")
        # any other character (outside the alphabet) is removed
    return _SPACES.sub(" ", "".join(out)).strip()


def strip_tones(text: str) -> str:
    """Remove the tones carried by a vowel; keep those carried by a consonant (they are letters)."""
    out, base = [], ""
    for c in unicodedata.normalize("NFD", text):
        if c in TONES:
            if base in VOWELS:
                continue
        else:
            base = c
        out.append(c)
    return "".join(out)


def eval_space(text: str, digits: bool = False) -> str:
    """Tone-free space (word error rate without tones)."""
    return " ".join(strip_tones(tonal_target(text, digits)).split())


def eval_space_tonal(text: str, digits: bool = False) -> str:
    """Tonal space (word error rate with tones)."""
    return " ".join(tonal_target(text, digits).split())


def rejoin_digits(t: str) -> str:
    """Output of a `digits=True` model: consecutive single digits become a number again ("2 1" -> "21")."""
    out: list[str] = []
    for w in t.split():
        if w.isdigit() and out and out[-1].isdigit():
            out[-1] += w
        else:
            out.append(w)
    return " ".join(out)


def digit_sequence(t: str) -> str:
    """Ordered sequence of the digits of a text, without separators (insensitive to grouping)."""
    return "".join(c for c in unicodedata.normalize("NFD", t) if c in DIGITS)


def _self_test() -> None:
    assert eval_space("Un nɔ́ ɖɔxó, kpódó mɛ!") == "un nɔ ɖɔxo kpodo mɛ"
    assert eval_space("D'é mɛ 2026 \"bureau\".") == "dé mɛ bureau".replace("é", "e")
    assert eval_space_tonal("Kɔ́ 2 026 F, 3.5 %", digits=True) == unicodedata.normalize("NFD", "kɔ́ 2 0 2 6 f 3 5")
    assert eval_space("ɖo 21h", digits=True) == "ɖo 2 1 h" and rejoin_digits("ɖo 2 1 h 0 5") == "ɖo 21 h 05"
    assert digit_sequence("Numéro : 05 55, le 2 026") == "05552026"
    assert eval_space("Kɔ 2026", digits=False) == "kɔ"
    assert eval_space_tonal("Mǐ ná yi") == unicodedata.normalize("NFD", "mǐ ná yi")
    assert eval_space("ɛ́ ε ð") == "ɛ ɛ ɖ"                  # look-alike characters
    assert strip_tones(unicodedata.normalize("NFD", "ń")) == unicodedata.normalize("NFD", "ń")
    assert eval_space("  a   ́ b ") == "a b"                 # orphan tone, multiple spaces
    for t in ("Nɛ́ ká gbɔn", "É cí nú mì ɖɔ"):
        assert normalize(normalize(t)) == normalize(t)     # idempotence
    print("normalize.py: self-test OK")


if __name__ == "__main__":
    _self_test()
