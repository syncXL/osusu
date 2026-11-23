#!/usr/bin/env python
"""Official score of the "Frugal Tonal ASR for Fongbe" competition.

    python score.py --reference Reference.csv --submission my_submission.csv [--usage Public|Private] [--output report.json]

The submission uses the Zindi format `ID, Target` (see SampleSubmission.csv); the columns `id, transcription`
of the training files are also accepted, so that local validation files can be scored as they are.
The reference has `id, transcription` (or `ID, Target`), and optionally `Usage`.
The text is first normalised by `normalize.py`: Unicode NFD, lower case, punctuation, digits and elision
apostrophes removed. Letters and tones are kept.

Each text is then split into syllables (onset + vowel + optional syllable-final "n", and its tone):
  TSER (70 %): edit distance between the syllable sequences (base + tone) / number of reference syllables, capped at 1
  conditional TER (30 %): among the syllables matched without looking at the tone, the share whose tone is wrong
  score = 0.7 x TSER + 0.3 x conditional TER  (lower is better)
Errors are accumulated over the whole set; an id missing from the submission counts as an empty transcription.
Diagnostics (no effect on the ranking): segmental syllable error without tones, matched syllable rate, TER per tone.
"""
import argparse
import collections
import csv
import json
import sys
import unicodedata as ud
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import tonal_target  # noqa: E402

from rapidfuzz.distance import Levenshtein  # required, version pinned in requirements.txt

VOWELS = set("aeiouɛɔə")
TONES = {"́": "high", "̀": "low", "̌": "rising", "̂": "falling", "̄": "mid"}


def units(word):
    """Base characters with their combining marks: [[char, [marks]], ...]."""
    out = []
    for c in word:
        if ud.category(c).startswith("M") and out:
            out[-1][1].append(c)
        elif not ud.category(c).startswith("M"):
            out.append([c, []])
    return out


def syllables(text):
    """List of (base, tone): base = onset + vowel (+ syllable-final n), tone = sorted tone marks ('' = unmarked)."""
    res = []
    for word in tonal_target(text or "").split():
        u = units(word); i = 0; onset = ""
        while i < len(u):
            b, m = u[i]
            tone = "".join(sorted(x for x in m if x in TONES)); other = "".join(x for x in m if x not in TONES)
            if b in VOWELS:
                base = onset + b + other; onset = ""
                if i + 1 < len(u) and u[i + 1][0] == "n" and not u[i + 1][1] and (i + 2 >= len(u) or u[i + 2][0] not in VOWELS):
                    base += "n"; i += 1
                res.append((base, tone)); i += 1
            elif tone:                                  # syllabic consonant carrying a tone (ń, m̀…)
                res.append((onset + b + other, tone)); onset = ""; i += 1
            else:
                onset += b + other; i += 1
        if onset:
            res.append((onset, ""))
    return res


def _distance(a, b):
    return Levenshtein.distance(a, b)


def _equal_pairs(a, b):
    """Pairs (i, j) matched identically in the rapidfuzz edit alignment."""
    for op in Levenshtein.opcodes(a, b):
        if op.tag == "equal":
            for k in range(op.src_end - op.src_start):
                yield op.src_start + k, op.dest_start + k


def measure(pairs):
    """pairs: list of (reference, hypothesis). Errors are accumulated over the whole set."""
    c = collections.Counter(); per_tone = collections.Counter(); per_tone_err = collections.Counter()
    for ref, hyp in pairs:
        r, h = syllables(ref), syllables(hyp)
        c["ref_syllables"] += len(r)
        c["tonal_distance"] += _distance([b + "|" + t for b, t in r], [b + "|" + t for b, t in h])
        rb, hb = [b for b, _ in r], [b for b, _ in h]
        c["segmental_distance"] += _distance(rb, hb)
        for i, j in _equal_pairs(rb, hb):
            tr, th = r[i][1], h[j][1]
            c["matched"] += 1; c["wrong_tone"] += tr != th
            cat = "+".join(TONES[x] for x in tr) or "unmarked"
            per_tone[cat] += 1; per_tone_err[cat] += tr != th
    tser = min(1.0, c["tonal_distance"] / max(1, c["ref_syllables"]))
    terc = c["wrong_tone"] / max(1, c["matched"])
    return {"score": 0.7 * tser + 0.3 * terc, "TSER": tser, "TER_conditional": terc,
            "segmental_SER_without_tones": c["segmental_distance"] / max(1, c["ref_syllables"]),
            "matched_syllable_rate": c["matched"] / max(1, c["ref_syllables"]),
            "TER_per_tone": {k: round(per_tone_err[k] / per_tone[k], 4) for k in sorted(per_tone)},
            "counts": {k: int(v) for k, v in c.items()}}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def id_and_text(row):
    """(id, text) of a CSV row, in the Zindi format `ID, Target` or in the format `id, transcription`."""
    if "ID" in row:
        return row["ID"], row.get("Target") or ""
    return row["id"], row.get("transcription") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--usage", choices=["Public", "Private"], help="score only one part of the test")
    ap.add_argument("--output", help="JSON file for the detailed report")
    a = ap.parse_args()
    refs = [r for r in read_csv(a.reference) if not a.usage or r.get("Usage") == a.usage]
    refs = [id_and_text(r) for r in refs]
    sub = dict(id_and_text(r) for r in read_csv(a.submission))
    missing = sum(1 for i, _ in refs if i not in sub)
    report = measure([(text, sub.get(i, "")) for i, text in refs])
    report.update({"utterances": len(refs), "missing_ids_scored_as_empty": missing, "usage": a.usage or "all"})
    print(f"score {report['score']:.4f}  TSER {report['TSER']:.4f}  conditional TER {report['TER_conditional']:.4f}  "
          f"({len(refs)} utterances, {missing} missing)")
    if a.output:
        Path(a.output).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
