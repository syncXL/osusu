# Conventions — Frugal Tonal ASR for Fongbe

## 1. How the transcriptions are written

The reference transcriptions follow the **writing guide of the linguists of the J'aime Ma Langue project**. They were
reviewed and validated by a second annotator (`Train_reliable.csv` and the general part of the test):

- every word carries its **tones**: high tone (acute, `á`), low tone (grave, `à`), rising tone (caron, `ǎ`);
- the **Fon alphabet** is used (`ɖ`, `ɛ`, `ɔ`…), with a single written form per word;
- common words borrowed from **French or English** keep their original spelling, without tones;
- some references keep **contractions** as the linguists wrote them (for example `ɖ'é`), and a few numbers are written
  with digits. The data is provided as it is: no reference was rewritten for the competition.

`Train_uncertain.csv` contains transcriptions whose tonal quality is **uncertain** (provided as they are).
`Train_religious.csv` contains Bible readings by **a single reader**; chapter titles are included.

## 2. Normalisation (`normalize.py`)

Before any score is computed, the references **and** the submissions go through the same script, `normalize.py`:

- Unicode **NFD**, lower case, mistyped characters fixed (`ε` -> `ɛ`…);
- **punctuation**, **digits** and **elision apostrophes** removed (`ɖ'é` -> `ɖé`);
- letters and **tones** are kept.

Writing a number with words or with digits therefore makes no difference for the digits (they are removed on both sides);
only the spoken words count.

## 3. Leaderboard metric: WER on syllables (`syllabify.py`)

The leaderboard uses the standard word error rate (WER) of Zindi, computed on syllables instead of words. The ground truth
has been passed through `syllabify()` by the organisers; **you must pass your transcriptions through the same function
before submitting** (see the README). `syllabify()` normalises the text as above, splits it into syllables (onset + vowel +
optional syllable-final "n"), writes the tone of each syllable as a digit (1 high, 2 low, 3 rising, 4 falling, 5 mid, no
digit when unmarked) and joins the syllables with spaces. Lower is better.

A syllable is correct only if its letters and its tone are correct: a wrong tone costs one syllable, not a whole word, and
the way words are split no longer matters (`kɔ́xojí` and `kɔ́ xo jí` give the same syllables).

To check a submission on your own validation data, `score.py` gives the same value (`TSER`, the syllable edit distance
divided by the number of reference syllables, here without its cap at 1) and tonal diagnostics that do not affect the
ranking (conditional TER: share of wrong tones among the correctly recognised syllables; TER per tone):

```bash
pip install -r requirements.txt
python score.py --reference my_reference.csv --submission my_plain_transcriptions.csv
```

`score.py` takes plain (not syllabified) transcriptions. `python test_score.py` checks its behaviour on known cases, and
`python syllabify.py` runs the self-test of the pre-processing.

## 4. Test and validation

- The test contains **audio only** (`Test.csv`). It has a **general** part and a **religious** part, each split between the
  public and the private leaderboard.
- **General part**: none of the people speaking in it appears in the training files; the voices of the public leaderboard
  and those of the private leaderboard are different.
- **Religious part**: same reader as `Train_religious.csv`; **whole books** of the Bible are held out.
- The `recommended_validation` column of `Train_reliable.csv` (value `1`) marks a group of voices (about 5.5 h) that we
  recommend keeping for validation: your model will not have heard these voices, as in the test.

## 5. Allowed resources

- Pre-trained models and public data: allowed if they are freely accessible, declared before the end of the competition,
  counted in the size of the system, and under a **licence that allows commercial use**.
- **Bible text**: only the Bible text provided in `Train_religious.csv` is allowed; any other Bible text in Fongbe is forbidden.
- Any data from the J'aime Ma Langue corpus other than the files provided is forbidden.
- Manual annotation or correction is forbidden; automatic filtering and weighting of the uncertain data are allowed.
