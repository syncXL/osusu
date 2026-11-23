---
language:
- fon
license: cc-by-sa-4.0
pretty_name: "Frugal Tonal ASR for Fongbe — competition data"
task_categories:
- automatic-speech-recognition
tags:
- fon
- fongbe
- benin
- tonal
- zindi
- jaime-ma-langue
---

# Frugal Tonal ASR for Fongbe — competition data

Fongbe speech data of the Zindi competition "Frugal Tonal ASR for Fongbe", released by **J'aime Ma Langue**, the
national initiative of Benin for speech technology in Beninese languages (MTDI, ASIN, IIDiA).

## Licence

Data Sharing: Confidential, for use in this challenge only. No redistribution. Deletion required at the end of the challenge.

## Content

| File | Content |
|---|---|
| `Train_reliable.csv` | ≈ 106 h, tonal transcriptions reviewed and validated (column `recommended_validation`) |
| `Train_uncertain.csv` | ≈ 198 h, transcriptions of uncertain tonal quality, provided as they are |
| `Train_religious.csv` | ≈ 40 h, Bible readings (a single reader), tonal transcriptions |
| `Test.csv` | audio only (ids to transcribe) |
| `SampleSubmission.csv` | submission example: `ID`, `Target` |
| `audio_*.tar` | archives of the WAV audio files (mono, 16 kHz, 16 bits); `SHA256SUMS` to check the download |
| `conventions.md` | writing rules, normalisation, score, allowed resources |
| `syllabify.py` | **mandatory pre-processing of your submission** (standard library only, see "Submission") |
| `normalize.py`, `score.py` | normalisation and detailed tonal diagnostics (`pip install -r requirements.txt`) |
| `test_score.py` | tests of the score: `python test_score.py` must print "11/11 tests passed" |
| `measure_frugality.py` | frugality measurement used for the verification: size, real-time factor on one core, memory, offline |
| `starter_notebook.ipynb` | step-by-step notebook from the data to a valid submission (baseline w2v-BERT 2.0, free T4 GPU) |

Columns of the training files: `id`, `audio_path`, `transcription`, `speaker_id`, `duration` (seconds) and the available
metadata (`recommended_validation`, `protocol`, `book`, `chapter`). Speaker ids are anonymous. `protocol` (`p1`, `p2`, `p3`)
identifies the recording protocol of the uncertain data; `book` uses the standard three-letter Bible book codes (`GEN`, `EXO`…).

## Test set

`Test.csv` lists 9,407 utterances (21.8 h), audio only. It has two parts: **general** speech (6,382 utterances, 60 voices
that never appear in the training files) and **religious** speech (3,025 utterances, the reader of `Train_religious.csv`,
whole Bible books held out: 2 Chronicles, Proverbs, Zephaniah, Haggai, 2 Kings, 1 Chronicles). Each part is split between
the public and the private leaderboard, with different voices and books on each side. Keep the voices with
`recommended_validation = 1` in `Train_reliable.csv` for your own validation: like the test, they are unseen voices.

## Starter notebook

`starter_notebook.ipynb` goes from the data to a valid `submission.csv`: download and checks, a look at the data, a
`facebook/w2v-bert-2.0` baseline (MIT licence) fine-tuned on one archive of `Train_reliable` (about 18 h), scoring on
held-out voices with `score.py`, test transcription, `syllabify()` and submission checks. On a free Kaggle T4 it runs in
**2 h 32** and reaches a syllable WER of about **21 %** (20.6 % on held-out validation voices, 21.8 % on the public
leaderboard, 21.3 % on the private one). It is a starting point, not a winning recipe, and it is not frugal as is.

## Extracting the audio

The audio files are grouped in archives of about 2 GB. In the repository folder:

```bash
sha256sum -c SHA256SUMS                       # checks the archives
for f in audio_*.tar; do tar -xf "$f"; done   # creates audio/train_reliable/, audio/train_uncertain/, audio/train_religious/, audio/test/
```

The paths of the `audio_path` column of the CSV files then point directly to the WAV files.

## Submission and leaderboard metric

A CSV file with the columns `ID` (the `id` of `Test.csv`) and `Target`, one row per id of `Test.csv`, where **each `Target` has been
passed through `syllabify()`**. The leaderboard metric is the standard word error rate (WER) computed on these
syllables: the ground truth has been pre-processed with the same function by the organisers. Lower is better.

Why syllables: Fongbe is tonal, so a single wrong tone mark makes a whole word wrong, and the way words are split varies
in the transcripts. On syllables, a wrong tone costs one syllable, and word splitting no longer matters.

```python
import pandas as pd
from syllabify import syllabify

plain = pd.read_csv("my_transcriptions.csv")          # columns: id, transcription (plain Fongbe text)
sub = pd.DataFrame({"ID": plain["id"], "Target": plain["transcription"].fillna("").map(syllabify)})
sub.to_csv("submission.csv", index=False)          # columns: ID, Target
```

or, without pandas: `python syllabify.py my_transcriptions.csv submission.csv` (writes the columns `ID`, `Target`).

`syllabify()` normalises the text (Unicode NFD, lower case, punctuation, digits and elision apostrophes removed), cuts
each word into syllables and writes the tone of each syllable as a digit (1 high, 2 low, 3 rising, 4 falling, 5 mid):
`syllabify("Un nɔ́ ɖu nǔ")` gives `un nɔ1 ɖu nu3`. A submission that is not syllabified gets a very bad score, since its
words cannot match the syllables of the ground truth. `score.py` gives the same value (`TSER` without the cap at 1) plus tonal diagnostics.

## Frugality constraints (eliminatory for the prizes)

Complete system, from audio to tonal transcription: size ≤ 1,000 MB (all the weights, after any quantisation),
real-time factor ≤ 1 on a single CPU core, memory ≤ 2 GB, offline and without GPU. There is no constraint on training.
The details are in the competition rules.

Check your system before submitting, with the script we will use (Linux; root rights to cut the network):

```bash
python measure_frugality.py --weights my_system/ \
    --command "python infer.py --csv {csv} --root {root} --output {output}" \
    --csv Train_reliable.csv --limit 300
```

Your command reads a CSV `id, audio_path` and writes `id, transcription`. With a CSV that contains the transcriptions
(for example the `recommended_validation` voices), the script also recomputes the score.

## Citation

```bibtex
@misc{jml2026fongbechallenge,
  title  = {Frugal Tonal ASR for Fongbe: competition data},
  author = {{J'aime Ma Langue}},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/JaimeMaLangue/JML-Hackathon-fon-asr-data}},
  note   = {MTDI, ASIN, IIDiA, Benin. CC-BY-SA-4.0}
}
```
