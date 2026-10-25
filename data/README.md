# Data

How the training and evaluation data for the Roman Urdu moderation model is obtained, cleaned and split. Everything here is reproducible from three commands. Raw, interim and processed CSV files are **not** committed (see "What is committed").


| Source | What it is | Rows used | License |
|---|---|---|---|
| **RUHSOLD** ([Hugging Face](https://huggingface.co/datasets/community-datasets/roman_urdu_hate_speech)) | Roman Urdu tweets with fine-grained hate and offensive-language labels | 7,208 (train split only) | MIT, as listed on the Hugging Face page |
| **RUACD** ([GitHub](https://github.com/shaheerakr/roman-urdu-abusive-comment-detector)) | Roman Urdu YouTube comments, binary abusive / not abusive | 147,180 | Not verified. Data is not redistributed in this repo |

Our own hand-labeled and synthetic data (for the classes the public sets don't cover) will be added under `data/custom/`.

## Known issues in the sources (and how we handle them)

1. **RUHSOLD official test split is unlabeled.** All 2,002 labels in the Hugging Face copy are empty. We cannot use it, and we cannot compare against published benchmark numbers on it.
2. **RUHSOLD fine-grained validation split is an exact copy of train** (7,208 rows, 100% overlap). Using it would inflate validation scores. We ignore it and build our own splits.
3. **RUHSOLD label ids are not "0 = clean".** In both configs id 1 is `Normal` and id 0 is `Abusive/Offensive`. Labels are mapped by name using the names stored in `data/raw/ruhsold/manifest.json`, never by id.
4. **RUHSOLD coarse "Abusive/Offensive" is an umbrella** over all four hate classes (3,358 coarse vs 3,357 = 1,729 + 563 + 603 + 462 fine non-normal rows). It is never used to train our `abusive` class.
5. **RUACD is larger than the papers describe.** The file in the repo has 147,180 rows, not about 10,000. We only know it is already preprocessed: all text is lowercase, with no punctuation, digits, emoji, URLs or mentions, and it contains a lot of English. We could not verify how the extra rows were labeled.
6. **RUACD "abusive" is broader than our `abusive` label** (it can include profanity or hate). It is used only as extra volume for the abusive-vs-clean boundary.
7. **Style mismatch.** RUACD (preprocessed) and RUHSOLD (raw tweets) look different. The model must see identically normalized text in training and in the API (planned: `ml/src/rumod/features/normalize.py`).

## Pipeline

Run from the repository root with the virtual environment active:

```
python ml/src/rumod/data/download.py   # raw files -> data/raw/ (+ manifest.json with checksums)
python ml/src/rumod/data/clean.py      # -> data/interim/cleaned.csv + cleaning_report.json
python ml/src/rumod/data/split.py      # -> data/processed/{train,validation,test}.csv + split_report.json
```

Dependencies: `pandas requests datasets pyyaml scikit-learn`.

## Unified schema

Every source is converted to the same columns (`ml/src/rumod/data/schema.py`):

| Column | Meaning |
|---|---|
| `text` | the message (lightly cleaned, wording untouched) |
| `label` | class name from `configs/labels.yaml` |
| `source` | `ruhsold`, `ruacd` (later `synthetic`, `real`) |
| `origin_split` | split the row came from in its source |
| `is_synthetic` | generated row (train only, never validation/test) |
| `is_hard_negative` | clean message that looks risky |
| `dedupe_key` | lowercase text without URLs, mentions, punctuation or stretched letters |
| `label_id` | numeric id from `labels.yaml` (added by `split.py`) |

RUHSOLD "Sexism" maps to our `sexual_hate`, "Profane/Untargeted" to `profane`, "Religious Hate" to `religious_hate`, "Abusive/Offensive" to `abusive` and "Normal" to `clean`.

## Cleaning results

Measured on 2026-10-25. Rows removed by each step:

| Step | Removed | Remaining |
|---|---|---|
| Start | | 154,388 |
| Empty or no Latin letters | 0 | 154,388 |
| Same message, different labels, same source | 4 | 154,384 |
| Duplicates within a source | 191 | 154,193 |
| Clean in one source, not clean in the other | 0 | 154,193 |
| Duplicates across sources (kept the RUHSOLD row) | 1 | 154,192 |

After cleaning: RUACD 146,996 rows and RUHSOLD 7,196 rows. By label: clean 77,286, abusive 75,282, sexual_hate 601, religious_hate 563, profane 460. The two sources almost never share a message, so training on one and testing on the other is a real generalization test.

## Splits

Seed 42. Each source is split on its own, stratified by label (RUHSOLD 70/15/15, RUACD 90/5/5). Only the RUACD part of **train** is capped, at 10,000 rows per class, so it does not drown out the RUHSOLD classes. Validation and test are not capped.

| Source / class | Train | Validation | Test |
|---|---|---|---|
| RUACD clean | 10,000 | 3,672 | 3,672 |
| RUACD abusive | 10,000 | 3,678 | 3,678 |
| RUHSOLD clean | 2,694 | 577 | 578 |
| RUHSOLD abusive | 1,206 | 258 | 259 |
| RUHSOLD sexual_hate | 421 | 90 | 90 |
| RUHSOLD religious_hate | 394 | 85 | 84 |
| RUHSOLD profane | 322 | 69 | 69 |
| **Total** | **25,037** | **8,429** | **8,430** |

`split.py` stops with an error if a message appears in two splits, a synthetic row ends up in validation or test, or any label is missing.

**Evaluation caveats**

- The three hate classes have only about 70 to 90 rows each in validation and test, so their scores are noisy. Report them with confidence intervals.
- About 87% of validation and test rows come from RUACD. Always report metrics per source and per class, never one overall accuracy.
- Near-duplicates that differ by more than case, punctuation, URLs, mentions or stretched letters (for example edited retweets) can still cross splits.

## What is committed

Committed: this README and `data/processed/split_report.json` (counts only). Ignored by git: `data/raw/`, `data/interim/` and `data/processed/*.csv`, because the source licenses are not all confirmed and the files contain abusive text. Rebuild everything with the three commands above.

## References

- Rizwan et al., "Hate-Speech and Offensive Language Detection in Roman Urdu", EMNLP 2020: https://aclanthology.org/2020.emnlp-main.197
- RUACD is used in "Abusive language detection from social media comments using conventional machine learning and deep learning approaches" (Multimedia Systems, 2021): https://link.springer.com/article/10.1007/s00530-021-00784-8. Check the repository and paper for the exact citation before publishing.