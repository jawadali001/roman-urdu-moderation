# Roman Urdu Moderation: Plan and Progress

_Last updated: 2026-10-25 · Status legend: ✅ done · ⏳ in progress · ☐ not started_

## 1. Goal

Build a production-style **Roman Urdu content-moderation system**: a text classifier, a FastAPI service that serves it, and three frontends that use the API. It is a portfolio project, so the emphasis is on engineering practice (reproducible data, tests, CI, Docker) and **honest evaluation**, not on a headline accuracy number.

Build order: **engine first** (data → model → API), then the frontends on top of a frozen API contract.

## 2. Scope

### Classes (8 defined in `configs/labels.yaml`, 5 active)

| Class | Action | Status | Data source |
|---|---|---|---|
| clean | allow | active | RUHSOLD + RUACD |
| abusive | block | active | RUHSOLD + RUACD |
| profane | flag | active | RUHSOLD |
| religious_hate | block | active | RUHSOLD |
| sexual_hate | block | active | RUHSOLD ("Sexism") |
| threat | block | planned | own data |
| spam | flag | planned | own data (synthetic + real) |
| personal_info | flag | planned | regex rules on raw text |

`clean_hard` (clean messages that look risky) is not a class. It is a flag column (`is_hard_negative`) on clean rows.

### Deliverables

| Part | Folder | Description |
|---|---|---|
| Engine | `ml/` | data pipeline, normalization, models, evaluation |
| API | `api/` | FastAPI: `/predict`, `/predict/batch`, `/health`; Docker; deployed |
| Chat simulator (main frontend) | `web/` | chat room where each message passes through the API; moderated messages are blurred; live log panel |
| Developer portal | `web/` | landing page, docs, "try it" box, code snippets, usage chart |
| Chrome extension (stretch) | `extension/` | blurs toxic comments on a page |
| Documentation | `docs/`, `data/README.md` | this file, API contract, evaluation, labeling guide |

## 3. Planned architecture

```
raw text ──► personal_info rules (raw text) ─┐
   │                                          ├─► thresholds + actions ─► allow / flag / block
   └─► normalize() ─► TF-IDF ─► classifier ──┘      (configs/labels.yaml)
```

- `normalize()` is a single function used for both training and serving. A saved model records the normalizer version and refuses to load on a mismatch.
- Class names, actions and thresholds live in one file, `configs/labels.yaml`. The starting thresholds are block 0.80 and flag 0.50. They are tuned on validation data only, never on test.

## 4. Roadmap and status

### Phase 0: Setup ✅
- ✅ Monorepo structure, MIT license, `labels.yaml`
- ✅ Virtual environment, `pyproject.toml` (`pip install -e "ml[dev]"`)
- ✅ CI for the ML package (ruff + pytest), green on GitHub Actions

### Phase 1: Data ✅ (tag `v0.1-data`)
- ✅ Download script with manifest and checksums
- ✅ Unified schema and label mapping by name
- ✅ Cleaning and de-duplication within and across sources
- ✅ Own train/validation/test splits (per source, stratified, seed 42)
- ✅ Data documentation (`data/README.md`)

### Phase 2: Engine foundation ⏳
- ✅ Text normalization with 24 unit tests
- ✅ Evaluation metrics: per-class and per-source scores with bootstrap confidence intervals
- ✅ Baseline model (character n-gram TF-IDF + logistic regression), first real run done (results in section 7)
- ✅ Toxic-vs-clean view with a false-alarm rate (share of clean messages that get flagged)
- ⏳ Experiments: data-source ablation and cross-dataset (train on one source, test on the other), plus class-weight and regularization comparison. Script written (`ml/scripts/run_experiments.py`), **run on the real data pending**
- ☐ Threshold tuning on validation data
- ☐ `personal_info` regex rules (run on raw text)

### Phase 3: Own data and the 3 extra classes ☐
- ☐ Labeling guide (`docs/labeling-guide.md`)
- ☐ Synthetic examples for spam, threat and personal_info (training only)
- ☐ Hand-labeled **real** test set, about 500-800 rows with at least 50 per class (validation/test only)
- ☐ Enable the three new classes in `labels.yaml`

### Phase 4: API ☐
- ☐ Freeze the API contract (`docs/api-contract.md`) and Pydantic schemas
- ☐ `/predict`, `/predict/batch`, `/health`; per-word contribution in the response
- ☐ API tests, Dockerfile, `ci-api.yml`
- ☐ Deploy (Render or Railway) and handle free-tier cold starts

### Phase 5: Frontends ☐
- ☐ Chat simulator (Next.js + shadcn/ui, dark theme)
- ☐ Developer portal as a second route
- ☐ Chrome extension (stretch goal; cut first if time runs short)

### Phase 6: Polish ☐
- ☐ Root `README.md` with architecture diagram, results and limitations
- ☐ `docs/architecture.md` and `docs/evaluation.md`
- ☐ Demo GIF, milestone tags

### Time estimate
The original estimate was 7-8 days at 8 hours a day. The foundation phases were deliberately given more time, because the frontends depend on them. If time runs short, cut in this order: Chrome extension, then dashboard statistics, then extra classes. Never cut tests, CI or the honest evaluation.

## 5. Progress log

All entries are from 2026-10-25.

| Commit | What |
|---|---|
| `1ba6156`, `41718bc` | Monorepo scaffold, package markers, `.gitignore` fixed |
| `87c577c` | MIT license and `labels.yaml` |
| `0a2f1cd` | Dataset download script |
| `92de2f5` | Unified schema; loader refuses the broken RUHSOLD splits |
| `be6cd99` | Cleaning, per-source splits, RUACD cap |
| `2a25aa6` | Fix: de-duplication key squeezes only repeated letters |
| `cff6c95`, `1fcbaba` | Data README (moved to the right folder), normalization with tests. **Tag `v0.1-data`** |
| `603dc82` | `pyproject.toml`, CI workflow, lint fixes. `ci-ml` green (run #12) |
| `348c41b` | Baseline model, evaluation metrics, training script, this roadmap |

## 6. Key decisions

| Decision | Reason |
|---|---|
| Engine first, frontends later | Frontends are only as good as the API; a frozen contract makes them quick to build |
| Monorepo (`ml/`, `api/`, `web/`, `extension/`) | One place that shows the whole story; per-component CI |
| 8 classes defined, 5 active | The five with real public data come first; the rest are added once their data exists |
| Map dataset labels by name, never by id | In RUHSOLD, id 1 is "Normal"; assuming 0 = clean would flip the labels |
| Build our own splits | The official RUHSOLD test split is unlabeled and its validation split is a copy of train |
| Soften class weights if full balancing over-flags (the `sqrt` option, tested in the experiments) | Fully balanced weights likely cause the false alarms on clean tweets seen in baseline v0 |
| Split each source separately, cap RUACD in train at 10,000 per class | RUACD is about 95% of the rows and would drown out the hate classes |
| One `normalize()` for training and serving | Mismatched normalization silently lowers accuracy |
| Report per source and per class, with confidence intervals | Hate classes have only about 70-90 validation/test rows each; one overall number would mostly measure RUACD |
| RUACD predictions collapsed to clean/abusive when scoring | RUACD's "abusive" is broader than our `abusive` class |
| Synthetic data is training-only; real hand-labeled data is evaluation-only | Prevents inflated scores from the model learning our own templates |
| Raw, interim and processed CSV files are not committed | Source licenses are not all confirmed and the files contain abusive text |
| Tests and CI from the start | Core goal of the project is demonstrable engineering practice |

## 7. Results so far

**Baseline v0** (`ml/scripts/train_baseline.py`, validation set, 25,037 training rows, class weights fully balanced, C = 4).

| Source | Macro-F1 [95% CI] | Notes |
|---|---|---|
| RUHSOLD (5 classes, n = 1,079) | **0.672** [0.635, 0.706] | accuracy 0.766 |
| RUACD (clean vs abusive, n = 7,350) | 0.979 [0.976, 0.982] | likely inflated, see section 10 |

RUHSOLD per class (F1, 95% CI): clean 0.882 · sexual_hate 0.677 · religious_hate 0.642 · abusive 0.624 · profane 0.536 [0.42, 0.63].
Toxic-vs-clean on RUHSOLD: precision 0.879, recall 0.839, F1 0.858, and **10.1% of clean tweets are flagged**.

What the confusion matrix shows:
- `abusive` is under-predicted (194 predicted vs 258 true): 54.7% correct, 20.5% sent to clean, the rest to other toxic classes.
- Over-flagging: 5.0% of clean tweets are labeled `religious_hate`, and the hate classes are predicted more often than they occur (108 predicted vs 85 true for religious_hate). Fully balanced class weights are the likely cause.
- `profane` is the weakest class (F1 0.536) and is mostly confused with `abusive`.
- Most errors are between neighbouring toxic classes, not between toxic and clean.

These are validation numbers from the first model. The test set has not been scored yet.

## 8. Data snapshot

Raw: RUHSOLD 7,208 labeled rows (train only is usable), RUACD 147,180 rows. After cleaning: **154,192** rows (RUACD 146,996, RUHSOLD 7,196).

| Class | Rows after cleaning |
|---|---|
| clean | 77,286 |
| abusive | 75,282 |
| sexual_hate | 601 |
| religious_hate | 563 |
| profane | 460 |

Splits: train 25,037 · validation 8,429 · test 8,430. Full breakdown and caveats are in [`data/README.md`](../data/README.md).

## 9. Evaluation protocol

1. **In-distribution:** held-out slices of RUHSOLD (all 5 classes) and RUACD (clean/abusive), reported separately.
2. **Cross-dataset:** train on one source, test on the other.
3. **Real hand-labeled set:** the most trustworthy number, since it comes from neither public dataset.
4. The validation set is for tuning. The test set is scored once per finished model, and that number is the one reported.

## 10. Known issues and risks

- **RUHSOLD packaging bugs:** unlabeled test split and duplicated validation split (handled, see `data/README.md`). Published benchmark numbers cannot be compared.
- **Very high in-distribution RUACD scores are expected** and should not be trusted on their own: the file is preprocessed, partly English, and its labels are unverified. The baseline's RUACD F1 of 0.979 against 0.672 on RUHSOLD shows the gap. The cross-dataset and real-data tests are what count.
- **Fine-grained RUHSOLD classes overlap** (abusive vs profane, and toxic classes with each other), which caps how high per-class scores can go.
- **Selection on the validation set:** comparing many variants on the same validation data gives optimistic results for the winner. The test set is scored once at the end.
- **Tiny hate classes:** 460-601 rows each overall, so their scores are noisy.
- **RUACD license is unverified;** its data is not redistributed in this repo.
- **Style mismatch** between RUACD (preprocessed) and raw tweets or chat, addressed by `normalize()`.
- **Roman Urdu only:** text in Urdu script or with no Latin letters is treated as empty and gets the default label.
- **Windows:** `make` is not installed by default, so the Makefile is not used yet.
- **Free-tier hosting cold starts** will make the first API call slow; the frontends need loading states.

## 11. How to reproduce

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e "ml[dev,data]"

python ml/src/rumod/data/download.py     # raw data
python ml/src/rumod/data/clean.py        # cleaned.csv
python ml/src/rumod/data/split.py        # train / validation / test
python ml/scripts/train_baseline.py      # model + validation report
python ml/scripts/run_experiments.py     # data-source and model comparisons (validation only)

cd ml; ruff check .; pytest -q           # lint and tests (same as CI)
```

## 12. Next actions

1. Run the experiments on the real data and decide the training setup: whether RUACD helps on RUHSOLD, which class-weight and C setting reduces false alarms without losing recall.
2. Tag `v0.2-baseline`.
3. Labeling guide, then the real hand-labeled set.
4. Score the test set once for the chosen model.
5. Update this file at every milestone and tag (`v0.3-api`, ...).