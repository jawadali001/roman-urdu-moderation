"""Train the baseline model and evaluate it.

Run from the repo root (after download.py, clean.py and split.py):

    python ml/scripts/train_baseline.py                  # train + validation report
    python ml/scripts/train_baseline.py --evaluate-test  # also score the test set

Use the validation report while experimenting. Score the test set only when you
are done tuning, and report that number: every extra look at the test set makes
it a little less trustworthy.

Outputs (ml/artifacts/):
    baseline.joblib                  the trained model (gitignored)
    baseline_meta.json               training settings, row counts, normalizer version
    baseline_metrics_validation.json full validation metrics
    baseline_metrics_test.json       only with --evaluate-test
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from rumod.data.schema import REPO_ROOT
from rumod.evaluation.metrics import evaluate_by_source, format_report
from rumod.models.predict import ModerationModel

DATA_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"


def read_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run split.py first.")
    return pd.read_csv(path, keep_default_na=False)[["text", "label", "source"]]


def evaluate(model: ModerationModel, df: pd.DataFrame, n_boot: int) -> dict:
    predictions = model.predict_labels(df["text"], progress=True)
    return evaluate_by_source(df, predictions, model.labels, n_boot=n_boot, progress=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline model")
    parser.add_argument("--evaluate-test", action="store_true", help="also score the test set")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    parser.add_argument("--C", type=float, default=None, help="logistic regression strength")
    args = parser.parse_args()

    train, validation = read_split("train"), read_split("validation")
    params = {} if args.C is None else {"C": args.C}

    total_steps = 4 if args.evaluate_test else 3
    started = time.time()

    print(f"[1/{total_steps}] Training on {len(train):,} rows")
    model = ModerationModel.fit(train["text"], train["label"], progress=True, **params)
    print(f"      trained in {time.time() - started:.0f}s; classes: {model.labels}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS_DIR / "baseline.joblib")
    (ARTIFACTS_DIR / "baseline_meta.json").write_text(
        json.dumps(model.meta, indent=2, default=str), encoding="utf-8"
    )

    print(f"[2/{total_steps}] Scoring the validation set ({len(validation):,} rows)")
    report = evaluate(model, validation, args.n_boot)
    (ARTIFACTS_DIR / "baseline_metrics_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nVALIDATION")
    print(format_report(report))

    if args.evaluate_test:
        print(f"[3/{total_steps}] Scoring the test set")
        test = read_split("test")
        test_report = evaluate(model, test, args.n_boot)
        (ARTIFACTS_DIR / "baseline_metrics_test.json").write_text(
            json.dumps(test_report, indent=2), encoding="utf-8"
        )
        print("\nTEST (report this one, and do not tune on it)")
        print(format_report(test_report))

    print(f"\n[{total_steps}/{total_steps}] Saved model and metrics to "
          f"{ARTIFACTS_DIR.relative_to(REPO_ROOT)}/ (total {time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()