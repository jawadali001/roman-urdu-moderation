"""Score the model on the hand-labeled set.

    python ml/scripts/score_real_set.py
    python ml/scripts/score_real_set.py --model ml/artifacts/baseline.joblib

Treat this as a TEST set: look at it, do not tune on it.

Outputs:
    ml/artifacts/real_set_metrics.json              metrics only (no text), fine to commit
    data/custom/real_labeled/real_set_errors.csv    misclassified messages with the model's
                                                    guess (contains text, gitignored)
"""
from __future__ import annotations

import argparse
import json

from rumod.data.real_set import load_labeled_batches, to_unified, validate
from rumod.data.schema import REPO_ROOT
from rumod.evaluation.metrics import format_report
from rumod.evaluation.real_score import error_table, score_real_set
from rumod.models.predict import ModerationModel

ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"
ERRORS_PATH = REPO_ROOT / "data" / "custom" / "real_labeled" / "real_set_errors.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a model on the hand-labeled set")
    parser.add_argument("--model", default=str(ARTIFACTS_DIR / "baseline.joblib"))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    df = load_labeled_batches()
    for problem in validate(df):
        print(f"note: {problem}")
    rows = to_unified(df)
    skipped = len(df[df["label"] != ""]) - len(rows)
    print(f"scoring {len(rows):,} labeled messages ({skipped} unsure/unusable skipped)\n")

    model = ModerationModel.load(args.model)
    result = score_real_set(model, rows, n_boot=args.n_boot, progress=True)

    print("\n##### ALL LABELED ROWS (candidate rows flatter the model)")
    print(format_report({"real_all": result["all"]}, show_confusion=True))
    if result["random_only"]:
        print("\n##### RANDOM ROWS ONLY (the fair estimate)")
        print(format_report({"real_random_only": result["random_only"]}, show_confusion=True))
    else:
        print("\nNo rows are marked as random, so random-only numbers are not available.")

    validation_path = ARTIFACTS_DIR / "baseline_metrics_validation.json"
    if validation_path.exists() and result["random_only"]:
        reference = json.loads(validation_path.read_text(encoding="utf-8"))["ruhsold"]
        print(f"\nFor comparison, RUHSOLD validation macro-F1 was {reference['macro_f1']:.3f}; "
              f"on the hand-labeled random rows it is {result['random_only']['macro_f1']:.3f}.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "real_set_metrics.json").write_text(
        json.dumps({"model": args.model, "scored": len(rows), "all": result["all"],
                    "random_only": result["random_only"]}, indent=2),
        encoding="utf-8",
    )
    errors = error_table(rows, result["predictions"])
    errors.to_csv(ERRORS_PATH, index=False)
    where = ERRORS_PATH.relative_to(REPO_ROOT)
    print(f"\n{len(errors):,} misclassified messages written to {where}")
    print("Reading them is the fastest way to see whether the model or a label is wrong.")


if __name__ == "__main__":
    main()