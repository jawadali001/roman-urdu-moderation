"""Compare your labels with the official RUHSOLD labels.

    python ml/scripts/score_agreement.py

Needs data/custom/agreement/agreement_labeled.csv (exported from the labeling tool)
and agreement_key.csv. Prints counts and rates only, no tweet text.

Outputs:
    ml/artifacts/agreement_metrics.json                 metrics only, fine to commit
    data/custom/agreement/agreement_disagreements.csv   the tweets where you differ (gitignored)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from rumod.data.real_set import EVAL_LABELS
from rumod.data.schema import REPO_ROOT
from rumod.evaluation.agreement import agreement_report, format_agreement, split_comparable

AGREEMENT_DIR = REPO_ROOT / "data" / "custom" / "agreement"
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.csv"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"


def natural_prior() -> dict[str, float]:
    """How common each class is in RUHSOLD's cleaned training data."""
    train = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    labels = train.loc[train["source"] == "ruhsold", "label"]
    shares = labels.value_counts(normalize=True)
    return {label: float(shares.get(label, 0.0)) for label in EVAL_LABELS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the annotation-agreement study")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--allow-partial", action="store_true",
                        help="compare only the tweets labeled so far")
    args = parser.parse_args()

    labeled_path = AGREEMENT_DIR / "agreement_labeled.csv"
    key_path = AGREEMENT_DIR / "agreement_key.csv"
    for path in (labeled_path, key_path):
        if not path.exists():
            raise SystemExit(f"{path} not found.")

    mine = pd.read_csv(labeled_path, keep_default_na=False, dtype=str)
    key = pd.read_csv(key_path, keep_default_na=False, dtype=str)
    both = mine.merge(key, on="id", how="inner")
    if len(both) != len(key):
        print(f"note: {len(key) - len(both)} sampled tweets are missing from your labeled file")

    unlabeled = int((both["label"].str.strip() == "").sum())
    if unlabeled and not args.allow_partial:
        raise SystemExit(f"{unlabeled} tweets are not labeled yet. Finish labeling, "
                         "or use --allow-partial.")

    official, yours, info = split_comparable(
        both["official_label"], both["label"].str.strip()
    )
    skipped = info["skipped"]
    print(f"skipped (not compared): unsure {skipped['unsure']}, unusable {skipped['unusable']}, "
          f"blank {skipped['']}")

    report = agreement_report(
        official, yours, list(EVAL_LABELS), natural_prior(), n_boot=args.n_boot, progress=True
    )
    print("\n" + format_agreement(report))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "agreement_metrics.json").write_text(
        json.dumps({**report, "skipped": skipped}, indent=2), encoding="utf-8"
    )
    kept = both[~both["label"].str.strip().isin(["unsure", "unusable", ""])]
    different = kept[kept["official_label"] != kept["label"].str.strip()]
    different[["id", "text", "official_label", "label", "notes"]].to_csv(
        AGREEMENT_DIR / "agreement_disagreements.csv", index=False
    )
    print(f"\n{len(different)} disagreements written to "
          f"data/custom/agreement/agreement_disagreements.csv")
    print("Reading them shows whether the guide or the official label is the stricter one.")


if __name__ == "__main__":
    main()