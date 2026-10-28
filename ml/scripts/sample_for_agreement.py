"""Create the sample for the annotation-agreement study.

    python ml/scripts/sample_for_agreement.py            # 30 tweets per class = 150

Writes to data/custom/agreement/ (gitignored):

    agreement_to_label.csv   id + text only. Open this in scripts/label_tool.html.
    agreement_key.csv        id -> official RUHSOLD label. DO NOT OPEN until you have
                             labeled everything, or you will bias your own labels.

The tweets come from the RUHSOLD rows of our validation split, so the final test
split stays untouched. Tweets already used in the hand-labeled batches are skipped.
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd

from rumod.data.agreement import sample_for_agreement
from rumod.data.clean import make_dedupe_key
from rumod.data.schema import REPO_ROOT

VALIDATION_PATH = REPO_ROOT / "data" / "processed" / "validation.csv"
REAL_DIR = REPO_ROOT / "data" / "custom" / "real_labeled"
OUT_DIR = REPO_ROOT / "data" / "custom" / "agreement"
SECONDS_PER_MESSAGE = 12


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample tweets for the agreement study")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    to_label_path = OUT_DIR / "agreement_to_label.csv"
    key_path = OUT_DIR / "agreement_key.csv"
    if to_label_path.exists() and not args.force:
        raise SystemExit(f"{to_label_path.name} already exists. Use --force to replace it "
                         "(this would discard any labeling you have done).")
    if not VALIDATION_PATH.exists():
        raise SystemExit(f"{VALIDATION_PATH} not found. Run split.py first.")

    validation = pd.read_csv(VALIDATION_PATH, keep_default_na=False)
    official = validation[validation["source"] == "ruhsold"][["text", "label"]]

    used: set[str] = set()
    for path in glob.glob(str(REAL_DIR / "batch_*_to_label.csv")):
        used |= {make_dedupe_key(t) for t in pd.read_csv(path, keep_default_na=False)["text"]}

    to_label, key = sample_for_agreement(official, args.per_class, args.seed, used)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    to_label.to_csv(to_label_path, index=False)
    key.to_csv(key_path, index=False)

    minutes = len(to_label) * SECONDS_PER_MESSAGE / 60
    print(f"{len(to_label)} tweets sampled, {args.per_class} per class "
          f"(about {minutes:.0f} minutes of labeling)")
    print(f"label this file : {to_label_path.relative_to(REPO_ROOT)}")
    print(f"key (do not open): {key_path.relative_to(REPO_ROOT)}")
    print("In the labeling tool, export as agreement_labeled.csv and move it next to the key.")


if __name__ == "__main__":
    main()