"""Create train / validation / test splits from data/interim/cleaned.csv.

Output (data/processed/)
------------------------
train.csv, validation.csv, test.csv   unified columns + dedupe_key + label_id
split_report.json                     row counts per split, source and label

Design
------
* The official RUHSOLD validation/test splits are unusable (see schema.py), so
  every split is built here, with a fixed seed.
* Each source is split on its own and stratified by label, so every split keeps
  the same class mix as its source. Every row keeps its `source` column, which
  lets us report metrics per source (RUHSOLD tweets vs RUACD comments) instead
  of one number dominated by the larger source.
* RUACD is ~95% of the data but only covers clean/abusive. To stop it from
  drowning out the RUHSOLD classes, only its TRAIN part is capped per class
  (--ruacd-cap). Validation and test are never capped.
* Rows marked is_synthetic only ever go to train. Our own hand-labeled
  "real" rows only ever go to validation/test.
* The cleaned data has unique dedupe_keys, so no message appears in two splits.
  The script asserts this, and also checks nothing synthetic leaked into
  validation/test.

Usage (from the repo root)
--------------------------
    python ml/src/rumod/data/split.py
    python ml/src/rumod/data/split.py --seed 42 --ruacd-cap 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running this file directly before the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rumod.data.schema import REPO_ROOT, load_labels  # noqa: E402

INTERIM_PATH = REPO_ROOT / "data" / "interim" / "cleaned.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# (train, validation, test) fractions per source.
SOURCE_FRACTIONS = {
    "ruhsold": (0.70, 0.15, 0.15),
    "ruacd": (0.90, 0.05, 0.05),
    "synthetic": (1.00, 0.00, 0.00),  # training only
    "real": (0.00, 0.50, 0.50),       # evaluation only
}


def stratified_three_way(
    df: pd.DataFrame, fractions: tuple[float, float, float], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one source into train/validation/test, stratified by label."""
    train_f, val_f, test_f = fractions
    empty = df.iloc[0:0]

    if val_f == 0 and test_f == 0:
        return df, empty, empty

    if train_f == 0:
        rest = df
    else:
        train, rest = train_test_split(
            df, test_size=val_f + test_f, stratify=df["label"], random_state=seed
        )

    if val_f == 0:
        val, test = empty, rest
    elif test_f == 0:
        val, test = rest, empty
    else:
        val, test = train_test_split(
            rest, test_size=test_f / (val_f + test_f), stratify=rest["label"], random_state=seed
        )

    if train_f == 0:
        train = empty
    return train, val, test


def cap_per_label(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Keep at most `cap` rows per label (random, reproducible)."""
    if cap <= 0:
        return df
    # Iterate over the groups instead of groupby().apply(): newer pandas drops
    # the grouping column inside apply(), which would silently erase the labels.
    parts = [
        group.sample(n=min(len(group), cap), random_state=seed)
        for _, group in df.groupby("label")
    ]
    return pd.concat(parts).sort_index()


def build_splits(df: pd.DataFrame, seed: int, ruacd_cap: int) -> dict[str, pd.DataFrame]:
    parts = {"train": [], "validation": [], "test": []}

    for source, group in df.groupby("source"):
        if source not in SOURCE_FRACTIONS:
            raise ValueError(f"No split rule for source '{source}'. Add it to SOURCE_FRACTIONS.")

        train, val, test = stratified_three_way(group, SOURCE_FRACTIONS[source], seed)
        if source == "ruacd":
            train = cap_per_label(train, ruacd_cap, seed)

        parts["train"].append(train)
        parts["validation"].append(val)
        parts["test"].append(test)

    return {
        name: pd.concat(frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for name, frames in parts.items()
    }


def check_splits(
    splits: dict[str, pd.DataFrame], enabled_labels: list[str], all_labels: list[str] | None = None
) -> None:
    """Fail loudly if the splits would make evaluation untrustworthy."""
    # 0. every row has a valid label and text
    valid = set(all_labels or enabled_labels)
    for name, df in splits.items():
        assert df["label"].isin(valid).all(), f"{name} has missing or unknown labels"
        assert (df["text"].astype(str).str.len() > 0).all(), f"{name} has empty text"

    # 1. no message in two splits
    keys = {name: set(df["dedupe_key"]) for name, df in splits.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = keys[a] & keys[b]
        assert not overlap, f"{len(overlap)} messages appear in both {a} and {b}"

    # 2. no synthetic rows in evaluation splits
    for name in ("validation", "test"):
        assert not splits[name]["is_synthetic"].any(), f"synthetic rows leaked into {name}"

    # 3. every enabled label is present in every split (for the RUHSOLD slice)
    for name, df in splits.items():
        present = set(df.loc[df["source"] == "ruhsold", "label"])
        missing = set(enabled_labels) - present
        if missing and (df["source"] == "ruhsold").any():
            raise AssertionError(f"{name} has no RUHSOLD rows for: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/validation/test splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ruacd-cap", type=int, default=10_000,
                        help="max RUACD rows per label in TRAIN (0 = no cap)")
    args = parser.parse_args()

    if not INTERIM_PATH.exists():
        sys.exit(f"{INTERIM_PATH} not found. Run clean.py first.")

    labels = load_labels()
    df = pd.read_csv(INTERIM_PATH, keep_default_na=False)
    df["is_synthetic"] = df["is_synthetic"].astype(bool)
    df["is_hard_negative"] = df["is_hard_negative"].astype(bool)

    unknown = set(df["label"]) - set(labels["name_to_id"])
    if unknown:
        sys.exit(f"Labels not in labels.yaml: {sorted(unknown)}")

    splits = build_splits(df, seed=args.seed, ruacd_cap=args.ruacd_cap)
    check_splits(splits, labels["enabled"], list(labels["name_to_id"]))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "ruacd_cap_per_label_in_train": args.ruacd_cap, "splits": {}}
    for name, part in splits.items():
        part = part.assign(label_id=part["label"].map(labels["name_to_id"]))
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        report["splits"][name] = {
            "rows": int(len(part)),
            "by_source": {k: int(v) for k, v in part["source"].value_counts().items()},
            "by_label": {k: int(v) for k, v in part["label"].value_counts().items()},
            "by_source_and_label": {
                f"{src}/{lab}": int(n)
                for (src, lab), n in part.groupby(["source", "label"]).size().items()
            },
        }
    (PROCESSED_DIR / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, info in report["splits"].items():
        print(f"\n{name}: {info['rows']:,} rows   by source {info['by_source']}")
        for label in labels["enabled"]:
            print(f"    {label:<15} {info['by_label'].get(label, 0):>7,}")
    out_dir = PROCESSED_DIR.relative_to(REPO_ROOT)
    print(f"\nWritten to {out_dir}/ (train.csv, validation.csv, test.csv)")


if __name__ == "__main__":
    main()