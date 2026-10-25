"""Unified schema and label mapping for all source datasets.

Every source (RUHSOLD, RUACD, later our own data) is converted into the same
columns so the rest of the pipeline never needs to know where a row came from.

Unified columns
---------------
text             the message
label            class name, must exist in configs/labels.yaml
source           ruhsold | ruacd | (later) synthetic | real
origin_split     split the row came from in its source (train/validation/test/all)
is_synthetic     True for generated rows (training only, never test)
is_hard_negative True for clean messages that look risky (for the clean class)

Only labels are mapped here. Cleaning and de-duplication live in clean.py and
splitting lives in split.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

# ml/src/rumod/data/schema.py -> parents[4] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_PATH = REPO_ROOT / "configs" / "labels.yaml"

UNIFIED_COLUMNS = [
    "text",
    "label",
    "source",
    "origin_split",
    "is_synthetic",
    "is_hard_negative",
]

# RUHSOLD fine-grained class names (as published) -> our label names.
# The mapping is done by NAME, never by id: in RUHSOLD, id 1 is "Normal" and
# id 0 is "Abusive/Offensive".
RUHSOLD_FINE_TO_OURS = {
    "Abusive/Offensive": "abusive",
    "Normal": "clean",
    "Religious Hate": "religious_hate",
    "Sexism": "sexual_hate",
    "Profane/Untargeted": "profane",
}

# RUACD has a binary `rating` column: TRUE = abusive, FALSE = not abusive.
RUACD_RATING_TO_OURS = {True: "abusive", False: "clean"}


def load_labels(path: Path = LABELS_PATH) -> dict:
    """Read configs/labels.yaml and return names, ids and the enabled subset."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels = config["labels"]
    return {
        "name_to_id": {l["name"]: l["id"] for l in labels},
        "enabled": [l["name"] for l in labels if l["enabled"]],
        "default": config["default_label"],
        "config": config,
    }


def _finish(df: pd.DataFrame, source: str, origin_split: str) -> pd.DataFrame:
    """Add the bookkeeping columns and enforce the unified column order."""
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["source"] = source
    df["origin_split"] = origin_split
    df["is_synthetic"] = False
    df["is_hard_negative"] = False
    return df[UNIFIED_COLUMNS]


def load_ruhsold_fine(split: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load one RUHSOLD fine-grained split in the unified schema.

    Label ids are decoded with the label_names that download.py stored in
    manifest.json, so nothing depends on a guessed id order.
    """
    folder = raw_dir / "ruhsold"
    file_name = f"Fine_Grained_{split}.csv"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    label_names = manifest[file_name]["label_names"]

    unknown = set(label_names) - set(RUHSOLD_FINE_TO_OURS)
    if unknown:
        raise ValueError(f"RUHSOLD has labels we have not mapped: {sorted(unknown)}")

    raw = pd.read_csv(folder / file_name)

    # Known problems in the Hugging Face copy (found by check_splits.py and
    # check_labels.py): the official test split has no labels at all, and the
    # fine-grained validation split is an exact copy of train. Using either
    # would silently break evaluation, so refuse loudly instead.
    if raw["label"].isna().any():
        raise ValueError(
            f"RUHSOLD {split} split has {int(raw['label'].isna().sum()):,} unlabeled rows "
            f"out of {len(raw):,}. It cannot be used for training or evaluation."
        )
    if split != "train":
        train = pd.read_csv(folder / "Fine_Grained_train.csv")
        if raw.equals(train):
            raise ValueError(
                f"RUHSOLD {split} split is an exact copy of train. "
                "Build validation from train in split.py instead."
            )

    out = pd.DataFrame(
        {
            "text": raw["tweet"],
            "label": raw["label"].map(lambda i: RUHSOLD_FINE_TO_OURS[label_names[i]]),
        }
    )
    return _finish(out, source="ruhsold", origin_split=split)


def load_ruacd(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load RUACD (binary abusive / not abusive) in the unified schema.

    RUACD has no official splits, so origin_split is "all". Its "abusive" class
    is broader than our `abusive` label (it can contain profanity or hate), so
    treat it as extra volume for the abusive-vs-clean boundary only.
    """
    raw = pd.read_csv(raw_dir / "ruacd" / "dataset.csv")
    rating = raw["rating"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    if rating.isna().any():
        raise ValueError("RUACD has rating values other than TRUE/FALSE")

    out = pd.DataFrame(
        {"text": raw["comment"], "label": rating.map(RUACD_RATING_TO_OURS)}
    )
    return _finish(out, source="ruacd", origin_split="all")


def validate_unified(df: pd.DataFrame, labels: dict | None = None) -> None:
    """Raise a clear error if a unified frame breaks the schema."""
    labels = labels or load_labels()

    missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    unknown = set(df["label"].unique()) - set(labels["name_to_id"])
    if unknown:
        raise ValueError(f"Labels not in labels.yaml: {sorted(unknown)}")

    if df["text"].isna().any() or (df["text"].str.len() == 0).any():
        raise ValueError("Empty or missing text found")


if __name__ == "__main__":
    labels = load_labels()
    print("enabled labels:", labels["enabled"])

    for split in ("train", "validation", "test"):
        try:
            df = load_ruhsold_fine(split)
        except ValueError as error:
            print(f"\nruhsold/{split}: SKIPPED - {error}")
            continue
        validate_unified(df, labels)
        print(f"\nruhsold/{split}: {len(df):,} rows")
        print(df["label"].value_counts().to_string())

    df = load_ruacd()
    validate_unified(df, labels)
    print(f"\nruacd: {len(df):,} rows")
    print(df["label"].value_counts().to_string())