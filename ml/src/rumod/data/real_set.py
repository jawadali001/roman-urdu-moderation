"""Load, check and summarize the hand-labeled evaluation set.

Files live in data/custom/real_labeled/ (gitignored, they contain tweet text):

    batch_NN_labeled.csv        exported from scripts/label_tool.html
                                columns: id, text, label, is_hard_negative, notes
    batch_NN_sampling_meta.csv  written by sample_for_labeling.py
                                columns: id, selection, model_label, model_confidence

`selection` says whether a row was drawn at random or picked as a candidate for a
rare class. Random rows give unbiased numbers; candidates flatter the model.

Rows labeled `unsure` or `unusable` are kept in the file but never scored.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from rumod.data.clean import make_dedupe_key
from rumod.data.schema import REPO_ROOT, UNIFIED_COLUMNS

REAL_DIR = REPO_ROOT / "data" / "custom" / "real_labeled"
EVAL_LABELS = ("clean", "abusive", "profane", "religious_hate", "sexual_hate")
ADMIN_LABELS = ("unsure", "unusable")
TARGET_PER_CLASS = 50
MAX_UNSURE_SHARE = 0.05


def load_labeled_batches(folder: Path = REAL_DIR) -> pd.DataFrame:
    """Read every batch_NN_labeled.csv and attach its sampling information."""
    paths = sorted(Path(folder).glob("batch_*_labeled.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No batch_NN_labeled.csv found in {folder}. Export your labels from the "
            "labeling tool and move the file there."
        )

    frames = []
    for path in paths:
        batch = int(re.search(r"batch_(\d+)_labeled", path.name).group(1))
        df = pd.read_csv(path, keep_default_na=False, dtype=str)
        missing = {"id", "text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for column in ("is_hard_negative", "notes"):
            if column not in df.columns:
                df[column] = ""

        meta_path = Path(folder) / f"batch_{batch:02d}_sampling_meta.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path, keep_default_na=False, dtype=str)
            df = df.merge(
                meta[["id", "selection", "model_label", "model_confidence"]], on="id", how="left"
            )
            df["selection"] = df["selection"].fillna("unknown")
        else:
            df["selection"], df["model_label"], df["model_confidence"] = "unknown", "", ""

        df["batch"] = batch
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["label"] = out["label"].str.strip()
    out["is_hard_negative"] = out["is_hard_negative"].str.strip().str.lower().eq("yes")
    out["dedupe_key"] = out["text"].map(make_dedupe_key)
    return out


def validate(df: pd.DataFrame) -> list[str]:
    """Problems worth fixing before scoring. An empty list means the set is clean."""
    problems = []
    unlabeled = int((df["label"] == "").sum())
    if unlabeled:
        problems.append(f"{unlabeled} rows have no label yet (they will be ignored)")

    known = set(EVAL_LABELS) | set(ADMIN_LABELS) | {""}
    unknown = sorted(set(df["label"]) - known)
    if unknown:
        problems.append(f"unknown labels: {unknown}")

    duplicate_ids = int(df["id"].duplicated().sum())
    if duplicate_ids:
        problems.append(f"{duplicate_ids} duplicate ids")

    labeled = df[df["label"] != ""]
    duplicate_texts = int(labeled["dedupe_key"].duplicated().sum())
    if duplicate_texts:
        problems.append(f"{duplicate_texts} labeled messages repeat an earlier one")

    if len(labeled) and (labeled["label"] == "unsure").mean() > MAX_UNSURE_SHARE:
        share = (labeled["label"] == "unsure").mean()
        problems.append(f"{share:.1%} of messages are 'unsure' (the guide asks for under 5%)")

    if (df["selection"] == "unknown").any():
        problems.append("some rows have no sampling information, so random-only numbers "
                        "cannot be computed for them")
    return problems


def summarize(df: pd.DataFrame, target: int = TARGET_PER_CLASS) -> dict:
    labeled = df[df["label"] != ""]
    counts = {label: int((labeled["label"] == label).sum())
              for label in (*EVAL_LABELS, *ADMIN_LABELS)}
    random_rows = labeled[labeled["selection"] == "random"]
    clean_rows = labeled[labeled["label"] == "clean"]
    return {
        "rows": int(len(df)),
        "labeled": int(len(labeled)),
        "unlabeled": int(len(df) - len(labeled)),
        "counts": counts,
        "counts_random_only": {label: int((random_rows["label"] == label).sum())
                               for label in EVAL_LABELS},
        "hard_negatives": int(clean_rows["is_hard_negative"].sum()),
        "shortfall": {label: target - counts[label]
                      for label in EVAL_LABELS if counts[label] < target},
    }


def to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Scoreable rows in the project's unified schema (source = real)."""
    rows = df[df["label"].isin(EVAL_LABELS)].copy()
    rows["source"] = "real"
    rows["origin_split"] = "all"
    rows["is_synthetic"] = False
    return rows[[*UNIFIED_COLUMNS, "dedupe_key", "selection", "id", "notes"]].reset_index(drop=True)