"""Clean and de-duplicate the unified datasets.

Input : data/raw/ (through the loaders in schema.py)
Output: data/interim/cleaned.csv          cleaned rows in the unified schema + dedupe_key
        data/interim/cleaning_report.json how many rows each step removed

What this step does
-------------------
1. Light text cleaning: Unicode NFKC, remove zero-width characters, collapse
   whitespace. Case, punctuation and emoji are kept on purpose; model-specific
   normalization belongs in features/normalize.py so training and serving use
   the exact same function.
2. Drops rows with fewer than 2 Latin letters/digits (empty, emoji-only, or
   non-Roman script, which is out of scope for this model).
3. Builds a dedupe_key (lowercase, no URLs/mentions/punctuation, repeated
   letters squeezed) so near-identical messages are treated as duplicates.
4. De-duplicates inside each source. If the same message carries different
   labels in the same source, the row is ambiguous and all copies are dropped.
5. De-duplicates across sources. A message that is clean in one source and
   not-clean in another is dropped. Otherwise one row is kept, preferring
   RUHSOLD because its labels are finer-grained.

Usage (from the repo root)
--------------------------
    python ml/src/rumod/data/clean.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Allow running this file directly before the package is installed:
# ml/src/rumod/data/clean.py -> parents[2] is ml/src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rumod.data.schema import (  # noqa: E402
    REPO_ROOT,
    UNIFIED_COLUMNS,
    load_labels,
    load_ruacd,
    load_ruhsold_fine,
    validate_unified,
)

INTERIM_DIR = REPO_ROOT / "data" / "interim"

# When two sources hold the same message, keep the one listed first.
SOURCE_PRIORITY = {"ruhsold": 0, "ruacd": 1}

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REPEATS = re.compile(r"(.)\1{2,}")
_LATIN_ALNUM = re.compile(r"[A-Za-z0-9]")


def light_clean(text: str) -> str:
    """Structural clean-up only. Never changes wording, case or punctuation."""
    text = unicodedata.normalize("NFKC", str(text)).translate(_ZERO_WIDTH)
    return " ".join(text.split())


def make_dedupe_key(text: str) -> str:
    """Key that is equal for messages that differ only in case, punctuation,
    URLs, mentions or letter elongation ("yaaaar" vs "yaar")."""
    text = _URL.sub(" ", text.lower())
    text = _MENTION.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    text = _REPEATS.sub(r"\1\1", text)
    return " ".join(text.split())


def load_sources() -> tuple[pd.DataFrame, dict]:
    """Load every usable source. Unusable ones are skipped and reported."""
    frames, skipped = [], {}

    try:
        frames.append(load_ruhsold_fine("train"))
    except (ValueError, FileNotFoundError) as error:
        skipped["ruhsold/train"] = str(error)

    # The official RUHSOLD validation/test splits are unusable (copy of train /
    # unlabeled). We still try them so that a fixed upstream copy is picked up.
    for split in ("validation", "test"):
        try:
            frames.append(load_ruhsold_fine(split))
        except (ValueError, FileNotFoundError) as error:
            skipped[f"ruhsold/{split}"] = str(error)

    try:
        frames.append(load_ruacd())
    except (ValueError, FileNotFoundError) as error:
        skipped["ruacd"] = str(error)

    if not frames:
        raise SystemExit("No usable source found. Run download.py first.")
    return pd.concat(frames, ignore_index=True), skipped


def _counts(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
        "by_label": {k: int(v) for k, v in df["label"].value_counts().items()},
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Run all cleaning steps and return the cleaned frame plus a step log."""
    steps: list[dict] = []

    def log(name: str, before: int, after: int, note: str = "") -> None:
        steps.append({"step": name, "removed": before - after, "remaining": after, "note": note})

    df = df.copy()
    df["text"] = df["text"].map(light_clean)

    # 1. too short / no Latin content
    before = len(df)
    df = df[df["text"].map(lambda s: len(_LATIN_ALNUM.findall(s)) >= 2)]
    log("drop_empty_or_no_latin", before, len(df),
        "fewer than 2 Latin letters/digits (empty, emoji-only, non-Roman script)")

    df["dedupe_key"] = df["text"].map(make_dedupe_key)
    before = len(df)
    df = df[df["dedupe_key"] != ""]
    log("drop_empty_key", before, len(df), "nothing left after removing URLs/mentions/punctuation")

    # 2. inside each source: conflicting labels -> drop all copies
    before = len(df)
    n_labels = df.groupby(["source", "dedupe_key"])["label"].transform("nunique")
    df = df[n_labels == 1]
    log("drop_within_source_label_conflicts", before, len(df),
        "same message, different labels, same source")

    # 3. inside each source: exact/near duplicates -> keep first
    before = len(df)
    df = df.drop_duplicates(subset=["source", "dedupe_key"], keep="first")
    log("dedupe_within_source", before, len(df), "same message repeated in one source")

    # 4. across sources: clean vs not-clean disagreement -> drop all copies
    before = len(df)
    is_clean = df["label"].eq("clean")
    mixed = (
        is_clean.groupby(df["dedupe_key"]).transform("any")
        & (~is_clean).groupby(df["dedupe_key"]).transform("any")
    )
    df = df[~mixed]
    log("drop_cross_source_clean_vs_toxic", before, len(df),
        "clean in one source, not clean in another")

    # 5. across sources: keep one row, preferring the finer-grained source
    before = len(df)
    df = (
        df.assign(_priority=df["source"].map(SOURCE_PRIORITY).fillna(99))
        .sort_values("_priority", kind="stable")
        .drop_duplicates(subset=["dedupe_key"], keep="first")
        .drop(columns="_priority")
        .sort_index()
    )
    log("dedupe_across_sources", before, len(df), "kept the RUHSOLD row where both sources agree")

    return df[UNIFIED_COLUMNS + ["dedupe_key"]].reset_index(drop=True), steps


def main() -> None:
    labels = load_labels()
    raw, skipped = load_sources()
    validate_unified(raw, labels)

    cleaned, steps = clean(raw)
    validate_unified(cleaned, labels)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "cleaned.csv"
    cleaned.to_csv(out_path, index=False)

    report = {
        "skipped_sources": skipped,
        "before": _counts(raw),
        "steps": steps,
        "after": _counts(cleaned),
    }
    (INTERIM_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"before: {report['before']['rows']:,} rows")
    for s in steps:
        print(f"  {s['step']:<40} removed {s['removed']:>7,}   remaining {s['remaining']:>8,}")
    print(f"after : {report['after']['rows']:,} rows -> {out_path.relative_to(REPO_ROOT)}")
    print("\nby source:", report["after"]["by_source"])
    print("by label :", report["after"]["by_label"])
    if skipped:
        print("\nskipped sources:")
        for name, reason in skipped.items():
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()