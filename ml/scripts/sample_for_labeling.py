"""Create a batch of messages for hand labeling.

Run from the repo root (after download.py, clean.py, split.py and
train_baseline.py):

    python ml/scripts/sample_for_labeling.py     # batch 1: 400 random + 60 per rare class
    python ml/scripts/sample_for_labeling.py --batch 2 --random 150 --per-class 40

Writes two files to data/custom/real_labeled/:

    batch_NN_to_label.csv       id, text + empty label columns. Open this in
                                scripts/label_tool.html.
    batch_NN_sampling_meta.csv  how each row was picked and what the model said.
                                Keep it away from the labeling session.

Messages used in earlier batches, and messages that also occur in the cleaned
training data, are never sampled again.
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd

from rumod.data.labeling import (
    assign_ids,
    build_pool,
    scores_frame,
    select_rows,
)
from rumod.data.schema import REPO_ROOT
from rumod.models.predict import ModerationModel

POOL_PATH = REPO_ROOT / "data" / "raw" / "ruhsold" / "Fine_Grained_test.csv"
CLEANED_PATH = REPO_ROOT / "data" / "interim" / "cleaned.csv"
MODEL_PATH = REPO_ROOT / "ml" / "artifacts" / "baseline.joblib"
OUT_DIR = REPO_ROOT / "data" / "custom" / "real_labeled"

SECONDS_PER_MESSAGE = 12  # rough guide for the time estimate


def used_keys() -> set[str]:
    """dedupe keys of everything that must not be sampled."""
    from rumod.data.clean import make_dedupe_key

    keys: set[str] = set()
    if CLEANED_PATH.exists():
        cleaned = pd.read_csv(CLEANED_PATH, keep_default_na=False, usecols=["dedupe_key"])
        keys |= set(cleaned["dedupe_key"])
    for path in glob.glob(str(OUT_DIR / "batch_*_to_label.csv")):
        previous = pd.read_csv(path, keep_default_na=False)
        keys |= {make_dedupe_key(text) for text in previous["text"]}
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample messages for hand labeling")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--random", type=int, default=400, help="uniformly random rows")
    parser.add_argument("--per-class", type=int, default=60,
                        help="model-picked candidates for each rare class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path, hint in [(POOL_PATH, "download.py"), (MODEL_PATH, "train_baseline.py")]:
        if not path.exists():
            raise SystemExit(f"{path} not found. Run {hint} first.")

    raw = pd.read_csv(POOL_PATH, keep_default_na=False)
    excluded = used_keys()
    pool = build_pool(raw, excluded)
    print(f"pool: {len(raw):,} tweets -> {len(pool):,} usable "
          f"(removed duplicates, rows without letters, and {len(excluded):,} keys already used)")

    model = ModerationModel.load(MODEL_PATH)
    results = model.predict(pool["text"], progress=True)
    scores = scores_frame(results, model.labels)

    picked = assign_ids(
        select_rows(pool, scores, args.random, args.per_class, args.seed), args.batch
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    to_label = picked[["id", "text"]].assign(label="", is_hard_negative="", notes="")
    meta = picked[["id", "selection", "model_label", "model_confidence"]]
    to_label_path = OUT_DIR / f"batch_{args.batch:02d}_to_label.csv"
    meta_path = OUT_DIR / f"batch_{args.batch:02d}_sampling_meta.csv"
    to_label.to_csv(to_label_path, index=False)
    meta.to_csv(meta_path, index=False)

    print(f"\nbatch {args.batch}: {len(picked):,} messages")
    for selection, count in picked["selection"].value_counts().items():
        print(f"  {selection:<26}{count:>5}")
    minutes = len(picked) * SECONDS_PER_MESSAGE / 60
    print(f"\nestimated labeling time: about {minutes:.0f} minutes "
          f"at {SECONDS_PER_MESSAGE}s per message")
    print(f"written: {to_label_path.relative_to(REPO_ROOT)}")
    print(f"         {meta_path.relative_to(REPO_ROOT)}  (keep away from the labeling session)")


if __name__ == "__main__":
    main()