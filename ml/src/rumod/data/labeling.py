"""Pick the messages that will be labeled by hand for the real evaluation set.

Pool: the 2,002 tweets in the official RUHSOLD test split. They have no labels,
so no model was ever trained on them. Tweets that also appear in our cleaned
training data (or were sampled in an earlier batch) are removed first.

Each batch mixes two kinds of rows, recorded in the `selection` column:

  random     drawn uniformly from the pool. These give an unbiased picture of how
             often each class occurs and are the rows to use for prevalence-
             weighted numbers.
  candidate  the pool rows the current model finds most likely to belong to one
             of the rare classes. Without them a purely random batch would hold
             fewer than 50 examples of the smaller classes.

Because candidates were chosen using the model's own scores, they are likely to
make the model look better on recall than it really is. Report metrics on the
random rows alone as well as on everything.

The labeler never sees the model's predictions (they are written to a separate
file), so they cannot anchor the labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rumod.data.clean import light_clean, make_dedupe_key

# Rarest first, so the scarce classes get first pick of the candidates.
DEFAULT_CANDIDATE_CLASSES = ("profane", "religious_hate", "sexual_hate", "abusive")


def build_pool(raw: pd.DataFrame, exclude_keys: set[str]) -> pd.DataFrame:
    """Clean the unlabeled tweets and drop duplicates and already-used messages."""
    pool = pd.DataFrame({"text": raw["tweet"].astype(str).map(light_clean)})
    pool["dedupe_key"] = pool["text"].map(make_dedupe_key)
    pool = pool[pool["dedupe_key"] != ""]
    pool = pool.drop_duplicates("dedupe_key")
    pool = pool[~pool["dedupe_key"].isin(exclude_keys)]
    return pool.reset_index(drop=True)


def scores_frame(results: list[dict], labels: list[str]) -> pd.DataFrame:
    """Turn ModerationModel.predict() output into one row of scores per message."""
    rows = []
    for result in results:
        row = {label: result["scores"].get(label, 0.0) for label in labels}
        row["model_label"] = result["label"]
        row["model_confidence"] = result["confidence"]
        rows.append(row)
    return pd.DataFrame(rows)


def select_rows(
    pool: pd.DataFrame,
    scores: pd.DataFrame,
    n_random: int,
    per_class: int,
    seed: int = 42,
    classes: tuple[str, ...] = DEFAULT_CANDIDATE_CLASSES,
) -> pd.DataFrame:
    """Random rows plus model-picked candidates for each rare class, shuffled."""
    if len(pool) != len(scores):
        raise ValueError("pool and scores must have the same number of rows")

    rng = np.random.default_rng(seed)
    n_random = min(n_random, len(pool))
    random_index = rng.choice(len(pool), size=n_random, replace=False)

    chosen = {int(i): "random" for i in random_index}
    for label in classes:
        if label not in scores.columns:
            raise ValueError(f"no scores for class '{label}'")
        order = scores[label].to_numpy().argsort(kind="stable")[::-1]  # highest first
        taken = 0
        for i in order:
            if taken == per_class:
                break
            if int(i) not in chosen:
                chosen[int(i)] = f"candidate:{label}"
                taken += 1

    picked = pool.loc[list(chosen)].copy()
    picked["selection"] = pd.Series(chosen)
    extra = scores.loc[list(chosen), ["model_label", "model_confidence"]]
    picked = picked.join(extra)
    # shuffle so the order does not reveal how a row was picked
    return picked.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def assign_ids(rows: pd.DataFrame, batch: int) -> pd.DataFrame:
    rows = rows.copy()
    rows.insert(0, "id", [f"b{batch:02d}-{i + 1:04d}" for i in range(len(rows))])
    return rows