"""Sample tweets for the annotation-agreement study.

Question: does our labeling guide draw the class boundaries where RUHSOLD's
annotators did? To measure it, we label tweets that already have an official
RUHSOLD label, without seeing that label, and compare.

Pool: the RUHSOLD rows of our own validation split (official labels known, and
the final test split stays untouched). Sampling is stratified by official label,
so the rare classes get enough rows for a per-class read.

Two files come out: one to label (id, text, empty label columns) and a separate
key (id, official_label) that must stay out of sight during labeling.
"""
from __future__ import annotations

import pandas as pd

from rumod.data.clean import make_dedupe_key
from rumod.data.real_set import EVAL_LABELS


def sample_for_agreement(
    official: pd.DataFrame,
    per_class: int = 30,
    seed: int = 42,
    exclude_keys: frozenset[str] | set[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (to_label, key). `official` needs `text` and `label` columns."""
    pool = official[official["label"].isin(EVAL_LABELS)].copy()
    pool["dedupe_key"] = pool["text"].map(make_dedupe_key)
    pool = pool[~pool["dedupe_key"].isin(exclude_keys)].drop_duplicates("dedupe_key")

    parts = []
    for label in EVAL_LABELS:
        group = pool[pool["label"] == label]
        parts.append(group.sample(n=min(per_class, len(group)), random_state=seed))
    picked = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    picked.insert(0, "id", [f"ag-{i + 1:04d}" for i in range(len(picked))])

    to_label = picked[["id", "text"]].assign(label="", is_hard_negative="", notes="")
    key = picked[["id", "label"]].rename(columns={"label": "official_label"})
    return to_label, key