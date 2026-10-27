"""Score a model on the hand-labeled set: all rows, and random rows only.

The set is a TEST set. Do not tune on it. Pick the training setup and thresholds
on the RUHSOLD/RUACD validation data, then look here once.

Candidate rows were picked with an earlier model's scores, which tends to flatter
a model, so the random-only numbers are the fair ones. They are noisier because
there are fewer rows, and the confidence intervals show how much.
"""
from __future__ import annotations

import pandas as pd

from rumod.evaluation.metrics import evaluate_by_source


def score_real_set(model, rows: pd.DataFrame, n_boot: int = 1000, progress: bool = False) -> dict:
    """Return {"predictions": [...], "all": report, "random_only": report | None}."""
    rows = rows.reset_index(drop=True)
    results = model.predict(rows["text"], progress=progress)
    predicted = [r["label"] for r in results]
    labels = sorted(set(model.labels) | set(rows["label"]))

    def report(mask, name: str):
        subset = rows[mask].assign(source=name).reset_index(drop=True)
        if subset.empty:
            return None
        picked = [p for p, keep in zip(predicted, mask, strict=True) if keep]
        return evaluate_by_source(subset, picked, labels, n_boot=n_boot, progress=progress)[name]

    everything = pd.Series(True, index=rows.index)
    return {
        "predictions": results,
        "all": report(everything, "real_all"),
        "random_only": report(rows["selection"] == "random", "real_random_only"),
    }


def error_table(rows: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    """Misclassified rows with the model's guess, for reading through by hand."""
    table = rows.reset_index(drop=True)[["id", "text", "label", "selection", "notes"]].copy()
    table["predicted"] = [r["label"] for r in results]
    table["confidence"] = [round(r["confidence"], 3) for r in results]
    wrong = table[table["label"] != table["predicted"]]
    return wrong.sort_values("confidence", ascending=False).reset_index(drop=True)