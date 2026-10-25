"""Evaluation metrics: per-class scores, per-source reporting, bootstrap CIs.

Why per source: about 87% of validation/test rows come from RUACD, so one
overall number would mostly measure RUACD. We report every source on its own.

RUACD only knows "abusive" vs "clean", and its "abusive" is broader than our
`abusive` class. Predictions on RUACD rows are therefore collapsed: any
non-clean prediction counts as "abusive".

The hate classes have only ~70-90 rows in validation/test, so every F1 comes
with a bootstrap 95% confidence interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLEAN = "clean"
COLLAPSE_TO_BINARY_SOURCES = ("ruacd",)


def _prf(true_idx: np.ndarray, pred_idx: np.ndarray, k: int):
    """Precision, recall, F1 and support per class from integer labels."""
    hits = true_idx[true_idx == pred_idx]
    tp = np.bincount(hits, minlength=k).astype(float)
    predicted = np.bincount(pred_idx, minlength=k).astype(float)
    support = np.bincount(true_idx, minlength=k).astype(float)

    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
    return precision, recall, f1, support


def _macro(f1: np.ndarray, support: np.ndarray) -> float:
    """Macro F1 over the classes that actually occur in y_true."""
    present = support > 0
    return float(f1[present].mean()) if present.any() else float("nan")


def classification_report(
    y_true, y_pred, labels: list[str], n_boot: int = 1000, seed: int = 42
) -> dict:
    """Per-class precision/recall/F1 with bootstrap CIs, plus a confusion matrix."""
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    t = np.array([index[x] for x in y_true])
    p = np.array([index[x] for x in y_pred])
    n = len(t)

    precision, recall, f1, support = _prf(t, p, k)
    macro = _macro(f1, support)

    rng = np.random.default_rng(seed)
    f1_boot = np.empty((n_boot, k))
    macro_boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.integers(0, n, n)
        _, _, f1_b, support_b = _prf(t[sample], p[sample], k)
        f1_boot[b] = f1_b
        macro_boot[b] = _macro(f1_b, support_b)

    def ci(values: np.ndarray) -> list[float]:
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    confusion = np.bincount(t * k + p, minlength=k * k).reshape(k, k)

    per_class = {}
    for label, i in index.items():
        has_support = support[i] > 0
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]) if has_support else None,
            "f1_ci": ci(f1_boot[:, i]) if has_support else None,
            "support": int(support[i]),
        }

    return {
        "n": int(n),
        "accuracy": float((t == p).mean()),
        "macro_f1": macro,
        "macro_f1_ci": ci(macro_boot),
        "per_class": per_class,
        "confusion": {"labels": list(labels), "matrix": confusion.tolist()},
    }


def toxic_vs_clean(y_true, y_pred) -> dict:
    """Moderation view: is the message flagged at all (any non-clean label)?"""
    t = np.asarray(y_true) != CLEAN
    p = np.asarray(y_pred) != CLEAN
    tp = float((t & p).sum())
    precision = tp / p.sum() if p.sum() else 0.0
    recall = tp / t.sum() if t.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": int(t.sum())}


def evaluate_by_source(
    df: pd.DataFrame, y_pred, labels: list[str], n_boot: int = 1000, seed: int = 42
) -> dict:
    """Report every source separately. `df` needs `label` and `source` columns."""
    predictions = pd.Series(np.asarray(y_pred), index=df.index)
    results = {}
    for source, part in df.groupby("source"):
        y_true = part["label"].to_numpy()
        y_hat = predictions.loc[part.index].to_numpy()

        collapsed = source in COLLAPSE_TO_BINARY_SOURCES
        if collapsed:
            y_hat = np.where(y_hat == CLEAN, CLEAN, "abusive")
            source_labels = [CLEAN, "abusive"]
        else:
            source_labels = list(labels)

        report = classification_report(y_true, y_hat, source_labels, n_boot=n_boot, seed=seed)
        report["toxic_vs_clean"] = toxic_vs_clean(y_true, y_hat)
        report["collapsed_to_binary"] = collapsed
        results[source] = report
    return results


def format_report(results: dict) -> str:
    """Readable text table of an evaluate_by_source() result."""
    lines = []
    for source, rep in results.items():
        note = "  (predictions collapsed to clean/abusive)" if rep["collapsed_to_binary"] else ""
        lo, hi = rep["macro_f1_ci"]
        lines.append(f"\n=== {source}: n={rep['n']:,}{note}")
        lines.append(f"accuracy {rep['accuracy']:.3f}   macro-F1 {rep['macro_f1']:.3f} "
                     f"[{lo:.3f}, {hi:.3f}]")
        header = f"{'class':<16}{'precision':>10}{'recall':>8}{'F1':>7}  {'95% CI':<15}{'n':>6}"
        lines.append(header)
        for label, c in rep["per_class"].items():
            if c["support"] == 0:
                continue
            ci = f"[{c['f1_ci'][0]:.2f}, {c['f1_ci'][1]:.2f}]"
            lines.append(f"{label:<16}{c['precision']:>10.3f}{c['recall']:>8.3f}"
                         f"{c['f1']:>7.3f}  {ci:<15}{c['support']:>6}")
        tv = rep["toxic_vs_clean"]
        lines.append(f"toxic-vs-clean: precision {tv['precision']:.3f}  "
                     f"recall {tv['recall']:.3f}  F1 {tv['f1']:.3f}")
    return "\n".join(lines)