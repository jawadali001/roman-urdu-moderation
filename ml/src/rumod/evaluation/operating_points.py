"""Threshold-free comparison and operating points for the toxic-vs-clean decision.

A classifier's flag/no-flag behaviour depends on a threshold. Instead of
comparing models at one arbitrary threshold, we compare:

* ROC-AUC of the toxic score (1 - P(clean)): quality independent of any threshold.
* Recall at a fixed false-alarm rate: "if at most 5% of clean messages may be
  flagged, how many toxic messages do we catch?" This is the number a product
  owner actually chooses.

Scores must come from held-out data. Thresholds picked here are exploratory:
the final threshold is chosen on the hand-labeled real set.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

CLEAN = "clean"
DEFAULT_TARGETS = (0.02, 0.05, 0.10)


def toxic_scores(model, texts) -> np.ndarray:
    """P(toxic) = 1 - P(clean) for every message. Empty messages score 0."""
    scores = []
    for result in model.predict(texts):
        if result["empty"]:
            scores.append(0.0)
        else:
            scores.append(1.0 - result["scores"][CLEAN])
    return np.asarray(scores)


def recall_at_false_alarm(y_toxic, score, target: float) -> dict:
    """Best recall reachable while flagging at most `target` of the clean messages."""
    fpr, tpr, thresholds = roc_curve(y_toxic, score)
    best = np.flatnonzero(fpr <= target)[-1]  # fpr and tpr both grow with the index
    threshold = float(thresholds[best])
    return {
        "target_false_alarm_rate": target,
        "false_alarm_rate": float(fpr[best]),
        "recall": float(tpr[best]),
        # flag a message when toxic score >= threshold (None means "flag nothing")
        "threshold": threshold if np.isfinite(threshold) else None,
    }


def operating_summary(y_toxic, score, targets=DEFAULT_TARGETS) -> dict:
    """AUC plus recall at each false-alarm target."""
    y_toxic = np.asarray(y_toxic, dtype=bool)
    return {
        "n": int(len(y_toxic)),
        "toxic_share": float(y_toxic.mean()),
        "auc": float(roc_auc_score(y_toxic, score)),
        "at": [recall_at_false_alarm(y_toxic, score, t) for t in targets],
    }