"""Baseline classifier: character n-gram TF-IDF + Logistic Regression.

Simple on purpose. It sets the bar that any later model has to beat, and
character n-grams cope well with Roman Urdu spelling variation ("kamina",
"kameena", "kamine").

This module only builds and trains the sklearn pipeline. Text normalization,
saving, loading and prediction all live in rumod.models.predict so training and
serving cannot drift apart.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEFAULT_PARAMS = {
    "ngram_range": (2, 5),
    "min_df": 2,
    "max_features": 300_000,
    "C": 4.0,
    "class_weight": "balanced",
    "max_iter": 1000,
}


def sqrt_balanced_weights(labels) -> dict[str, float]:
    """Square root of sklearn's "balanced" weights: a gentler boost for small classes.

    Fully balanced weights can over-flag, because a class with 300 rows gets a
    weight about 40x that of a class with 12,000 rows. The square root keeps the
    direction but halves the gap on a log scale.
    """
    labels = list(labels)
    classes = sorted(set(labels))
    total, k = len(labels), len(classes)
    return {c: float((total / (k * labels.count(c))) ** 0.5) for c in classes}


def build_pipeline(**overrides) -> Pipeline:
    """TF-IDF over character n-grams inside words, then multinomial LR.

    class_weight="balanced" counters the large gap between the hate classes
    (a few hundred training rows) and clean/abusive (about 10,000+ each).
    """
    params = {**DEFAULT_PARAMS, **overrides}
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(params["ngram_range"]),
                    min_df=params["min_df"],
                    max_features=params["max_features"],
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=params["C"],
                    class_weight=params["class_weight"],
                    max_iter=params["max_iter"],
                ),
            ),
        ]
    )