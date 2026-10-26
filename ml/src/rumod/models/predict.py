"""Training wrapper, saving/loading and prediction, shared by scripts and the API.

ModerationModel is the ONLY place that applies text normalization, for both
fitting and predicting, so training and serving always see identical text.
A saved model remembers the normalizer version it was trained with and refuses
to load if the code's version differs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from rumod.features.normalize import NORMALIZER_VERSION, normalize_batch
from rumod.models.baseline import DEFAULT_PARAMS, build_pipeline, sqrt_balanced_weights
from rumod.utils.progress import stage, track

BUNDLE_FORMAT_VERSION = 1


class ModelVersionError(ValueError):
    """The saved model does not match this version of the code."""


class ModerationModel:
    def __init__(self, pipeline, labels: list[str], default_label: str = "clean", meta=None):
        self.pipeline = pipeline
        self.labels = list(labels)
        self.default_label = default_label
        self.meta = meta or {}

    # ---- training -------------------------------------------------------
    @classmethod
    def fit(
        cls, texts, labels, default_label: str = "clean", progress: bool = False, **params
    ) -> ModerationModel:
        texts = list(texts)
        labels = list(labels)
        normalized = normalize_batch(
            track(texts, "normalizing text", total=len(texts), enabled=progress)
        )
        keep = [i for i, t in enumerate(normalized) if t]  # drop rows that normalize to ""
        kept_labels = [labels[i] for i in keep]
        if params.get("class_weight") == "sqrt":
            params = {**params, "class_weight": sqrt_balanced_weights(kept_labels)}
        pipeline = build_pipeline(**params)
        kept_texts = [normalized[i] for i in keep]
        # Same as pipeline.fit(), split in two so each slow step can be timed.
        with stage(f"building TF-IDF features ({len(kept_texts):,} texts)", enabled=progress):
            features = pipeline.named_steps["tfidf"].fit_transform(kept_texts)
        with stage("fitting classifier", enabled=progress):
            pipeline.named_steps["clf"].fit(features, kept_labels)

        classes = [str(c) for c in pipeline.classes_]
        meta = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "normalizer_version": NORMALIZER_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "params": {**DEFAULT_PARAMS, **params},
            "train_rows": len(keep),
            "train_rows_dropped_empty": len(texts) - len(keep),
            "train_label_counts": {c: labels.count(c) for c in classes},
        }
        return cls(pipeline, classes, default_label=default_label, meta=meta)

    # ---- prediction -----------------------------------------------------
    def predict(self, texts, progress: bool = False) -> list[dict]:
        """One result per input message.

        Messages with no Latin letters (empty after normalization) cannot be
        scored: they get the default label, confidence 0 and `empty=True`.
        """
        texts = list(texts)
        normalized = normalize_batch(
            track(texts, "normalizing text", total=len(texts), enabled=progress)
        )
        results: list[dict] = [
            {"label": self.default_label, "confidence": 0.0, "scores": {}, "empty": True}
            for _ in texts
        ]
        keep = [i for i, t in enumerate(normalized) if t]
        if keep:
            with stage(f"scoring {len(keep):,} messages", enabled=progress):
                proba = self.pipeline.predict_proba([normalized[i] for i in keep])
            for row, i in zip(proba, keep, strict=True):
                best = int(np.argmax(row))
                results[i] = {
                    "label": self.labels[best],
                    "confidence": float(row[best]),
                    "scores": {lab: float(p) for lab, p in zip(self.labels, row, strict=True)},
                    "empty": False,
                }
        return results

    def predict_labels(self, texts, progress: bool = False) -> list[str]:
        return [r["label"] for r in self.predict(texts, progress=progress)]

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "labels": self.labels,
                "default_label": self.default_label,
                "meta": self.meta,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModerationModel:
        bundle = joblib.load(path)
        meta = bundle["meta"]
        if meta.get("format_version") != BUNDLE_FORMAT_VERSION:
            raise ModelVersionError(
                f"Model file format {meta.get('format_version')} != code format "
                f"{BUNDLE_FORMAT_VERSION}. Retrain the model."
            )
        if meta.get("normalizer_version") != NORMALIZER_VERSION:
            raise ModelVersionError(
                f"Model was trained with normalizer v{meta.get('normalizer_version')} but the "
                f"code has v{NORMALIZER_VERSION}. Retrain the model."
            )
        return cls(
            bundle["pipeline"], bundle["labels"], bundle["default_label"], meta=meta
        )