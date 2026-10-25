import numpy as np
import pandas as pd
import pytest

from rumod.evaluation.metrics import (
    classification_report,
    evaluate_by_source,
    toxic_vs_clean,
)


def test_perfect_predictions():
    y = ["clean", "abusive", "clean", "abusive"]
    rep = classification_report(y, y, ["clean", "abusive"], n_boot=50)
    assert rep["accuracy"] == 1.0
    assert rep["macro_f1"] == 1.0
    assert rep["per_class"]["abusive"]["f1"] == 1.0


def test_hand_computed_example():
    # true  a a b b
    # pred  a b b b
    # a: precision 1.0, recall 0.5, F1 2/3 | b: precision 2/3, recall 1.0, F1 0.8
    rep = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"], n_boot=50)
    assert rep["accuracy"] == pytest.approx(0.75)
    assert rep["per_class"]["a"]["precision"] == pytest.approx(1.0)
    assert rep["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert rep["per_class"]["a"]["f1"] == pytest.approx(2 / 3)
    assert rep["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert rep["per_class"]["b"]["f1"] == pytest.approx(0.8)
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert rep["confusion"]["matrix"] == [[1, 1], [0, 2]]


def test_class_without_support_is_excluded_from_macro():
    rep = classification_report(["a", "a", "b"], ["a", "a", "b"], ["a", "b", "c"], n_boot=50)
    assert rep["per_class"]["c"]["f1"] is None
    assert rep["per_class"]["c"]["support"] == 0
    assert rep["macro_f1"] == 1.0


def test_bootstrap_ci_is_ordered_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y_true = rng.choice(["a", "b"], size=300)
    y_pred = np.where(rng.random(300) < 0.8, y_true, np.where(y_true == "a", "b", "a"))
    first = classification_report(y_true, y_pred, ["a", "b"], n_boot=200, seed=1)
    again = classification_report(y_true, y_pred, ["a", "b"], n_boot=200, seed=1)
    lo, hi = first["macro_f1_ci"]
    assert lo <= first["macro_f1"] <= hi
    assert first["macro_f1_ci"] == again["macro_f1_ci"]


def test_toxic_vs_clean():
    res = toxic_vs_clean(["clean", "abusive", "profane", "clean"],
                         ["clean", "profane", "clean", "abusive"])
    # true toxic: rows 2,3 | predicted toxic: rows 2,4 | hit: row 2
    assert res["precision"] == pytest.approx(0.5)
    assert res["recall"] == pytest.approx(0.5)
    assert res["support"] == 2
    # clean rows: 1 and 4; row 4 was flagged -> 1 of 2
    assert res["false_alarm_rate"] == pytest.approx(0.5)


def test_ruacd_predictions_are_collapsed_to_binary():
    df = pd.DataFrame(
        {
            "label": ["abusive", "abusive", "clean", "clean"],
            "source": ["ruacd"] * 4,
        }
    )
    labels = ["clean", "abusive", "religious_hate"]
    # "religious_hate" on an RUACD-abusive row is still a correct "abusive" call
    pred = ["religious_hate", "abusive", "clean", "clean"]
    rep = evaluate_by_source(df, pred, labels, n_boot=50)["ruacd"]
    assert rep["collapsed_to_binary"] is True
    assert rep["per_class"]["abusive"]["recall"] == 1.0
    assert rep["accuracy"] == 1.0


def test_sources_are_reported_separately():
    df = pd.DataFrame(
        {
            "label": ["clean", "abusive", "clean", "abusive"],
            "source": ["ruhsold", "ruhsold", "ruacd", "ruacd"],
        }
    )
    pred = ["clean", "clean", "clean", "abusive"]
    out = evaluate_by_source(df, pred, ["clean", "abusive"], n_boot=50)
    assert set(out) == {"ruhsold", "ruacd"}
    assert out["ruhsold"]["accuracy"] == 0.5
    assert out["ruacd"]["accuracy"] == 1.0