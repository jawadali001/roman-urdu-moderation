import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from rumod.data.agreement import sample_for_agreement
from rumod.data.real_set import EVAL_LABELS
from rumod.evaluation.agreement import (
    agreement_report,
    confusion_matrix,
    format_agreement,
    kappa_from_matrix,
    split_comparable,
    wilson_interval,
)
from rumod.evaluation.metrics import classification_report, format_report

LABELS = list(EVAL_LABELS)
PRIOR = {"clean": 0.55, "abusive": 0.25, "profane": 0.06,
         "religious_hate": 0.07, "sexual_hate": 0.07}


def official_frame(per_class=40):
    rows = [(f"{label} tweet number {i}", label) for label in LABELS for i in range(per_class)]
    return pd.DataFrame(rows, columns=["text", "label"])


# ---- sampling -------------------------------------------------------------
def test_sample_is_stratified_and_hides_the_official_label():
    to_label, key = sample_for_agreement(official_frame(), per_class=10, seed=1)
    assert len(to_label) == 50
    assert list(to_label.columns) == ["id", "text", "label", "is_hard_negative", "notes"]
    assert (to_label["label"] == "").all()
    assert key["official_label"].value_counts().to_dict() == dict.fromkeys(LABELS, 10)
    assert set(key.columns) == {"id", "official_label"}
    assert not any(label in " ".join(to_label.columns) for label in ("official",))


def test_sample_ids_match_and_are_unique_and_shuffled():
    to_label, key = sample_for_agreement(official_frame(), per_class=10, seed=1)
    assert to_label["id"].is_unique
    assert list(to_label["id"]) == list(key["id"])
    merged = to_label.merge(key, on="id")
    assert all(row.official_label in row.text for row in merged.itertuples())
    assert merged["official_label"].ne(merged["official_label"].shift()).sum() > 10


def test_sample_is_deterministic_and_respects_exclusions():
    a = sample_for_agreement(official_frame(), 10, seed=3)
    b = sample_for_agreement(official_frame(), 10, seed=3)
    pd.testing.assert_frame_equal(a[0], b[0])
    from rumod.data.clean import make_dedupe_key

    banned = {make_dedupe_key(t) for t in a[0]["text"]}
    again, _ = sample_for_agreement(official_frame(), 10, seed=3, exclude_keys=banned)
    assert not set(again["text"]) & set(a[0]["text"])


def test_small_classes_are_not_oversampled():
    frame = official_frame(per_class=40)
    number = frame.text.str.extract(r"(\d+)")[0].astype(int)
    frame = frame[~((frame.label == "profane") & (number >= 5))]
    _, key = sample_for_agreement(frame, per_class=30)
    assert (key["official_label"] == "profane").sum() == 5


# ---- statistics -----------------------------------------------------------
def test_kappa_matches_scikit_learn():
    rng = np.random.default_rng(0)
    a = rng.choice(LABELS, size=300)
    b = np.where(rng.random(300) < 0.7, a, rng.choice(LABELS, size=300))
    ours = kappa_from_matrix(confusion_matrix(a, b, LABELS))
    assert ours == pytest.approx(cohen_kappa_score(a, b))


def test_weighted_kappa_equals_kappa_on_replicated_rows():
    rng = np.random.default_rng(1)
    a = rng.choice(LABELS[:3], size=60)
    b = np.where(rng.random(60) < 0.6, a, rng.choice(LABELS[:3], size=60))
    weights = rng.integers(1, 5, size=60)
    ours = kappa_from_matrix(confusion_matrix(a, b, LABELS[:3], weights))
    expected = cohen_kappa_score(np.repeat(a, weights), np.repeat(b, weights))
    assert ours == pytest.approx(expected)


def test_perfect_agreement_gives_kappa_one():
    official = np.array(LABELS * 6)
    rep = agreement_report(official, official.copy(), LABELS, PRIOR, n_boot=50)
    assert rep["five_class"]["kappa"]["value"] == pytest.approx(1.0)
    assert rep["five_class"]["weighted_agreement"]["value"] == pytest.approx(1.0)
    assert rep["toxic_vs_clean"]["kappa"]["value"] == pytest.approx(1.0)


def test_toxic_vs_clean_is_more_forgiving_than_five_classes():
    # we always call religious_hate and sexual_hate "abusive": wrong class, right toxicity
    official = np.array(LABELS * 20)
    mine = np.where(np.isin(official, ["religious_hate", "sexual_hate"]), "abusive", official)
    rep = agreement_report(official, mine, LABELS, PRIOR, n_boot=100)
    assert rep["toxic_vs_clean"]["kappa"]["value"] == pytest.approx(1.0)
    assert rep["five_class"]["kappa"]["value"] < 0.9
    assert rep["per_class"]["religious_hate"]["main_alternative"] == "abusive"
    assert rep["per_class"]["religious_hate"]["agreement"] == 0.0


def test_weighted_agreement_uses_the_natural_class_mix():
    official = np.array(LABELS * 10)
    mine = official.copy()
    mine[official == "clean"] = "abusive"          # everything clean is mislabeled
    rep = agreement_report(official, mine, LABELS, PRIOR, n_boot=50)
    assert rep["five_class"]["agreement"]["value"] == pytest.approx(0.8)
    assert rep["five_class"]["weighted_agreement"]["value"] == pytest.approx(1 - PRIOR["clean"])


def test_confidence_intervals_are_ordered_and_reproducible():
    rng = np.random.default_rng(2)
    official = np.array(LABELS * 30)
    mine = np.where(rng.random(150) < 0.75, official, rng.choice(LABELS, size=150))
    a = agreement_report(official, mine, LABELS, PRIOR, n_boot=100, seed=5)
    b = agreement_report(official, mine, LABELS, PRIOR, n_boot=100, seed=5)
    lo, hi = a["five_class"]["kappa"]["ci"]
    assert lo <= a["five_class"]["kappa"]["value"] <= hi
    assert a["five_class"]["kappa"]["ci"] == b["five_class"]["kappa"]["ci"]


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.490, abs=0.005) and hi == pytest.approx(0.943, abs=0.005)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert np.isnan(wilson_interval(0, 0)[0])


def test_unsure_unusable_and_blank_are_not_compared():
    official, mine, info = split_comparable(
        ["clean", "abusive", "profane", "clean"], ["clean", "unsure", "unusable", ""]
    )
    assert list(official) == ["clean"] and list(mine) == ["clean"]
    assert info["skipped"] == {"unsure": 1, "unusable": 1, "": 1}


def test_report_text_contains_no_tweet_text_and_has_both_views():
    official = np.array(LABELS * 6)
    text = format_agreement(agreement_report(official, official, LABELS, PRIOR, n_boot=30))
    assert "5 classes" in text and "toxic vs clean only" in text and "confusion" in text


# ---- printed confusion matrix in the real-set report ----------------------
def test_report_can_print_the_confusion_matrix():
    rep = classification_report(["a", "a", "b"], ["a", "b", "b"], ["a", "b"], n_boot=10)
    rep.update({"toxic_vs_clean": {"precision": 1, "recall": 1, "f1": 1, "false_alarm_rate": 0},
                "collapsed_to_binary": False})
    plain = format_report({"x": rep})
    shown = format_report({"x": rep}, show_confusion=True)
    assert "confusion" not in plain and "confusion (rows = true" in shown