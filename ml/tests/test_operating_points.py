import numpy as np
import pytest

from rumod.evaluation.operating_points import (
    operating_summary,
    recall_at_false_alarm,
    toxic_scores,
)
from rumod.models.predict import ModerationModel


def test_recall_at_false_alarm_hand_example():
    # clean scores: 0.1 0.2 0.3 0.9 | toxic scores: 0.4 0.6 0.8 0.95
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    score = [0.1, 0.2, 0.3, 0.9, 0.4, 0.6, 0.8, 0.95]
    # allowing 1 of 4 clean messages (25%) to be flagged: threshold 0.4 catches all toxic
    best = recall_at_false_alarm(y, score, 0.25)
    assert best["recall"] == 1.0
    assert best["false_alarm_rate"] == pytest.approx(0.25)
    assert best["threshold"] == pytest.approx(0.4)
    # allowing none: only the 0.95 message is above every clean score
    strict = recall_at_false_alarm(y, score, 0.0)
    assert strict["false_alarm_rate"] == 0.0
    assert strict["recall"] == pytest.approx(0.25)
    assert strict["threshold"] == pytest.approx(0.95)


def test_recall_never_decreases_when_more_false_alarms_are_allowed():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.4
    score = np.where(y, rng.normal(0.65, 0.2, 500), rng.normal(0.4, 0.2, 500))
    recalls = [recall_at_false_alarm(y, score, t)["recall"] for t in (0.01, 0.05, 0.1, 0.3)]
    assert recalls == sorted(recalls)


def test_summary_has_auc_and_all_targets():
    y = [0, 0, 1, 1]
    summary = operating_summary(y, [0.1, 0.4, 0.35, 0.8], targets=(0.0, 0.5))
    assert summary["auc"] == pytest.approx(0.75)
    assert len(summary["at"]) == 2
    assert summary["toxic_share"] == 0.5


def test_perfect_separation_gives_auc_one():
    summary = operating_summary([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert summary["auc"] == 1.0
    assert all(point["recall"] == 1.0 for point in summary["at"])


def test_toxic_scores_use_clean_probability_and_zero_for_empty_text():
    texts = ["tum bohat acha ho", "tum gandu kamina ho"] * 20
    labels = ["clean", "abusive"] * 20
    model = ModerationModel.fit(texts, labels, min_df=1, ngram_range=(2, 4))
    scores = toxic_scores(model, ["tum gandu kamina ho", "tum bohat acha ho", ""])
    assert scores[0] > 0.5 > scores[1]
    assert scores[2] == 0.0
    assert np.all((scores >= 0) & (scores <= 1))