import numpy as np
import pandas as pd
import pytest

from rumod.data.labeling import assign_ids, build_pool, scores_frame, select_rows

LABELS = ["abusive", "clean", "profane", "religious_hate", "sexual_hate"]


def make_pool_and_scores(n=100, seed=0):
    rng = np.random.default_rng(seed)
    pool = pd.DataFrame({"text": [f"message number {i}" for i in range(n)],
                         "dedupe_key": [f"message number {i}" for i in range(n)]})
    raw = rng.random((n, len(LABELS)))
    raw = raw / raw.sum(axis=1, keepdims=True)
    scores = pd.DataFrame(raw, columns=LABELS)
    scores["model_label"] = scores[LABELS].idxmax(axis=1)
    scores["model_confidence"] = scores[LABELS].max(axis=1)
    return pool, scores


# ---- build_pool ----------------------------------------------------------
def test_build_pool_cleans_dedupes_and_excludes():
    raw = pd.DataFrame({"tweet": [
        "Yaaar  tum  kaise ho!!",      # kept
        "yaar tum kaise ho",           # duplicate of the first (case, punctuation, stretch)
        "\U0001F602\U0001F602",        # no letters
        "already used message",        # excluded below
        "bilkul theek hai",            # kept
    ]})
    pool = build_pool(raw, exclude_keys={"already used message"})
    assert list(pool["text"]) == ["Yaaar tum kaise ho!!", "bilkul theek hai"]
    assert set(pool.columns) == {"text", "dedupe_key"}


# ---- select_rows ---------------------------------------------------------
def test_counts_and_no_duplicates():
    pool, scores = make_pool_and_scores()
    picked = select_rows(pool, scores, n_random=20, per_class=5, seed=1)
    assert len(picked) == 20 + 4 * 5
    assert picked["text"].is_unique
    assert (picked["selection"] == "random").sum() == 20
    for label in ("profane", "religious_hate", "sexual_hate", "abusive"):
        assert (picked["selection"] == f"candidate:{label}").sum() == 5


def test_candidates_are_the_highest_scoring_unpicked_rows():
    pool, scores = make_pool_and_scores()
    picked = select_rows(pool, scores, n_random=10, per_class=5, seed=2)
    random_texts = set(picked.loc[picked["selection"] == "random", "text"])
    profane = picked[picked["selection"] == "candidate:profane"]
    available = scores[~pool["text"].isin(random_texts)]
    expected_floor = available["profane"].nlargest(5).min()
    chosen_scores = scores.loc[pool["text"].isin(profane["text"]), "profane"]
    assert chosen_scores.min() == pytest.approx(expected_floor)


def test_same_seed_gives_the_same_batch_and_other_seed_differs():
    pool, scores = make_pool_and_scores()
    a = select_rows(pool, scores, 15, 3, seed=7)
    b = select_rows(pool, scores, 15, 3, seed=7)
    c = select_rows(pool, scores, 15, 3, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["text"]) != list(c["text"])


def test_asking_for_more_than_the_pool_holds_does_not_fail():
    pool, scores = make_pool_and_scores(n=12)
    picked = select_rows(pool, scores, n_random=10, per_class=50, seed=3)
    assert len(picked) == 12 and picked["text"].is_unique


def test_missing_class_scores_raise():
    pool, scores = make_pool_and_scores()
    with pytest.raises(ValueError):
        select_rows(pool, scores.drop(columns=["profane"]), 5, 2)
    with pytest.raises(ValueError):
        select_rows(pool, scores.iloc[:-1], 5, 2)


def test_model_information_is_kept_for_the_meta_file():
    pool, scores = make_pool_and_scores()
    picked = select_rows(pool, scores, 5, 2, seed=4)
    assert {"selection", "model_label", "model_confidence"} <= set(picked.columns)
    assert picked["model_confidence"].between(0, 1).all()


# ---- ids and score frame -------------------------------------------------
def test_assign_ids_are_unique_and_formatted():
    pool, scores = make_pool_and_scores()
    ids = assign_ids(select_rows(pool, scores, 5, 2), batch=3)["id"]
    assert ids.is_unique
    assert ids.iloc[0] == "b03-0001"


def test_scores_frame_handles_empty_results():
    results = [
        {"label": "clean", "confidence": 0.9, "scores": {"clean": 0.9, "abusive": 0.1}},
        {"label": "clean", "confidence": 0.0, "scores": {}},
    ]
    frame = scores_frame(results, ["abusive", "clean"])
    assert list(frame.columns) == ["abusive", "clean", "model_label", "model_confidence"]
    assert frame.loc[1, "abusive"] == 0.0