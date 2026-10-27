import pandas as pd
import pytest

from rumod.data.real_set import (
    ADMIN_LABELS,
    EVAL_LABELS,
    load_labeled_batches,
    summarize,
    to_unified,
    validate,
)
from rumod.evaluation.real_score import error_table, score_real_set
from rumod.models.predict import ModerationModel


def write_batch(folder, batch, rows, meta=True):
    labeled = pd.DataFrame(rows, columns=["id", "text", "label", "is_hard_negative", "notes"])
    labeled.to_csv(folder / f"batch_{batch:02d}_labeled.csv", index=False)
    if meta:
        info = pd.DataFrame(
            {
                "id": labeled["id"],
                "selection": ["random" if i % 2 == 0 else "candidate:profane"
                              for i in range(len(labeled))],
                "model_label": "clean",
                "model_confidence": "0.9",
            }
        )
        info.to_csv(folder / f"batch_{batch:02d}_sampling_meta.csv", index=False)


def sample_rows(prefix="b01", n=10):
    labels = ["clean", "abusive", "profane", "religious_hate", "sexual_hate"]
    return [(f"{prefix}-{i:04d}", f"unique message {prefix} {i}", labels[i % 5],
             "yes" if i % 5 == 0 and i % 2 == 0 else "", "") for i in range(n)]


# ---- loading ---------------------------------------------------------------
def test_loads_and_joins_sampling_information(tmp_path):
    write_batch(tmp_path, 1, sample_rows())
    df = load_labeled_batches(tmp_path)
    assert len(df) == 10
    assert set(df["selection"]) == {"random", "candidate:profane"}
    assert df["is_hard_negative"].dtype == bool and df["is_hard_negative"].any()
    assert (df["batch"] == 1).all()


def test_several_batches_are_combined(tmp_path):
    write_batch(tmp_path, 1, sample_rows("b01", 5))
    write_batch(tmp_path, 2, sample_rows("b02", 5))
    assert sorted(load_labeled_batches(tmp_path)["batch"].unique()) == [1, 2]


def test_missing_meta_is_marked_unknown(tmp_path):
    write_batch(tmp_path, 1, sample_rows(), meta=False)
    assert set(load_labeled_batches(tmp_path)["selection"]) == {"unknown"}


def test_no_files_and_bad_columns_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labeled_batches(tmp_path)
    incomplete = pd.DataFrame({"id": ["a"], "text": ["x"]})
    incomplete.to_csv(tmp_path / "batch_01_labeled.csv", index=False)
    with pytest.raises(ValueError):
        load_labeled_batches(tmp_path)


# ---- validation and summary -----------------------------------------------
def test_clean_set_has_no_problems(tmp_path):
    write_batch(tmp_path, 1, sample_rows(n=10))
    assert validate(load_labeled_batches(tmp_path)) == []


def test_validate_reports_each_kind_of_problem(tmp_path):
    rows = sample_rows(n=10)
    rows[0] = (rows[0][0], rows[0][1], "", "", "")                  # unlabeled
    rows[1] = (rows[1][0], rows[1][1], "banana", "", "")            # unknown label
    rows[2] = (rows[1][0], "another text", "clean", "", "")         # duplicate id
    rows[3] = ("b01-0099", rows[4][1], "clean", "", "")             # repeated text
    for i in (5, 6, 7):                                             # many unsure
        rows[i] = (rows[i][0], rows[i][1], "unsure", "", "")
    write_batch(tmp_path, 1, rows)
    text = " | ".join(validate(load_labeled_batches(tmp_path)))
    for expected in ("no label yet", "unknown labels", "duplicate ids", "repeat", "unsure"):
        assert expected in text


def test_summary_counts_random_rows_and_shortfalls(tmp_path):
    write_batch(tmp_path, 1, sample_rows(n=10))
    summary = summarize(load_labeled_batches(tmp_path), target=3)
    assert summary["labeled"] == 10 and summary["unlabeled"] == 0
    assert summary["counts"]["clean"] == 2
    assert sum(summary["counts_random_only"].values()) == 5
    assert summary["shortfall"]["clean"] == 1
    assert summary["hard_negatives"] >= 1


def test_to_unified_drops_unsure_and_unusable(tmp_path):
    rows = sample_rows(n=10)
    rows[0] = (rows[0][0], rows[0][1], "unsure", "", "")
    rows[1] = (rows[1][0], rows[1][1], "unusable", "", "")
    write_batch(tmp_path, 1, rows)
    unified = to_unified(load_labeled_batches(tmp_path))
    assert len(unified) == 8
    assert set(unified["label"]) <= set(EVAL_LABELS)
    assert not set(unified["label"]) & set(ADMIN_LABELS)
    assert (unified["source"] == "real").all() and not unified["is_synthetic"].any()


# ---- scoring ----------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model():
    texts = ["tum bohat acha ho", "tum gandu kamina ho"] * 25
    labels = ["clean", "abusive"] * 25
    return ModerationModel.fit(texts, labels, min_df=1, ngram_range=(2, 4))


def make_rows():
    return pd.DataFrame(
        {
            "text": ["tum bohat acha ho", "tum gandu kamina ho", "bohat acha hai", "gandu kamina"],
            "label": ["clean", "abusive", "clean", "abusive"],
            "selection": ["random", "random", "candidate:abusive", "candidate:abusive"],
            "id": ["1", "2", "3", "4"],
            "notes": ["", "", "", ""],
        }
    )


def test_scoring_reports_all_rows_and_random_rows_separately(tiny_model):
    result = score_real_set(tiny_model, make_rows(), n_boot=20)
    assert result["all"]["n"] == 4
    assert result["random_only"]["n"] == 2
    assert len(result["predictions"]) == 4


def test_no_random_rows_gives_none(tiny_model):
    rows = make_rows().assign(selection="candidate:abusive")
    assert score_real_set(tiny_model, rows, n_boot=20)["random_only"] is None


def test_error_table_lists_only_mistakes():
    rows = make_rows()
    results = [{"label": "clean", "confidence": 0.9}, {"label": "clean", "confidence": 0.6},
               {"label": "clean", "confidence": 0.8}, {"label": "abusive", "confidence": 0.7}]
    errors = error_table(rows, results)
    assert list(errors["id"]) == ["2"]
    assert errors.loc[0, "predicted"] == "clean"