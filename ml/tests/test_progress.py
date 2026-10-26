import pytest

from rumod.models.predict import ModerationModel
from rumod.utils.progress import stage, track

TEXTS = ["tum bohat acha ho", "tum gandu kamina ho"] * 20
LABELS = ["clean", "abusive"] * 20


def test_track_disabled_returns_the_same_object():
    items = [1, 2, 3]
    assert track(items, "x", enabled=False) is items


def test_track_enabled_still_yields_every_item(capsys):
    assert list(track(range(5), "counting", total=5)) == [0, 1, 2, 3, 4]
    assert "counting" in capsys.readouterr().err


def test_stage_prints_start_and_end_when_not_a_terminal(capsys):
    with stage("fitting thing"):
        pass
    err = capsys.readouterr().err
    assert "fitting thing ..." in err
    assert "fitting thing: done in" in err


def test_stage_disabled_prints_nothing(capsys):
    with stage("hidden", enabled=False):
        pass
    assert capsys.readouterr().err == ""


def test_stage_reports_completion_even_if_the_block_fails(capsys):
    with pytest.raises(ValueError), stage("failing step"):
        raise ValueError("boom")
    assert "failing step: done in" in capsys.readouterr().err


def test_progress_does_not_change_the_model(capsys):
    quiet = ModerationModel.fit(TEXTS, LABELS, min_df=1, ngram_range=(2, 4))
    capsys.readouterr()
    loud = ModerationModel.fit(TEXTS, LABELS, min_df=1, ngram_range=(2, 4), progress=True)
    err = capsys.readouterr().err
    assert "fitting classifier" in err and "building TF-IDF features" in err

    samples = ["tum gandu ho", "bohat acha hai", ""]
    assert loud.predict(samples) == quiet.predict(samples)


def test_predict_with_progress_matches_silent_predict(capsys):
    model = ModerationModel.fit(TEXTS, LABELS, min_df=1, ngram_range=(2, 4))
    samples = ["tum gandu ho", "bohat acha hai"]
    assert model.predict(samples, progress=True) == model.predict(samples)
    assert "scoring 2 messages" in capsys.readouterr().err