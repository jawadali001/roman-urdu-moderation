import pytest

from rumod.models.predict import ModelVersionError, ModerationModel

CLEAN_TEXTS = [
    "tum bohat acha insan ho", "aaj mausam bohat acha hai", "shukriya dost bohat meherbani",
    "kal milte hain inshallah", "ye video bohat zabardast hai", "khush raho sab log",
    "acha kaam kiya tum ne", "bohat shukriya bhai", "mazaa aa gaya yaar", "sab theek hai",
]
ABUSIVE_TEXTS = [
    "tum gandu kamina ho", "oye kamine insan door ho", "gandu log hain sab", "kamina kahin ka",
    "tum jaise gandu kamine", "ullu gandu kamina", "gandu insan hai tu", "kamine tum sab",
    "gandu kamina bakwas", "tu gandu hai",
]


@pytest.fixture(scope="module")
def model():
    texts = CLEAN_TEXTS * 5 + ABUSIVE_TEXTS * 5
    labels = ["clean"] * len(CLEAN_TEXTS) * 5 + ["abusive"] * len(ABUSIVE_TEXTS) * 5
    return ModerationModel.fit(texts, labels, min_df=1, ngram_range=(2, 4))


def test_predicts_obvious_cases(model):
    out = model.predict(["tum bohat acha ho", "tum gandu kamina ho"])
    assert out[0]["label"] == "clean"
    assert out[1]["label"] == "abusive"


def test_scores_are_probabilities(model):
    result = model.predict(["tum gandu ho"])[0]
    assert sum(result["scores"].values()) == pytest.approx(1.0)
    assert result["confidence"] == max(result["scores"].values())
    assert set(result["scores"]) == {"clean", "abusive"}


def test_raw_and_normalized_input_give_the_same_answer(model):
    raw = model.predict(["GANDU Kaaamina!!! \U0001F621"])[0]
    clean = model.predict(["gandu kaamina"])[0]
    assert raw["label"] == clean["label"]
    assert raw["confidence"] == pytest.approx(clean["confidence"])


def test_obfuscated_spelling_is_handled(model):
    assert model.predict(["tum g4ndu k@mina ho"])[0]["label"] == "abusive"


@pytest.mark.parametrize("text", ["", "   ", "\U0001F602\U0001F602", "\u0627\u0631\u062f\u0648"])
def test_text_without_latin_letters_gets_default_label(model, text):
    result = model.predict([text])[0]
    assert result["empty"] is True
    assert result["label"] == "clean"
    assert result["confidence"] == 0.0


def test_batch_keeps_order_with_empty_items(model):
    out = model.predict(["tum gandu ho", "", "bohat acha hai"])
    assert [r["empty"] for r in out] == [False, True, False]
    assert out[0]["label"] == "abusive"
    assert out[2]["label"] == "clean"


def test_save_and_load_round_trip(model, tmp_path):
    path = model.save(tmp_path / "m.joblib")
    loaded = ModerationModel.load(path)
    samples = ["tum gandu ho", "bohat acha hai", ""]
    assert loaded.predict(samples) == model.predict(samples)
    assert loaded.labels == model.labels


def test_loading_with_a_different_normalizer_version_fails(model, tmp_path):
    original = model.meta["normalizer_version"]
    model.meta["normalizer_version"] = "999"
    try:
        path = model.save(tmp_path / "old.joblib")
    finally:
        model.meta["normalizer_version"] = original
    with pytest.raises(ModelVersionError):
        ModerationModel.load(path)


def test_training_metadata_is_recorded(model):
    assert model.meta["train_rows"] == 100
    assert model.meta["train_label_counts"] == {"abusive": 50, "clean": 50}

def test_sqrt_balanced_weights():
    from rumod.models.baseline import sqrt_balanced_weights

    weights = sqrt_balanced_weights(["a"] * 100 + ["b"] * 25)
    # balanced would be a: 0.625, b: 2.5 -> square roots
    assert weights["a"] == pytest.approx(0.625**0.5)
    assert weights["b"] == pytest.approx(2.5**0.5)


def test_fit_accepts_sqrt_class_weight():
    texts = CLEAN_TEXTS * 3 + ABUSIVE_TEXTS * 3
    labels = ["clean"] * len(CLEAN_TEXTS) * 3 + ["abusive"] * len(ABUSIVE_TEXTS) * 3
    model = ModerationModel.fit(texts, labels, min_df=1, ngram_range=(2, 4), class_weight="sqrt")
    assert model.predict(["tum gandu kamina ho"])[0]["label"] == "abusive"
    assert set(model.meta["params"]["class_weight"]) == {"clean", "abusive"}