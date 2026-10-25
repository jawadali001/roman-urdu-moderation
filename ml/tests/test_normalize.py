"""Tests for rumod.features.normalize.

Run from the repo root:
    python -m pytest ml/tests -q
"""
import sys
from pathlib import Path

import pytest

# Allow running before the package is installed: ml/tests -> ml/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rumod.features.normalize import NORMALIZER_VERSION, normalize, normalize_batch  # noqa: E402


@pytest.mark.parametrize(
    "raw, expected",
    [
        # case, punctuation, stretched letters
        ("Yaaaar!!! Tum KAISE ho?", "yaar tum kaise ho"),
        ("bohat   acha\n\nhai", "bohat acha hai"),
        # RUACD style: apostrophes become spaces
        ("can't stop", "can t stop"),
        # URLs, e-mails, mentions removed; hashtag keeps its word
        ("check https://t.co/xyz @user nice", "check nice"),
        ("mail me at john@gmail.com ok", "mail me at ok"),
        ("#PakistanZindabad great", "pakistanzindabad great"),
        # obfuscation between letters is undone
        ("k@mina", "kamina"),
        ("b3hn", "behn"),
        ("h4raami", "haraami"),
        ("p0rn", "porn"),
        # digits elsewhere are dropped
        ("gandu123", "gandu"),
        ("100 log", "log"),
        ("4u", "u"),
        # emoji and symbols removed
        ("acha \U0001F602\U0001F602 hai", "acha hai"),
        # accents removed
        ("caf\u00e9", "cafe"),
        # nothing Latin left -> empty string
        ("\U0001F602\U0001F602", ""),
        ("\u0627\u0631\u062f\u0648", ""),
        ("", ""),
        ("   ", ""),
    ],
)
def test_examples(raw, expected):
    assert normalize(raw) == expected


def test_idempotent():
    samples = [
        "Yaaaar!!! Tum KAISE ho?", "k@mina", "b3hn!!", "gandu123", "can't stop",
        "check https://t.co/xyz @user", "\U0001F602", "hahahahaha", "Ab$urd",
    ]
    for text in samples:
        once = normalize(text)
        assert normalize(once) == once, text


def test_output_only_lowercase_letters_and_single_spaces():
    out = normalize("Aa!!1 b@@d $$ 3 ZZZ... #tag \U0001F602 caf\u00e9")
    assert all(ch.islower() or ch == " " for ch in out)
    assert "  " not in out and out == out.strip()


def test_rejects_non_strings():
    with pytest.raises(TypeError):
        normalize(None)
    with pytest.raises(TypeError):
        normalize(123)


def test_batch_matches_single():
    texts = ["Yaaar!!", "k@mina", ""]
    assert normalize_batch(texts) == [normalize(t) for t in texts]


def test_version_is_set():
    assert isinstance(NORMALIZER_VERSION, str) and NORMALIZER_VERSION