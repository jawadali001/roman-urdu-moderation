"""Text normalization shared by training and the API.

ONE function, `normalize()`, is used everywhere. If training and serving ever
normalize differently, accuracy silently drops, so never copy this logic.

Why these rules
---------------
RUACD (95% of our rows) is already lowercase with no punctuation, digits,
emoji, URLs or mentions, and it turns apostrophes into spaces ("can t").
RUHSOLD tweets and real chat messages are raw. Normalizing everything to the
RUACD style makes all sources look alike, so the model learns the language and
not the source style.

Steps
-----
1. Unicode NFKC, remove accents, lowercase.
2. Remove URLs, e-mail addresses and @mentions (a mention is an @ at the start
   of a word, so the obfuscation "k@mina" is NOT treated as a mention).
   Hashtags keep their word.
3. Undo common obfuscation: 0->o 1->i 3->e 4->a 5->s 7->t @->a $->s, but only
   when the character sits directly between two letters ("b3hn" -> "behn",
   "k@mina" -> "kamina"). Digits elsewhere ("gandu123", "4u") are dropped.
   Not handled: several obfuscation characters in a row ("k@@mina").
4. Everything that is not a letter a-z becomes a space (punctuation, digits,
   emoji, non-Latin script).
5. Squeeze letters repeated 3+ times to 2 ("yaaaar" -> "yaar").
6. Collapse whitespace.

Text with no Latin letters (emoji only, Urdu script) normalizes to "". Callers
must handle the empty string.

Personal-information detection (phone numbers etc.) must run on the RAW text,
before normalization, because normalization deletes digits.

NORMALIZER_VERSION is stored with every trained model so a mismatch between
training and serving can be detected.
"""
from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "1"

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_MENTION = re.compile(r"(?<!\w)@\w+")
_LEET_INSIDE = re.compile(r"(?<=[a-z])[013457@$](?=[a-z])")
_NON_LETTER = re.compile(r"[^a-z]+")
_REPEATS = re.compile(r"([a-z])\1{2,}")

# Only these are undone, and only strictly inside a word.
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Return the normalized form of one message (may be the empty string)."""
    if not isinstance(text, str):
        raise TypeError(f"normalize() expects str, got {type(text).__name__}")

    text = _strip_accents(unicodedata.normalize("NFKC", text)).lower()
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _LEET_INSIDE.sub(lambda m: _LEET[m.group()], text)
    text = _NON_LETTER.sub(" ", text)
    text = _REPEATS.sub(r"\1\1", text)
    return " ".join(text.split())


def normalize_batch(texts) -> list[str]:
    """Normalize an iterable of messages."""
    return [normalize(t) for t in texts]