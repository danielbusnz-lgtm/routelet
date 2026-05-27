"""Intent-independent input normalization applied at both training and inference.

Rules applied in order:
  1. Secret keyword tail: mask everything after a secret keyword with <SECRET>.
  2. Email addresses -> <EMAIL>.
  3. Digit runs of length >= 4 -> <NUM>.

Over-redaction is acceptable; leaking is not.
"""

import re

_SECRET = re.compile(
    r"(?i)\b(password|passcode|pin|ssn|secret|token|api\s*key|api\s*secret|credit card|card number)\b.*$",  # noqa: E501
    re.MULTILINE,
)
_EMAIL = re.compile(r"(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
_NUM = re.compile(r"\b\d{4,}\b")


def preprocess(text: str) -> str:
    """Normalize text before encoding. Safe to call without knowing the intent."""
    text = _SECRET.sub(lambda m: m.group(1) + " <SECRET>", text)
    text = _EMAIL.sub("<EMAIL>", text)
    text = _NUM.sub("<NUM>", text)
    return text
