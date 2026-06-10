"""Intent-independent input normalization applied at both training and inference.

Rules applied in order:
  1. Lowercase. The training corpus is all lowercase; STT output is mixed-case,
     so normalizing here closes the train/serve surface skew.
  2. Strip terminal punctuation (trailing whitespace and .! runs), but keep a
     terminal question mark, normalized to exactly one. STT appends periods the
     training data never had, those are noise. A terminal "?" is signal: it
     marks the question register (capability checks, tag questions) that
     otherwise embeds like a command. The augmenter emits "?" variants of
     question-shaped training rows so the signal exists at training time too.
  3. Secret keyword tail: mask everything after a secret keyword with <SECRET>.
  4. Email addresses -> <EMAIL>.
  5. Digit runs of length >= 4 -> <NUM>.

Over-redaction is acceptable; leaking is not.
"""

import re

_TERMINAL_PUNCT = re.compile(r"[\s.!?]+$")
_SECRET = re.compile(
    r"(?i)\b(password|passcode|pin|ssn|secret|token|api\s*key|api\s*secret|credit card|card number)\b.*$",  # noqa: E501
    re.MULTILINE,
)
_EMAIL = re.compile(r"(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
_NUM = re.compile(r"\b\d{4,}\b")


def preprocess(text: str) -> str:
    """Normalize text before encoding. Safe to call without knowing the intent."""
    text = text.lower()
    tail = _TERMINAL_PUNCT.search(text)
    is_question = tail is not None and "?" in tail.group()
    text = _TERMINAL_PUNCT.sub("", text)
    text = _SECRET.sub(lambda m: m.group(1) + " <SECRET>", text)
    text = _EMAIL.sub("<EMAIL>", text)
    text = _NUM.sub("<NUM>", text)
    if is_question:
        text += "?"
    return text
