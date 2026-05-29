"""Disfluency augmenter: turn clean training utterances into voice-like variants.

Real Aegis input is spoken, so it is disfluent: fillers ("uhh", "like"), hedges
("i was wondering if you could"), dropped apostrophes, self-corrections, run-ons.
The clean synthetic training set does not look like that, which is the documented
distribution gap behind the chat->integration and agent->integration leaks.

This module applies conservative, label-preserving string transforms to existing
labeled rows and writes the results to a SEPARATE file, data/augmented.jsonl, so
the augmentation is ablatable (delete the file, retrain, compare). train_setfit
globs data/*.jsonl, so the file is picked up automatically.

Run: .venv/bin/python -m routelet.augment
"""

import json
import random
import re
from pathlib import Path

from routelet.data import Intent, load

DATA_DIR = Path("data")
HOLDOUT = Path("evals/holdout.jsonl")
OUT = DATA_DIR / "augmented.jsonl"
SOURCES = ["agent", "chat", "find_action", "integration", "memory"]
VARIANTS_PER_ROW = 2

# Chaining one utterance to another with "and then" turns one action into two,
# which flips integration/find_action into agent. So same-class chaining is only
# label-safe for classes where two instances stay the same intent: chat (still
# chat), memory (still memory), and agent (already multi-step). It is NOT safe
# for integration or find_action.
CHAINABLE = {Intent.CHAT, Intent.MEMORY, Intent.AGENT}

FILLERS = ["uh", "uhh", "um", "like", "you know", "i mean"]
HEDGES = ["can you", "i was wondering if you could", "hey can u", "could you"]
TAILS = ["plz", "thanks", "please", "thx"]


def _norm(text: str) -> str:
    """Loose key for dedup: lowercase, drop apostrophes and punctuation."""
    text = text.lower().strip().replace("'", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _drop_apostrophes(text: str) -> str:
    return text.replace("'", "")


def _insert_filler(text: str, rng: random.Random) -> str:
    """Drop one filler at a random word boundary (not the very end)."""
    words = text.split()
    if len(words) < 2:
        return text
    pos = rng.randint(1, len(words) - 1)
    words.insert(pos, rng.choice(FILLERS))
    return " ".join(words)


def _repeat_word(text: str, rng: random.Random) -> str:
    """~10% chance to stutter-repeat one word ('the the')."""
    words = text.split()
    if len(words) < 2 or rng.random() >= 0.10:
        return text
    pos = rng.randint(0, len(words) - 1)
    words.insert(pos, words[pos])
    return " ".join(words)


def _prepend_hedge(text: str, rng: random.Random) -> str:
    hedge = rng.choice(HEDGES)
    return f"{hedge} {text}"


def _append_tail(text: str, rng: random.Random) -> str:
    return f"{text} {rng.choice(TAILS)}"


def _drop_punct(text: str) -> str:
    return text.rstrip(".?!").rstrip()


def make_variant(text: str, intent: Intent, rng: random.Random) -> str:
    """Build one disfluent, label-preserving variant of a clean utterance.

    Each transform fires probabilistically so the two variants of a row differ.
    Order matters: hedges/tails wrap the sentence, fillers/stutters go inside.
    """
    out = text
    if rng.random() < 0.85:
        out = _insert_filler(out, rng)
    out = _repeat_word(out, rng)
    if rng.random() < 0.70:
        out = _drop_apostrophes(out)
    if rng.random() < 0.45:
        out = _prepend_hedge(out, rng)
    if rng.random() < 0.35:
        out = _append_tail(out, rng)
    if rng.random() < 0.40:
        out = _drop_punct(out)
    return re.sub(r"\s+", " ", out).strip()


def make_chain(
    text: str, partner: str, intent: Intent, rng: random.Random
) -> str | None:
    """For chainable classes, join two same-class utterances with 'and then'.

    Returns None when chaining is not label-safe for this intent. The result is
    then lightly disfluent so it does not read as a clean concatenation.
    """
    if intent not in CHAINABLE:
        return None
    a = _drop_apostrophes(text) if rng.random() < 0.6 else text
    b = _drop_apostrophes(partner) if rng.random() < 0.6 else partner
    joined = f"{_drop_punct(a)} and then {_drop_punct(b)}"
    if rng.random() < 0.5:
        joined = _insert_filler(joined, rng)
    return re.sub(r"\s+", " ", joined).strip()


def augment() -> dict[str, int]:
    """Generate ~2 disfluent variants per source row into data/augmented.jsonl.

    Dedups generated variants against the frozen holdout and against each other,
    and skips any variant that collapses back to its source form. Returns a
    per-class count of rows written.
    """
    rng = random.Random()
    rng.seed(0)

    blocked = {_norm(e.text) for e in load(HOLDOUT)}
    written: list[tuple[str, str]] = []
    counts: dict[str, int] = {s: 0 for s in SOURCES}

    for name in SOURCES:
        rows = load(DATA_DIR / f"{name}.jsonl")
        intent = Intent(name)
        texts = [e.text for e in rows]
        for i, e in enumerate(rows):
            made = 0
            attempts = 0
            while made < VARIANTS_PER_ROW and attempts < 12:
                attempts += 1
                # Roughly 1 in 4 variants for a chainable class is a same-class
                # "and then" run-on; the rest are single disfluent rephrasings.
                if intent in CHAINABLE and len(texts) > 1 and rng.random() < 0.25:
                    partner = rng.choice(texts)
                    if partner == e.text:
                        continue
                    variant = make_chain(e.text, partner, intent, rng)
                    if variant is None:
                        continue
                else:
                    variant = make_variant(e.text, intent, rng)
                key = _norm(variant)
                if not key or key == _norm(e.text):
                    continue
                if key in blocked:
                    continue
                blocked.add(key)
                written.append((variant, name))
                counts[name] += 1
                made += 1

    with OUT.open("w") as f:
        for text, intent in written:
            f.write(json.dumps({"text": text, "intent": intent}) + "\n")
    return counts


def main() -> None:
    counts = augment()
    total = sum(counts.values())
    print(f"wrote {total} augmented rows to {OUT}")
    for name in SOURCES:
        print(f"  {name}: {counts[name]}")


if __name__ == "__main__":
    main()
