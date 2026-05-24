"""The Intent label schema and dataset prep for intent routing.

Reads labeled JSONL, one ``{"text", "intent"}`` object per line, into the
deterministic train/test splits that train.py and evaluate.py consume.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sklearn.model_selection import train_test_split


class Intent(StrEnum):
    """The label set, frozen. Definitions and boundary rules in docs/taxonomy.md.

    StrEnum gives two things we rely on: members serialize as plain strings, and
    Intent("typo") raises ValueError, which is how labels get validated on load.
    """

    FIND_ACTION = "find_action"
    INTEGRATION = "integration"
    CHAT = "chat"
    MEMORY = "memory"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    intent: Intent


def load(path: str | Path) -> list[Example]:
    """Read a labeled JSONL file into validated Examples.

    Each line is one ``{"text", "intent"}`` object. An unknown intent raises
    ValueError tagged with the file and line number, so a typo'd label surfaces
    here instead of mid-training. Blank lines are skipped.
    """
    examples: list[Example] = []
    path = Path(path)
    with path.open() as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                intent = Intent(row["intent"])
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
            examples.append(Example(text=row["text"], intent=intent))
    return examples


def load_dir(directory: str | Path) -> list[Example]:
    """Load and concatenate every .jsonl file in a directory.

    Pulls the whole generated training pool out of data/ in one call. Order
    follows sorted filenames, so shuffle or split downstream if order matters.
    """
    examples: list[Example] = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        examples.extend(load(path))
    return examples


def split(
    examples: list[Example],
    test_size: float = 0.25,
    seed: int = 0,
) -> tuple[list[Example], list[Example]]:
    """Deterministic, stratified train/test split.

    Stratified so each intent keeps its proportion in both halves. With classes
    this small an unstratified split can leave an intent with zero test rows and
    silently blind the eval to it. ``seed`` fixes the partition so accuracy is
    comparable across runs. Raises ValueError if any intent has fewer than 2
    examples, since it can't land in both halves.

    This splits the hand-labeled pool. The real frozen eval set will live in
    evals/, sourced apart from the training data.
    """
    labels = [e.intent for e in examples]
    train, test = train_test_split(
        examples, test_size=test_size, random_state=seed, stratify=labels
    )
    return train, test
