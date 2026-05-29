"""The Intent label schema and dataset prep for intent routing.

Reads labeled JSONL, one ``{"text", "intent"}`` object per line, into the
validated Example lists that train.py and evaluate.py consume.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Intent(StrEnum):
    """The label set. The first five are the real intents (definitions and
    boundary rules in docs/taxonomy.md). NONE is the reject class: out-of-
    distribution or garbled input the router should not act on, used to give the
    model an explicit "I don't know" instead of confidently mislabeling junk.

    StrEnum gives two things we rely on: members serialize as plain strings, and
    Intent("typo") raises ValueError, which is how labels get validated on load.
    """

    FIND_ACTION = "find_action"
    INTEGRATION = "integration"
    CHAT = "chat"
    MEMORY = "memory"
    AGENT = "agent"
    NONE = "none"


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
