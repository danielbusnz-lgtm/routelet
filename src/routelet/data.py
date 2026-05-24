"""The Intent label schema and dataset prep for intent routing.

Reads labeled JSONL, one ``{"text", "intent"}`` object per line, into the
deterministic train/test splits that train.py and evaluate.py consume.
"""

from enum import StrEnum


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


# TODO: load examples/*.jsonl, validate the intent labels, split into train/test.
