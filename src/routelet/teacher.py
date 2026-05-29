"""Offline Claude teacher: the strong-model labeler for distillation.

The taxonomy prompt and label schema live here so the eval baseline
(``evaluate.py``) and the ingest labeler (``Scripts/ingest_samples.py``)
classify with identical instructions. If the two drifted, the eval would be
measuring a different teacher than the one that labeled the training data.

Teacher, not student: this calls Claude and runs offline only, never in
production. See docs/distillation.md.
"""

import json
import os

import anthropic

from routelet.data import Intent

# The teacher's whole job is label accuracy, so this defaults to the strongest
# model. Bulk runs that care more about cost can override with the
# ROUTELET_TEACHER_MODEL env var (e.g. claude-haiku-4-5).
DEFAULT_MODEL = "claude-opus-4-8"

# Condensed from docs/taxonomy.md. Kept stable so it caches across the
# per-command calls.
SYSTEM = """You are an intent router. Classify each user command into exactly one of five intents:

- find_action: locate or operate a UI element on the current screen.
- integration: one discrete action against an app or service.
- chat: answer from general knowledge or conversation. No app action, no personal data.
- memory: store or recall a personal fact (storing needs an explicit remember/note/save).
- agent: a task needing two or more chained steps or a plan.

Tie-breakers for the tricky cases:
- Two or more chained actions ("X and then Y") is agent, not integration.
- A question answered from a stored personal fact is memory; from world knowledge it is chat.
- A UI verb (click/tap/scroll) on a named element is find_action, even if an app is named.
- Playback (skip, pause, next, volume) is integration, not find_action, unless a button is named.
- "explain how to..." or "talk me through..." is chat, even when it names an app action."""

INTENT_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": [i.value for i in Intent]}},
    "required": ["intent"],
    "additionalProperties": False,
}


def resolve_model() -> str:
    """The teacher model to use: ROUTELET_TEACHER_MODEL if set, else the default."""
    return os.environ.get("ROUTELET_TEACHER_MODEL", DEFAULT_MODEL)


def classify(client: anthropic.Anthropic, text: str, model: str | None = None) -> Intent:
    """Label one command with the teacher. Structured output pins the response to
    the five-intent enum, so the parse can't return anything off-taxonomy."""
    resp = client.messages.create(
        model=model or resolve_model(),
        max_tokens=100,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": text}],
        output_config={"format": {"type": "json_schema", "schema": INTENT_SCHEMA}},
    )
    out = next(b.text for b in resp.content if b.type == "text")
    return Intent(json.loads(out)["intent"])
