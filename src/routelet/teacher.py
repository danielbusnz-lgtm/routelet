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
SYSTEM = """You are an intent classifier for short voice commands. Classify each into one of five
intents. First strip filler words (um, uh, like) and self-corrections, then classify.

- find_action: locate or operate a named UI element on the current screen.
- integration: one discrete action against an app or service.
- chat: general knowledge or conversation. No app action, no personal data. The default.
- memory: store or recall a personal fact. Storing needs an explicit remember/note/save.
- agent: a task needing two or more steps or a plan.

Apply these boundary rules first for the tricky cases:
- Steps: two or more chained actions ("X and then Y"), or an implied multi-step task
  needing a plan ("book me a restaurant"), is agent, not integration.
- Source: answered from a stored personal fact is memory; from world knowledge it is chat.
- Storing: "remember/note/save X" is memory even when it looks like another intent.
- A UI verb (click/tap/scroll/select) on a named element is find_action, even if an app is named.
- Playback (skip, pause, next, volume) is integration, not find_action, unless a button is named.
- A question ABOUT a UI element or action is chat, not find_action: "what does X do", "what's the
  X button", "tell me about X", "explain how to X", "talk me through X". find_action needs a command
  to locate or operate something, not a request to explain it.

If a command still fits more than one after these rules, pick the first match in this order:
agent, memory, integration, find_action, chat."""

# Real intents only: the teacher labels actual commands, never the reject class
# (NONE), which is learned from generated OOD data, not from the LLM.
_REAL_INTENTS = [i.value for i in Intent if i is not Intent.NONE]

INTENT_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": _REAL_INTENTS}},
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
