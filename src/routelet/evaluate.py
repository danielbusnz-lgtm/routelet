"""LLM baseline: classify the held-out eval set with Claude.

One Claude call per command, constrained to a single intent via structured
output. This is the bar routelet's classifier is trying to match: it reports
Claude's accuracy and per-call latency to set next to the TF-IDF baseline.

Model is Haiku 4.5 on purpose. Routing is latency-critical, so the honest
baseline is the fast model you'd actually route with, not Opus. Change MODEL to
claude-opus-4-7 to measure the accuracy ceiling instead.

Needs ANTHROPIC_API_KEY in the environment.
"""

import json
import time

import anthropic
from sklearn.metrics import accuracy_score, classification_report

from routelet.data import Intent, load

MODEL = "claude-haiku-4-5"
EVAL_FILE = "evals/holdout.jsonl"

# Condensed from docs/taxonomy.md. Kept stable so it caches across the per-command
# calls (though a prompt this short may sit under the model's min cacheable prefix).
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


def main() -> None:
    client = anthropic.Anthropic()
    examples = load(EVAL_FILE)

    preds: list[str] = []
    latencies: list[float] = []
    cache_reads = 0
    for ex in examples:
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": ex.text}],
            output_config={"format": {"type": "json_schema", "schema": INTENT_SCHEMA}},
        )
        latencies.append(time.perf_counter() - t0)
        cache_reads += resp.usage.cache_read_input_tokens
        text = next(b.text for b in resp.content if b.type == "text")
        preds.append(json.loads(text)["intent"])

    true = [e.intent.value for e in examples]
    labels = [i.value for i in Intent]
    latencies.sort()

    print(f"model {MODEL}   eval {len(examples)}")
    print(f"accuracy   {accuracy_score(true, preds):.2f}")
    print(f"latency    p50 {latencies[len(latencies) // 2] * 1000:.0f} ms   "
          f"p95 {latencies[int(len(latencies) * 0.95)] * 1000:.0f} ms")
    print(f"cache_read_input_tokens total: {cache_reads}")
    print()
    print(classification_report(true, preds, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
