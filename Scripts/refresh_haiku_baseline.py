"""Refresh report/baselines.json by scoring Claude Haiku on the current frozen
holdout. This is the paid step the report's cached baseline stands in for.

Mirrors src/routelet/evaluate.py exactly (same model, same shared teacher
prompt and schema, raw text in) so the cached number stays methodologically
identical to the eval baseline. Run after the holdout changes:

    ANTHROPIC_API_KEY=... uv run python Scripts/refresh_haiku_baseline.py

Writes accuracy, p50/p95 latency, and per-class precision/recall/f1 to
report/baselines.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anthropic
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from routelet.data import Intent, load  # noqa: E402
from routelet.teacher import INTENT_SCHEMA, SYSTEM  # noqa: E402

MODEL = "claude-haiku-4-5"
HOLDOUT = PROJECT_ROOT / "evals" / "holdout.jsonl"
BASELINES = PROJECT_ROOT / "report" / "baselines.json"


def main() -> None:
    client = anthropic.Anthropic()
    examples = load(HOLDOUT)
    labels = [i.value for i in Intent]

    preds: list[str] = []
    latencies: list[float] = []
    for ex in examples:
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": ex.text}],
            output_config={"format": {"type": "json_schema", "schema": INTENT_SCHEMA}},
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        text = next(b.text for b in resp.content if b.type == "text")
        preds.append(json.loads(text)["intent"])

    gold = [e.intent.value for e in examples]
    acc = float(accuracy_score(gold, preds))
    lat = np.array(latencies)
    report = classification_report(gold, preds, labels=labels, output_dict=True, zero_division=0)

    per_class = {
        lbl: {
            "precision": round(report[lbl]["precision"], 2),
            "recall": round(report[lbl]["recall"], 2),
            "f1": round(report[lbl]["f1-score"], 2),
            "support": int(report[lbl]["support"]),
        }
        for lbl in labels
    }

    out = {
        "_note": (
            "Cached Claude baseline so report.py does not re-spend on the paid API "
            "every regen. Regenerate by running Scripts/refresh_haiku_baseline.py "
            "with ANTHROPIC_API_KEY against the current evals/holdout.jsonl."
        ),
        "haiku": {
            "model": MODEL,
            "eval_n": len(examples),
            "accuracy": round(acc, 3),
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 0),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 0),
            "per_class": per_class,
        },
    }
    BASELINES.write_text(json.dumps(out, indent=2) + "\n")

    print(f"scored {len(examples)} rows on {MODEL}")
    print(f"  accuracy {acc:.3f}")
    print(f"  latency  p50 {np.percentile(lat, 50):.0f}ms  p95 {np.percentile(lat, 95):.0f}ms")
    print(f"wrote {BASELINES}")


if __name__ == "__main__":
    main()
