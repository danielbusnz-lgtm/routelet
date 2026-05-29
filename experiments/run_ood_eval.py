"""OOD eval for models/setfit_tools_proto against evals/tool_routing_ood.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from routelet.preprocess import preprocess

# ---------------------------------------------------------------------------
# 1. Load OOD eval
# ---------------------------------------------------------------------------

OOD_PATH = PROJECT_ROOT / "evals" / "tool_routing_ood.jsonl"
TRAIN_PATH = PROJECT_ROOT / "data" / "integration.jsonl"
MODEL_PATH = PROJECT_ROOT / "models" / "setfit_tools_proto"

TOOLS = ["spotify", "gmail", "github", "youtube", "no_tool"]

ood_rows: list[tuple[str, str]] = []
with open(OOD_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        ood_rows.append((obj["text"], obj["tool"]))

print(f"OOD eval rows: {len(ood_rows)}")
from collections import Counter
dist = Counter(lbl for _, lbl in ood_rows)
for t in TOOLS:
    print(f"  {t:<12} {dist[t]:>3}")

# ---------------------------------------------------------------------------
# 2. Leakage check: exact text match against training pool
# ---------------------------------------------------------------------------

train_texts: set[str] = set()
with open(TRAIN_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        train_texts.add(obj["text"].strip().lower())

overlaps = [
    (text, lbl)
    for text, lbl in ood_rows
    if text.strip().lower() in train_texts
]
print(f"\nLeakage check: {len(overlaps)} overlap(s) with integration.jsonl (must be 0)")
for text, lbl in overlaps:
    print(f"  DUPLICATE: {text!r} ({lbl})")

# ---------------------------------------------------------------------------
# 3. Load model and predict
# ---------------------------------------------------------------------------

from setfit import SetFitModel  # noqa: E402

print(f"\nLoading model from {MODEL_PATH} ...")
model = SetFitModel.from_pretrained(str(MODEL_PATH))

texts = [text for text, _ in ood_rows]
labels = [lbl for _, lbl in ood_rows]

preprocessed = [preprocess(t) for t in texts]
preds_raw = model.predict(preprocessed)
preds = [str(p) for p in preds_raw]

# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------

from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

print("\n" + "=" * 64)
print("OOD EVAL RESULTS")
print("=" * 64)

print(classification_report(labels, preds, labels=TOOLS, zero_division=0))

cm = confusion_matrix(labels, preds, labels=TOOLS)
col_w = max(len(t) for t in TOOLS) + 2
header = " " * col_w + "".join(f"{t:>{col_w}}" for t in TOOLS)
print("Confusion matrix (rows=true, cols=pred):")
print(header)
for i, row_lbl in enumerate(TOOLS):
    row_str = f"{row_lbl:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in cm[i])
    print(row_str)

nt_true = [lbl == "no_tool" for lbl in labels]
nt_correct = sum(p == "no_tool" for p, t in zip(preds, labels) if t == "no_tool")
nt_total = sum(nt_true)
nt_recall = nt_correct / nt_total if nt_total else 0.0
print(f"\nno_tool recall: {nt_recall:.3f}  ({nt_correct}/{nt_total})")

correct = sum(p == t for p, t in zip(preds, labels))
acc = correct / len(labels)
print(f"overall accuracy: {acc:.3f}  ({correct}/{len(labels)})")

print("\nMisclassifications:")
errors = [(labels[i], preds[i], texts[i]) for i in range(len(labels)) if preds[i] != labels[i]]
if not errors:
    print("  none")
else:
    for true_lbl, pred_lbl, text in errors:
        print(f"  true={true_lbl:<12} pred={pred_lbl:<12} {text!r}")
