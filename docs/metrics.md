# Performance report

A reproducible figure report. `report.py` scores the models on the frozen
holdout and writes figures plus `metrics.json`. Re-run after every retrain.

Status: planned, not built.

## Figures

1. Model comparison: TF-IDF baseline, SetFit (shipped), Claude Haiku. Accuracy on the same holdout.
2. Latency vs accuracy: routelet (~40ms, on-device) against Haiku (~700ms) and TF-IDF.
3. Confusion matrix, shipped model.
4. Per-class precision, recall, F1 with Wilson confidence intervals.
5. Confidence histogram, correct vs wrong predictions.

## Rules

- Score every model on the same frozen `evals/holdout.jsonl`. Never compare numbers across different eval sets.
- Apply `preprocess()` before inference, same as training.
- Haiku accuracy and latency come from `evaluate.py`. Needs `ANTHROPIC_API_KEY`.
- matplotlib only. Install into the venv, not pyproject.

## Output

- `report/*.png`: the figures.
- `report/metrics.json`: the numbers behind the figures. Single source for the README and any later dashboard.
