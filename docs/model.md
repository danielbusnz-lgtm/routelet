# How the model was made

routelet's intent classifier, end to end.

## Model

SetFit: a fine-tuned `BAAI/bge-small-en-v1.5` encoder (384-d, on-device sized)
plus a logistic-regression head. Text in, one of 5 intents out.

## Pipeline

1. Data: hand-written seed commands per intent in `data/*.jsonl`, plus disfluent
   variants from `augment.py` in `data/augmented.jsonl`. ~1100 rows.
2. preprocess: every command is normalized the same way at train and inference
   (redact secrets, emails, long numbers), so train and serve never diverge. See
   `preprocess.py`; the rules are pinned by a shared fixture with the Aegis side.
3. Train (`train_setfit.py`): contrastive fine-tune (unique sampling, 2 epochs),
   then fit the LR head with class_weight balanced, then fit a temperature scalar
   for confidence calibration.
4. Eval: score on the frozen `evals/holdout.jsonl` (200 rows, 40 per intent).
5. Export (`export_onnx.py`): encoder to `embedder.onnx` (opset 14) and head to
   `head.json`, for on-device inference in Aegis via tract.

## Numbers (200-row holdout)

- routelet (SetFit): 0.89
- TF-IDF baseline: 0.60 (the floor)
- Claude Haiku: 0.94 (the bar), at ~960ms vs routelet's ~40ms on-device

## Honest limits

The data is synthetic, so the holdout number is optimistic relative to real
voice input. The real lever from here is the data loop (log real commands,
Claude relabels them, retrain), not a bigger model.

## Tool routing (explored, not shipped)

We tried routing the `integration` branch to a specific tool on-device, so it
could pick spotify/gmail/github/youtube or detect "no tool, escalate."

- Semantic retrieval, reusing the encoder: failed. The fine-tuned encoder
  collapses cosine distances to ~0.01, unusable for retrieval.
- Classify into tools, a second fine-tune: works in-distribution (0.98) but
  drops to 0.63 out-of-distribution (`evals/tool_routing_ood.jsonl`). The
  synthetic data taught surface keywords, not intent.

Not shipped. It needs real or much more varied data first. Aegis keeps letting
Claude pick the tool for now.
