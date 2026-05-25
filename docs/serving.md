# Serving and the data loop

How routelet ships into Aegis and improves over time.

## Serving: on-device, not an API

- Runs on the user's machine. No network call per command.
- Why not an API: re-adds the latency routelet exists to kill (~50-200ms round trip), breaks offline, and sends every command (including secrets) to a server.

## Inference path: ONNX + tract

- Train SetFit (bge-small) in Python. Export with `export_onnx(model_body, model_head, ...)`; the LR head is baked into the graph, so one file does embed then classify.
- Ship 3 assets beside the binary: `routelet.onnx` (int8), `tokenizer.json`, `labels.json` (class index to intent).
- Rust: `tokenizers` (encode, pad to fixed length e.g. 64) then `tract-onnx` (load, run) then argmax then `labels.json` to `Intent`. Behind one function: `classify(&str) -> Intent`.
- Verify the export: ONNX predictions must match the PyTorch model on the holdout before trusting the artifact.
- Prototype shortcut: a local Python sidecar closes the loop first; swap to tract for release.

## Footprint

- int8: ~33MB disk, ~50-100MB RAM. Dwarfed by Aegis's ASR model.

## Data loop (where labels come from)

- Device infers and logs locally: `{redacted_text, predicted_intent, corrected_intent}`.
- Labels are user corrections, not raw commands. No corrections, no signal.

## Redaction (no leaked secrets)

- Keep the intent skeleton, drop the sensitive payload: "remember my wifi password is X" becomes "remember my wifi password is `<SECRET>`".
- Redact at capture. Store redacted text only, never raw.
- Apply the same redaction in training and inference, so there is no skew and the classifier never sees a secret.
- Local-only, opt-in, gitignored, never published raw. The highest-risk class (`memory`) is structurally simple, so log it least and redact it hardest.
- Tools: regex for passwords/emails/numbers; Presidio or NER for names/addresses later.

## Retraining (central, gated)

- Not on device (no torch there). Collect, redact, retrain SetFit, export ONNX, ship the update.
- Trigger on data volume or measured drift, not a daily cron. Daily retrains on noise.
- Gate every candidate on `evals/holdout.jsonl`: promote only if it beats the current model. Keep the old ONNX for rollback.
- One global model from aggregated consented data, not a per-user model (data-starved, complex).

## Build order

1. ONNX inference in Aegis plus local correction logging.
2. Accumulate real corrections.
3. Retrain manually (run the export script).
4. Automate the trigger last, only when volume justifies it.
