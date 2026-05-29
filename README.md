# routelet

A tiny fine-tuned intent router. Classify text into a fixed set of intents in milliseconds, on-device, instead of paying for an LLM call every time you need to decide *what kind* of request something is.

The first real user is [Aegis](https://github.com/danielbusnz-lgtm/aegis), a voice assistant that routes each command into one of five paths. Doing that with a Claude call adds latency, which is rough for voice. A small fine-tuned classifier does the same job an order of magnitude faster, for free, and offline.

Early days. The plan:

1. Label a dataset of `(text, intent)` pairs.
2. Fine-tune a small model to classify them.
3. Serve it behind a fast HTTP endpoint.
4. Prove it matches the LLM baseline on accuracy while being far faster and cheaper.

The frozen intent set and labeling rules live in [docs/taxonomy.md](docs/taxonomy.md).

## Layout

The pipeline runs **data → train → export → evaluate**. Two files are the spine
everything else imports: `data.py` (the `Intent` label set and JSONL loaders)
and `preprocess.py` (the redaction/normalization applied identically at training
and inference, mirrored in the Rust consumer).

```
src/routelet/
  data.py            Intent enum (the 5 labels) + JSONL load/split. Source of truth.
  preprocess.py      mask secrets/emails/digits. Same pass at train and inference.

  augment.py         clean rows -> voice-like disfluent variants (data/augmented.jsonl)
  checkdata.py       dataset health: per-class counts, train/eval leakage

  train.py           TF-IDF + logistic-regression baseline (the floor to beat)
  train_setfit.py    the shipped model: SetFit fine-tune of bge-small + LR head,
                     then temperature calibration -> models/setfit
  setfit_baseline.py benchmark harness that trains+discards SetFit (not the ship path)
  export_onnx.py     models/setfit -> embedder.onnx + tokenizer.json + head.json (int8)

  evaluate.py        the Claude LLM baseline (the accuracy ceiling), one call per row
  teacher.py         shared taxonomy prompt + schema + classify(), used by eval + ingest
  serve.py           optional FastAPI endpoint (stub; Aegis runs the ONNX in-process)

Scripts/             runnable tools (not library code)
  pull_samples.py    pull redacted samples from the proxy's R2 -> one JSONL
  ingest_samples.py  label them (free Claude label or teacher), dedup -> data/collected.jsonl
  refresh_haiku_baseline.py   re-run the paid Haiku eval -> report/baselines.json

report/              report.py scores the models on the holdout -> figures + metrics.json
evals/               frozen eval data (holdout.jsonl). Never trained on.
experiments/         throwaway prototypes, not part of the library (see its README)
```

The data-collection loop closes back on itself: Aegis logs redacted samples ->
proxy -> R2, then `Scripts/pull_samples.py` + `ingest_samples.py` turn those into
new labeled training data.

## Data format

One JSON object per line:

```json
{"text": "play despacito on spotify", "intent": "integration"}
```

## License

MIT
