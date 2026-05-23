# routelet

A tiny fine-tuned intent router. Classify text into a fixed set of intents in milliseconds, on-device, instead of paying for an LLM call every time you need to decide *what kind* of request something is.

The first real user is [Aegis](https://github.com/danielbusnz-lgtm/aegis), a voice assistant that routes each command into one of five paths. Doing that with a Claude call adds latency, which is rough for voice. A small fine-tuned classifier does the same job an order of magnitude faster, for free, and offline.

Early days. The plan:

1. Label a dataset of `(text, intent)` pairs.
2. Fine-tune a small model to classify them.
3. Serve it behind a fast HTTP endpoint.
4. Prove it matches the LLM baseline on accuracy while being far faster and cheaper.

## Layout

```
src/routelet/
  data.py       label schema and train/test prep
  train.py      fine-tune a small model on the labels
  serve.py      FastAPI endpoint: text in, intent out
  evaluate.py   routelet vs the LLM baseline (accuracy, latency, cost)
examples/
  intents.sample.jsonl   example label format
```

## Data format

One JSON object per line:

```json
{"text": "play despacito on spotify", "intent": "integration"}
```

## License

MIT
