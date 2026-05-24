"""SetFit candidate: a few-shot sentence-transformer classifier.

Same train/eval as train.py (synthetic pool in data/, hand-written seed as eval),
so its accuracy lands directly beside the TF-IDF floor and the Claude bar. SetFit
fine-tunes a small sentence-transformer with contrastive pairs, then fits a
logistic-regression head on the embeddings: semantic, tiny, millisecond-fast.
"""

import time

import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

from routelet.data import Intent, load, load_dir

# all-MiniLM-L6-v2 is tiny and fast, which keeps routelet's speed pitch intact.
# Swap to BAAI/bge-small-en-v1.5 for a stronger (still small) encoder.
BASE = "BAAI/bge-small-en-v1.5"
TRAIN_DIR = "data"
EVAL_FILE = "evals/holdout.jsonl"


def main() -> None:
    train = load_dir(TRAIN_DIR)
    test = load(EVAL_FILE)
    labels = [i.value for i in Intent]

    train_ds = Dataset.from_dict(
        {"text": [e.text for e in train], "label": [e.intent.value for e in train]}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    model = SetFitModel.from_pretrained(BASE, labels=labels, device=device)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(batch_size=16, num_epochs=1),
        train_dataset=train_ds,
    )
    trainer.train()

    texts = [e.text for e in test]
    true = [e.intent.value for e in test]

    model.predict([texts[0]])  # warm up before timing
    lat = []
    for t in texts:
        s = time.perf_counter()
        model.predict([t])
        lat.append(time.perf_counter() - s)
    preds = list(model.predict(texts))
    lat.sort()

    print(f"\nbase {BASE}   train {len(train)}   eval {len(test)}")
    print(f"accuracy   {accuracy_score(true, preds):.2f}")
    print(f"latency    p50 {lat[len(lat) // 2] * 1000:.1f} ms/command (single, {device})")
    print()
    print(classification_report(true, preds, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
