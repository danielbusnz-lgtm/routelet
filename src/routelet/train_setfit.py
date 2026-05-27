"""Train and persist a SetFit intent classifier.

setfit_baseline.py is a benchmark harness: it trains, evaluates, then discards
the model. This script does the same training but saves the fitted model so
export_onnx.py can bake it into a runtime artifact.
"""

import random

import numpy as np
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report

from routelet.data import Intent, load, load_dir

BASE = "BAAI/bge-small-en-v1.5"
TRAIN_DIR = "data"
EVAL_FILE = "evals/holdout.jsonl"
MODEL_OUT = "models/setfit"


def main() -> None:
    # Seed the three RNGs that touch training. SetFit's contrastive pair
    # sampling runs inside sentence-transformers and can still vary slightly
    # between runs; this gets us as close to determinism as the library allows.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train = load_dir(TRAIN_DIR)
    test = load(EVAL_FILE)
    labels = [i.value for i in Intent]

    train_ds = Dataset.from_dict(
        {"text": [e.text for e in train], "label": [e.intent.value for e in train]}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"training on {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
    )

    model = SetFitModel.from_pretrained(BASE, labels=labels, device=device)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(batch_size=16, num_epochs=1),
        train_dataset=train_ds,
    )
    trainer.train()

    texts = [e.text for e in test]
    true = [e.intent.value for e in test]
    preds = list(model.predict(texts))

    print(f"\nbase {BASE}   train {len(train)}   eval {len(test)}\n")
    print(classification_report(true, preds, labels=labels, zero_division=0))

    model.save_pretrained(MODEL_OUT)
    print(f"saved {MODEL_OUT}")


if __name__ == "__main__":
    main()
