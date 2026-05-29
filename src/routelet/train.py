"""Train the intent classifier and report accuracy on the held-out eval set.

Baseline model: TF-IDF features into logistic regression. It is the floor the
fine-tuned model (the ``train`` extra) has to beat, and it doubles as routelet's
v1 served model since it fits in milliseconds and needs no GPU. Trains on the
pool in data/, evaluates on the frozen evals/holdout.jsonl, which was written
apart from the training data so the eval isn't just memorized train.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from routelet.data import Intent, load, load_dir

TRAIN_DIR = "data"
EVAL_FILE = "evals/holdout.jsonl"
MODEL_OUT = Path("models/baseline.joblib")


def main() -> None:
    train = load_dir(TRAIN_DIR)
    test = load(EVAL_FILE)

    # Word unigrams + bigrams. Bigrams catch the cues that separate these
    # intents ("remember that", "and then", "where is"), which single words miss.
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    model.fit([e.text for e in train], [e.intent.value for e in train])

    pred = model.predict([e.text for e in test])
    true = [e.intent.value for e in test]
    labels = [i.value for i in Intent]

    print(f"train {len(train)}  eval {len(test)}\n")
    print(classification_report(true, pred, labels=labels, zero_division=0))
    print("confusion (rows=true, cols=pred):", labels)
    print(confusion_matrix(true, pred, labels=labels))

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nsaved {MODEL_OUT}")


if __name__ == "__main__":
    main()
