"""Dataset health check: per-intent counts and train/eval leakage.

Run as ``python -m routelet.checkdata`` while hand-writing evals/holdout.jsonl.
Watches per-intent balance, flags any eval line that also appears in the training
pool (which would invalidate the eval), and shows how far each intent is from the
target count.
"""

from collections import Counter

from routelet.data import Intent, load, load_dir

TRAIN_DIR = "data"
EVAL_FILE = "evals/holdout.jsonl"
TARGET_PER_INTENT = 30


def _counts(examples) -> dict[str, int]:
    c = Counter(e.intent.value for e in examples)
    return {i.value: c.get(i.value, 0) for i in Intent}


def main() -> None:
    train = load_dir(TRAIN_DIR)
    eval_set = load(EVAL_FILE)

    print(f"train pool ({len(train)}): {_counts(train)}")
    print(f"eval set   ({len(eval_set)}): {_counts(eval_set)}")

    train_texts = {e.text.lower().strip() for e in train}
    leaks = [e.text for e in eval_set if e.text.lower().strip() in train_texts]
    print(f"leakage (eval lines also in train): {leaks or 'none'}")

    short = {
        k: TARGET_PER_INTENT - n
        for k, n in _counts(eval_set).items()
        if n < TARGET_PER_INTENT
    }
    print(f"to reach {TARGET_PER_INTENT}/intent, still need: {short or 'target met'}")


if __name__ == "__main__":
    main()
