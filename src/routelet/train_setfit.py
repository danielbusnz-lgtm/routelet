"""Train and persist a SetFit intent classifier.

setfit_baseline.py is a benchmark harness: it trains, evaluates, then discards
the model. This script does the same training but saves the fitted model so
export_onnx.py can bake it into a runtime artifact.

After saving the model a single temperature-scaling parameter T is fitted on a
stratified 20% calibration split carved from the original (non-augmented) data
files only (data/{agent,chat,find_action,integration,memory}.jsonl). T is
written to models/setfit/temperature.json so export_onnx.py can bake it into
head.json for the Rust consumer.
"""

import json
import random
from pathlib import Path

import numpy as np
import scipy.optimize
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from routelet.data import Example, Intent, load, load_dir
from routelet.preprocess import preprocess

BASE = "BAAI/bge-small-en-v1.5"
TRAIN_DIR = "data"
ORIGINAL_FILES = [
    "data/agent.jsonl",
    "data/chat.jsonl",
    "data/find_action.jsonl",
    "data/integration.jsonl",
    "data/memory.jsonl",
]
EVAL_FILE = "evals/holdout.jsonl"
MODEL_OUT = "models/setfit"
TEMPERATURE_OUT = "models/setfit/temperature.json"
CALIB_FRAC = 0.20
CALIB_SEED = 7


def _ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error over equally-spaced confidence bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def calibrate_temperature(
    model: SetFitModel,
    calib_examples: list[Example],
    labels: list[str],
) -> float:
    """Fit a scalar temperature T on calib_examples.

    Minimizes NLL of softmax(logits / T) against true labels with
    scipy.optimize.minimize_scalar (bounded method) over T in [0.5, 20.0].

    Returns T > 0.

    IMPORTANT: logit columns follow head.classes_ order (sklearn alphabetical),
    NOT the Intent enum order. label_to_idx must be built from head.classes_.
    """
    # head.classes_ gives the column ordering for coef_ / logits (sklearn alphabetical).
    head_labels = model.model_head.classes_.tolist()
    label_to_idx = {lbl: i for i, lbl in enumerate(head_labels)}
    texts = [preprocess(e.text) for e in calib_examples]
    true_idx = np.array([label_to_idx[e.intent.value] for e in calib_examples])

    # Raw embeddings from the body (already L2-normalized by the pipeline).
    embeddings = model.model_body.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    coef = model.model_head.coef_          # (n_classes, n_features)
    intercept = model.model_head.intercept_  # (n_classes,)
    logits = embeddings @ coef.T + intercept  # (n_samples, n_classes)

    n = len(true_idx)

    def neg_log_likelihood(T: float) -> float:
        probs = _softmax(logits / T)
        # Clip to avoid log(0).
        p_true = probs[np.arange(n), true_idx].clip(1e-12, 1.0)
        return -np.log(p_true).mean()

    # Search T in [0.5, 20.0]. T < 1 sharpens the distribution; T > 1 softens
    # it. We do not allow T below 0.5 because on a well-calibrated model with
    # large logit margins the NLL optimizer would push T toward 0, collapsing
    # all softmax outputs to ~1.0 and making the confidence gate degenerate.
    result = scipy.optimize.minimize_scalar(
        neg_log_likelihood,
        bounds=(0.5, 20.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    T_opt = float(result.x)

    # If the optimizer hit the lower bound (T_opt ~= 0.5) and the model is
    # already well-calibrated at T=1.0 (ECE < 0.05), clamp to T=1.0.
    # Rationale: T < 1 only sharpens an already-confident model further.
    # The Rust consumer uses max-softmax as its confidence gate; a T that
    # pushes all probs to 1.0 destroys the gate's discriminating power.
    probs_before = _softmax(logits)
    conf_before = probs_before.max(axis=1)
    correct = (probs_before.argmax(axis=1) == true_idx)
    ece_t1 = _ece(conf_before, correct.astype(float))

    if T_opt < 1.0 and ece_t1 < 0.05:
        T = 1.0
        clamped_note = f" (clamped from {T_opt:.4f}; ECE@T=1 already {ece_t1:.4f} < 0.05)"
    else:
        T = T_opt
        clamped_note = ""

    probs_after = _softmax(logits / T)
    conf_after = probs_after.max(axis=1)

    ece_before = ece_t1
    ece_after = _ece(conf_after, correct.astype(float))

    print(f"\n--- temperature calibration (calib split n={n}) ---")
    print(f"T = {T:.4f}{clamped_note}")
    print(f"ECE before scaling (T=1.0): {ece_before:.4f}")
    print(f"ECE after  scaling (T={T:.4f}): {ece_after:.4f}")
    print(f"mean confidence before: {conf_before.mean():.3f}")
    print(f"mean confidence after:  {conf_after.mean():.3f}")
    print(f"accuracy on calib split: {correct.mean():.3f}")
    print("---")

    return T


def threshold_analysis(
    model: SetFitModel,
    test_examples: list[Example],
    labels: list[str],
    T: float,
) -> None:
    """Print deferral / accuracy table for candidate confidence thresholds.

    Shows two views:
    - T-scaled (T = fitted value): the confidence the Rust consumer will see.
    - raw T=1.0: the uncalibrated softmax output, useful as a sanity-check
      when T < 1.0 pushes all probs near 1.0 (degenerate gating).
    """
    # Use head.classes_ ordering to match logit column ordering.
    head_labels = model.model_head.classes_.tolist()
    label_to_idx = {lbl: i for i, lbl in enumerate(head_labels)}
    texts = [preprocess(e.text) for e in test_examples]
    true_idx = np.array([label_to_idx[e.intent.value] for e in test_examples])

    embeddings = model.model_body.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    coef = model.model_head.coef_
    intercept = model.model_head.intercept_
    logits = embeddings @ coef.T + intercept

    # Predictions are the same for any positive T (argmax is scale-invariant).
    pred_idx = logits.argmax(axis=1)
    correct = pred_idx == true_idx

    n = len(test_examples)

    def _print_table(conf: np.ndarray, label: str) -> None:
        print(f"\n--- confidence threshold analysis ({n}-row holdout, {label}) ---")
        print(
            f"{'threshold':>10}  {'deferred':>10}  {'defer%':>8}  "
            f"{'kept_acc':>10}  {'defer_acc':>10}"
        )
        for tau in [0.55, 0.60, 0.65, 0.70]:
            above = conf >= tau
            below = ~above
            n_below = int(below.sum())
            frac_below = n_below / n
            kept_acc = float(correct[above].mean()) if above.sum() > 0 else float("nan")
            defer_acc = float(correct[below].mean()) if below.sum() > 0 else float("nan")
            print(
                f"{tau:>10.2f}  {n_below:>10d}  {frac_below:>8.1%}  "
                f"{kept_acc:>10.3f}  {defer_acc:>10.3f}"
            )
        print("---")

    # T-scaled view (what Rust sees).
    probs_scaled = _softmax(logits / T)
    _print_table(probs_scaled.max(axis=1), f"T={T:.4f} scaled")

    # Raw T=1 view (always shown for comparison).
    probs_raw = _softmax(logits)
    _print_table(probs_raw.max(axis=1), "T=1.0 raw")

    # Flag the two low-confidence rows so we can see what's hard.
    conf_raw = probs_raw.max(axis=1)
    uncertain = [(i, conf_raw[i], test_examples[i].intent.value, head_labels[pred_idx[i]])
                 for i in range(n) if conf_raw[i] < 0.90]
    if uncertain:
        print("\nlow-confidence rows on holdout (raw T=1.0 max-prob < 0.90):")
        for idx, c, true_lbl, pred_lbl in uncertain:
            mark = "OK" if true_lbl == pred_lbl else "WRONG"
            print(
                f"  [{idx}] conf={c:.3f} true={true_lbl!r} pred={pred_lbl!r} "
                f"[{mark}]  {test_examples[idx].text!r}"
            )


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

    # Load original (non-augmented) data for the calibration split.
    original_examples: list[Example] = []
    for path in ORIGINAL_FILES:
        original_examples.extend(load(path))
    print(f"original (non-augmented) pool: {len(original_examples)} examples")

    # Stratified 20% calibration split from original data only.
    orig_texts = [e.text for e in original_examples]
    orig_labels_str = [e.intent.value for e in original_examples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=CALIB_FRAC, random_state=CALIB_SEED)
    _, calib_idx = next(sss.split(orig_texts, orig_labels_str))
    calib_examples = [original_examples[i] for i in calib_idx]
    print(f"calibration split: {len(calib_examples)} examples (stratified 20%)")

    train_ds = Dataset.from_dict(
        {
            "text": [preprocess(e.text) for e in train],
            "label": [e.intent.value for e in train],
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"training on {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
    )

    # class_weight="balanced" counters the over-firing on agent/find_action by
    # up-weighting under-represented classes in the LR head.
    model = SetFitModel.from_pretrained(
        BASE,
        labels=labels,
        device=device,
        head_params={"class_weight": "balanced"},
    )
    # Verify the LR head actually received class_weight="balanced".
    cw = model.model_head.get_params().get("class_weight")
    assert cw == "balanced", f"class_weight not applied; got {cw!r}"
    print(f"head class_weight: {cw}")

    # sampling_strategy="unique": draws every sentence-pair combination exactly
    # once (no duplication). Valid in setfit 1.1.3 (confirmed in source).
    # num_epochs=2 doubles the embedding training time vs. the previous 1-epoch
    # run, giving the contrastive head more signal on the larger 1115-row pool.
    args = TrainingArguments(
        batch_size=16,
        num_epochs=2,
        sampling_strategy="unique",
    )
    print("\nTrainingArguments:")
    print(f"  batch_size:        {args.batch_size}")
    print(f"  num_epochs:        {args.num_epochs}")
    print(f"  sampling_strategy: {args.sampling_strategy}")
    assert args.sampling_strategy == "unique", (
        f"sampling_strategy not applied; got {args.sampling_strategy!r}"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )
    trainer.train()

    # Holdout evaluation.
    texts = [preprocess(e.text) for e in test]
    true = [e.intent.value for e in test]
    preds = list(model.predict(texts))

    print(f"\nbase {BASE}   train {len(train)}   eval {len(test)}\n")
    print(classification_report(true, preds, labels=labels, zero_division=0))

    # Confusion matrix.
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true, preds, labels=labels)
    print("confusion matrix (rows=true, cols=pred):")
    col_width = max(len(lbl) for lbl in labels) + 2
    header = " " * col_width + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_str = f"{row_label:<{col_width}}" + "".join(f"{v:>{col_width}}" for v in cm[i])
        print(row_str)

    model.save_pretrained(MODEL_OUT)
    print(f"\nsaved {MODEL_OUT}")

    # Fit temperature on calibration split.
    T = calibrate_temperature(model, calib_examples, labels)

    # Persist T alongside the model for export_onnx.py to pick up.
    Path(TEMPERATURE_OUT).write_text(json.dumps({"temperature": T}) + "\n")
    print(f"saved {TEMPERATURE_OUT}  (T={T:.4f})")

    # Threshold / deferral analysis on the frozen holdout.
    threshold_analysis(model, test, labels, T)


if __name__ == "__main__":
    main()
