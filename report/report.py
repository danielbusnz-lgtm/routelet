"""Reproducible performance report. Scores the models on the frozen holdout and
writes figures plus metrics.json. Re-run after every retrain.

Scoring rules (docs/metrics.md):
  * Every model is scored on the same frozen evals/holdout.jsonl.
  * Each model is fed text the way it was trained: the TF-IDF baseline on raw
    text, the SetFit model on preprocess()'d text.
  * Claude Haiku numbers come from report/baselines.json (cached so we don't
    re-spend on the paid API every regen). If that cache was measured on a
    different holdout size than the current one, it is flagged, not silently
    plotted as comparable.

matplotlib only, installed into the venv (not pyproject).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from routelet.data import load  # noqa: E402
from routelet.preprocess import preprocess  # noqa: E402

HOLDOUT = PROJECT_ROOT / "evals" / "holdout.jsonl"
TFIDF_MODEL = PROJECT_ROOT / "models" / "baseline.joblib"
SETFIT_DIR = PROJECT_ROOT / "models" / "setfit"
BASELINES = PROJECT_ROOT / "report" / "baselines.json"
OUT_DIR = PROJECT_ROOT / "report"

# Okabe-Ito (colorblind-safe). Gray baseline, green hero (routelet), amber oracle.
C_TFIDF = "#999999"
C_ROUTELET = "#009E73"
C_HAIKU = "#E69F00"


def score_tfidf(texts: list[str], gold: list[str]) -> float:
    """TF-IDF + logistic regression baseline. Trained on raw text, so score on
    raw text (no preprocess) to match its training."""
    model = joblib.load(TFIDF_MODEL)
    preds = model.predict(texts)
    return float(accuracy_score(gold, preds))


def score_setfit(texts: list[str], gold: list[str]) -> float:
    """The shipped SetFit model. Trained on preprocess()'d text, so score the
    same way. This is the fine-tuned bge-small body + LR head in torch; int8
    ONNX (what Rust actually runs) is verified equivalent at export time."""
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(SETFIT_DIR))
    preds = list(model.predict([preprocess(t) for t in texts]))
    return float(accuracy_score(gold, preds))


def load_haiku(eval_n: int) -> dict | None:
    """Read the cached Claude baseline. Returns the record with a `stale` flag
    set when it was measured on a different holdout size than the current one."""
    if not BASELINES.exists():
        return None
    data = json.loads(BASELINES.read_text())
    haiku = data.get("haiku")
    if not haiku:
        return None
    haiku["stale"] = haiku.get("eval_n") != eval_n
    return haiku


def plot_model_comparison(metrics: dict) -> Path:
    """Figure 1: accuracy on the frozen holdout, baseline vs shipped vs oracle."""
    bars = [
        ("TF-IDF\nbaseline", metrics["tfidf"]["accuracy"], C_TFIDF, "raw text, in-proc"),
        ("routelet\n(SetFit, on-device)", metrics["setfit"]["accuracy"], C_ROUTELET,
         "on-device, free"),
    ]
    haiku = metrics.get("haiku")
    if haiku:
        note = f"{haiku['latency_p50_ms']:.0f}ms p50, cloud"
        if haiku.get("stale"):
            note += f"\n(n={haiku['eval_n']}, stale)"
        bars.append(("Claude Haiku\n(LLM oracle)", haiku["accuracy"], C_HAIKU, note))

    labels = [b[0] for b in bars]
    accs = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    notes = [b[3] for b in bars]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(bars))
    ax.bar(x, accs, color=colors, width=0.62, zorder=3)

    for i, (acc, note) in enumerate(zip(accs, notes)):
        ax.text(i, acc + 0.015, f"{acc:.0%}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.text(i, 0.04, note, ha="center", va="bottom", fontsize=8,
                color="white", fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy on frozen holdout")
    ax.set_title(f"Intent routing accuracy (n={metrics['eval_n']})", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out = OUT_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    examples = load(HOLDOUT)
    texts = [e.text for e in examples]
    gold = [e.intent.value for e in examples]
    eval_n = len(examples)

    print(f"scoring on {eval_n}-row holdout")
    tfidf_acc = score_tfidf(texts, gold)
    print(f"  TF-IDF  {tfidf_acc:.3f}")
    setfit_acc = score_setfit(texts, gold)
    print(f"  SetFit  {setfit_acc:.3f}")

    metrics: dict = {
        "eval_n": eval_n,
        "tfidf": {"accuracy": tfidf_acc},
        "setfit": {"accuracy": setfit_acc},
    }
    haiku = load_haiku(eval_n)
    if haiku:
        metrics["haiku"] = haiku
        if haiku["stale"]:
            print(f"  Haiku   {haiku['accuracy']:.3f}  [STALE: cached on n={haiku['eval_n']}, "
                  f"current holdout is n={eval_n}; refresh with routelet.evaluate]")
        else:
            print(f"  Haiku   {haiku['accuracy']:.3f}  (cached)")

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig = plot_model_comparison(metrics)
    print(f"wrote {fig}")
    print(f"wrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
