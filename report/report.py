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
OOD_PROBE = PROJECT_ROOT / "report" / "ood_probe.txt"
OUT_DIR = PROJECT_ROOT / "report"

# Okabe-Ito (colorblind-safe). Gray baseline, green hero (routelet), amber oracle.
C_TFIDF = "#999999"
C_ROUTELET = "#009E73"
C_HAIKU = "#E69F00"
# In-distribution vs OOD: blue and vermillion, distinguishable for colorblind
# readers (and labeled, so not relying on hue alone).
C_INDIST = "#0072B2"
C_OOD = "#D55E00"

# Routelet's on-device confidence gate: below this, Aegis defers to the Claude
# fallback. Mirrors ROUTELET_CONFIDENCE_THRESHOLD in the aegis crate's tuning.rs.
GATE = 0.55


def score_tfidf(texts: list[str], gold: list[str]) -> float:
    """TF-IDF + logistic regression baseline. Trained on raw text, so score on
    raw text (no preprocess) to match its training."""
    model = joblib.load(TFIDF_MODEL)
    preds = model.predict(texts)
    return float(accuracy_score(gold, preds))


def _load_temperature() -> float:
    """The calibrated temperature baked into the model dir, or 1.0 if absent.
    Must match what Aegis applies so the gate confidence here equals production."""
    path = SETFIT_DIR / "temperature.json"
    if path.exists():
        return float(json.loads(path.read_text())["temperature"])
    return 1.0


def load_setfit() -> dict:
    """Load the shipped SetFit model once into a reusable bundle (body + head
    weights + calibrated temperature), so the holdout and the OOD probe can both
    be scored without reloading the model."""
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(SETFIT_DIR))
    head = model.model_head
    return {
        "body": model.model_body,
        "coef": head.coef_,
        "intercept": head.intercept_,
        "labels": head.classes_.tolist(),
        "temperature": _load_temperature(),
    }


def setfit_predict(bundle: dict, texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Run the model the way Aegis does: embed preprocess()'d text, apply the LR
    head, temperature-scale, softmax, argmax. Returns (max-softmax confidence per
    row, predicted labels). The int8 ONNX Aegis ships is verified equivalent at
    export time."""
    emb = bundle["body"].encode(
        [preprocess(t) for t in texts], convert_to_numpy=True, show_progress_bar=False
    )
    logits = (emb @ bundle["coef"].T + bundle["intercept"]) / bundle["temperature"]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    idx = probs.argmax(axis=1)
    conf = probs[np.arange(len(idx)), idx]
    preds = [bundle["labels"][i] for i in idx]
    return conf, preds


def load_ood_probes() -> list[str]:
    """Read the OOD/garbled probe lines (skipping blanks and # comments)."""
    return [
        ln.strip()
        for ln in OOD_PROBE.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


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


def plot_confidence_histogram(indist: np.ndarray, ood: np.ndarray, gate: float) -> Path:
    """Figure 2: routelet confidence on in-distribution holdout vs OOD/garbled
    probes, with the gate line. The cascade works if in-distribution input sits
    above the gate (kept on-device) while OOD input falls below it (deferred to
    Claude). Overlapping (not stacked) histograms, densities so the two groups
    are comparable despite different counts."""
    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(indist, bins=bins, color=C_INDIST, alpha=0.6, density=True,
            label=f"in-distribution holdout (n={len(indist)})", zorder=3)
    ax.hist(ood, bins=bins, color=C_OOD, alpha=0.7, density=True,
            label=f"OOD / garbled probe (n={len(ood)})", zorder=3)

    ax.axvline(gate, color="black", linestyle="--", linewidth=1.2, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.text(gate - 0.012, ymax * 0.96, f"{gate:.2f} gate", ha="right", va="top", fontsize=9)
    ax.text(gate - 0.012, ymax * 0.55, "← defer to Claude", ha="right", fontsize=8, color="#555")
    ax.text(gate + 0.012, ymax * 0.55, "kept on-device →", ha="left", fontsize=8, color="#555")

    ood_deferred = int((ood < gate).sum())
    indist_kept = int((indist >= gate).sum())
    ax.set_title(
        f"OOD defers {ood_deferred}/{len(ood)}, in-distribution keeps "
        f"{indist_kept}/{len(indist)} at the {gate:.2f} gate",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("routelet confidence (temperature-scaled max softmax)")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out = OUT_DIR / "confidence_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_deferral_tradeoff(indist: np.ndarray, ood: np.ndarray, gate: float) -> Path:
    """Figure 3: as the confidence cutoff rises, what fraction of in-distribution
    commands get wrongly deferred to Claude vs what fraction of OOD/garbled input
    gets caught. The two lines never separate cleanly, which is why no cutoff
    makes the gate work: catching OOD means deferring real commands too."""
    thresholds = np.linspace(0.5, 1.0, 101)
    id_deferred = np.array([(indist < t).mean() for t in thresholds]) * 100
    ood_caught = np.array([(ood < t).mean() for t in thresholds]) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thresholds, ood_caught, color=C_OOD, linewidth=2.4,
            label="OOD / garbled caught (good)", zorder=3)
    ax.plot(thresholds, id_deferred, color=C_INDIST, linewidth=2.4,
            label="real commands wrongly deferred (bad)", zorder=3)

    ax.axvline(gate, color="#777", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(gate, 102, f"current cutoff {gate:.2f}", ha="center", va="bottom",
            fontsize=8, color="#555")

    ax.set_xlabel("confidence cutoff (defer to Claude below it)")
    ax.set_ylabel("% of inputs deferred")
    ax.set_title(
        "No cutoff works: catching garbage means deferring real commands too",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out = OUT_DIR / "deferral_tradeoff.png"
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

    bundle = load_setfit()
    conf, preds = setfit_predict(bundle, texts)
    correct = np.array([p == g for p, g in zip(preds, gold)])
    setfit_acc = float(correct.mean())
    print(f"  SetFit  {setfit_acc:.3f}")

    # OOD/garbled probe: confidence on inputs the gate is meant to defer.
    ood_texts = load_ood_probes()
    ood_conf, _ = setfit_predict(bundle, ood_texts)

    # Confidence gate operating point: what the cascade does at GATE.
    indist_kept = conf >= GATE
    ood_deferred = ood_conf < GATE
    gate_stats = {
        "threshold": GATE,
        "in_distribution_kept_share": round(float(indist_kept.mean()), 3),
        "ood_deferred_share": round(float(ood_deferred.mean()), 3),
        "kept_accuracy": (
            round(float(correct[indist_kept].mean()), 3) if indist_kept.any() else None
        ),
    }
    print(f"  gate {GATE}: in-dist keeps {gate_stats['in_distribution_kept_share']:.0%}, "
          f"OOD defers {gate_stats['ood_deferred_share']:.0%}")

    metrics: dict = {
        "eval_n": eval_n,
        "tfidf": {"accuracy": tfidf_acc},
        "setfit": {"accuracy": setfit_acc},
        "gate": gate_stats,
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
    print(f"wrote {plot_model_comparison(metrics)}")
    print(f"wrote {plot_confidence_histogram(conf, ood_conf, GATE)}")
    print(f"wrote {plot_deferral_tradeoff(conf, ood_conf, GATE)}")
    print(f"wrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
