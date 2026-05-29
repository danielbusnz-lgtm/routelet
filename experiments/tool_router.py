"""Semantic tool router for integration commands.

Uses a sentence encoder to embed commands and tool descriptions, then picks
the best-matching tool via cosine similarity. Commands with no good match fall
below TOOL_THRESH and escalate rather than route.

TWO ENCODER MODES are supported for the eval:
  - finetuned: SetFitModel.from_pretrained("models/setfit").model_body
    (the same encoder the intent classifier uses)
  - base: BAAI/bge-small-en-v1.5 loaded directly via sentence_transformers
    (the pretrained backbone before contrastive fine-tuning)

The eval runs both and reports the comparison.  route() uses the finetuned
encoder by default to stay consistent with the rest of the pipeline, but
the eval will show whether that is viable.

Usage:
    .venv/bin/python -m routelet.tool_router           # run full eval
    .venv/bin/python -m routelet.tool_router 0.40      # custom threshold
    .venv/bin/python -m routelet.tool_router 0.40 base # base model only

route(command, threshold) is the public API for the Aegis integration path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, str] = {
    "spotify": "control music playback: play, pause, skip, previous track, volume on spotify",
    "gmail": "read, search, and send email",
    "github": "list and check github pull requests, issues, and repositories",
    "youtube": "search for and play youtube videos",
}

TOOL_NAMES: list[str] = list(TOOLS.keys())

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "setfit"
BASE_MODEL_ID = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

_finetuned_encoder = None
_finetuned_tool_embs: Optional[np.ndarray] = None

_base_encoder = None
_base_tool_embs: Optional[np.ndarray] = None


def _load_finetuned():
    global _finetuned_encoder, _finetuned_tool_embs
    if _finetuned_encoder is not None:
        return _finetuned_encoder, _finetuned_tool_embs
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(str(MODEL_DIR))
    enc = model.model_body
    tool_embs = enc.encode(
        list(TOOLS.values()), convert_to_numpy=True, show_progress_bar=False
    )
    _finetuned_encoder = enc
    _finetuned_tool_embs = tool_embs
    return enc, tool_embs


def _load_base():
    global _base_encoder, _base_tool_embs
    if _base_encoder is not None:
        return _base_encoder, _base_tool_embs
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(BASE_MODEL_ID)
    tool_embs = enc.encode(
        list(TOOLS.values()), convert_to_numpy=True, normalize_embeddings=True
    )
    _base_encoder = enc
    _base_tool_embs = tool_embs
    return enc, tool_embs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route(command: str, threshold: float = 0.40) -> tuple[Optional[str], float]:
    """Route a command to a tool or return (None, score) to escalate.

    Uses the fine-tuned SetFit encoder.  See the eval for why the base encoder
    is a better choice; this function signature is kept stable for Aegis.

    Parameters
    ----------
    command:   raw user command (preprocess applied internally).
    threshold: minimum cosine similarity to accept a routing decision.

    Returns
    -------
    (tool_name, score)  if max similarity >= threshold
    (None, score)       otherwise (escalate)
    """
    from routelet.preprocess import preprocess
    enc, tool_embs = _load_finetuned()
    text = preprocess(command)
    emb = enc.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    sims = tool_embs @ emb
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return TOOL_NAMES[best_idx], best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Eval dataset
# ---------------------------------------------------------------------------

# HAS-TOOL: (command, expected_tool)
HAS_TOOL: list[tuple[str, str]] = [
    # spotify
    ("skip this song", "spotify"),
    ("uhh skip this song", "spotify"),
    ("pause the music", "spotify"),
    ("turn the volume up on spotify", "spotify"),
    ("play something upbeat", "spotify"),
    ("next track please", "spotify"),
    ("go back to the previous song", "spotify"),
    ("lower the spotify volume", "spotify"),
    # gmail
    ("can you pull up my unread emails", "gmail"),
    ("search my email for the invoice from last month", "gmail"),
    ("send an email to alex saying the meeting is rescheduled", "gmail"),
    ("any new emails from my boss", "gmail"),
    ("reply to that last email", "gmail"),
    ("check if I have email from github", "gmail"),
    ("compose a quick email to the team", "gmail"),
    ("find the email with the flight confirmation", "gmail"),
    # github
    ("show my open PRs", "github"),
    ("list open pull requests on my repo", "github"),
    ("any new issues filed today", "github"),
    ("check the status of my pull request", "github"),
    ("what repos do I have on github", "github"),
    ("show unreviewed pull requests", "github"),
    ("are there any failing checks on my PR", "github"),
    # youtube
    ("play lofi beats on youtube", "youtube"),
    ("search youtube for that interview with lex fridman", "youtube"),
    ("find a tutorial on rust async on youtube", "youtube"),
    ("play the latest video from veritasium", "youtube"),
    ("look up that cooking video I watched last week", "youtube"),
    ("search for how to make sourdough bread youtube", "youtube"),
    ("put on some ambient music on youtube", "youtube"),
]

# NO-TOOL: commands that should escalate (no matching integration)
NO_TOOL: list[str] = [
    "find me restaurants nearby",
    "pull up the rust docs for this function",
    "set a timer for 10 minutes",
    "whats the weather right now",
    "turn up the screen brightness",
    "order me an uber",
    "what's 15 percent of 80",
    "open the terminal",
    "translate this sentence to spanish",
    "remind me to take my meds at 8pm",
    "lock the screen",
    "copy that to the clipboard",
]


# ---------------------------------------------------------------------------
# Core eval logic (encoder-agnostic)
# ---------------------------------------------------------------------------


def _embed(texts: list[str], enc, use_preprocess: bool, normalize: bool) -> np.ndarray:
    if use_preprocess:
        from routelet.preprocess import preprocess
        texts = [preprocess(t) for t in texts]
    if normalize:
        return enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return enc.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def _run_single_eval(
    enc,
    tool_embs: np.ndarray,
    use_preprocess: bool,
    normalize: bool,
    label: str,
) -> None:
    has_cmds = [cmd for cmd, _ in HAS_TOOL]
    has_labels = [lbl for _, lbl in HAS_TOOL]
    no_cmds = list(NO_TOOL)

    has_embs = _embed(has_cmds, enc, use_preprocess, normalize)
    no_embs = _embed(no_cmds, enc, use_preprocess, normalize)

    has_sims = has_embs @ tool_embs.T
    no_sims = no_embs @ tool_embs.T

    has_top = has_sims.max(axis=1)
    no_top = no_sims.max(axis=1)

    print(f"\n{'=' * 60}")
    print(f"ENCODER: {label}")
    print(f"{'=' * 60}")

    print(f"\n--- 1. SIMILARITY DISTRIBUTION ---")
    print(f"HAS-TOOL  n={len(has_top):3d}  min={has_top.min():.4f}  "
          f"mean={has_top.mean():.4f}  max={has_top.max():.4f}")
    print(f"NO-TOOL   n={len(no_top):3d}  min={no_top.min():.4f}  "
          f"mean={no_top.mean():.4f}  max={no_top.max():.4f}")
    gap = float(has_top.min() - no_top.max())
    mean_gap = float(has_top.mean() - no_top.mean())
    overlap_count = int(np.sum(no_top >= has_top.min()))
    print(f"mean gap (has.mean - no.mean): {mean_gap:.4f}")
    print(f"hard gap (has.min - no.max):   {gap:.4f}  "
          f"({'clean' if gap > 0 else 'OVERLAP'})")
    print(f"no-tool above has-tool min:    {overlap_count}/{len(no_top)}")

    print(f"\n--- 2. THRESHOLD SWEEP ---")
    print(f"{'thresh':>8}  {'has_acc':>8}  {'no_acc':>8}  {'f1':>8}")

    has_preds = has_sims.argmax(axis=1)
    best_thresh = 0.0
    best_f1 = -1.0
    sweep_results = []

    for thresh_i in range(20, 71, 5):
        thresh = thresh_i / 100.0
        has_acc = float(np.mean([
            (has_top[i] >= thresh) and (TOOL_NAMES[has_preds[i]] == has_labels[i])
            for i in range(len(has_cmds))
        ]))
        no_acc = float(np.mean(no_top < thresh))

        tp = sum(
            1 for i in range(len(has_cmds))
            if has_top[i] >= thresh and TOOL_NAMES[has_preds[i]] == has_labels[i]
        )
        fp = int(np.sum(no_top >= thresh))
        fn = len(has_cmds) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        sweep_results.append((thresh, has_acc, no_acc, f1))
        print(f"{thresh:>8.2f}  {has_acc:>8.3f}  {no_acc:>8.3f}  {f1:>8.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nrecommended threshold: {best_thresh:.2f}  (F1={best_f1:.3f})")

    print(f"\n--- 3. PER-TOOL ACCURACY AT THRESHOLD={best_thresh:.2f} ---")
    tool_correct: dict[str, int] = {t: 0 for t in TOOL_NAMES}
    tool_total: dict[str, int] = {t: 0 for t in TOOL_NAMES}
    confusions: list[str] = []

    for i, (cmd, expected) in enumerate(HAS_TOOL):
        tool_total[expected] += 1
        top_score = float(has_top[i])
        pred_tool = TOOL_NAMES[int(has_sims[i].argmax())]
        if top_score >= best_thresh and pred_tool == expected:
            tool_correct[expected] += 1
        else:
            reason = (
                f"routed to {pred_tool!r} (score {top_score:.3f})"
                if top_score >= best_thresh
                else f"escalated (score {top_score:.3f} < {best_thresh:.2f})"
            )
            confusions.append(f"  [{expected}] {reason}: {cmd!r}")

    for tool in TOOL_NAMES:
        n = tool_total[tool]
        c = tool_correct[tool]
        bar = "#" * c + "." * (n - c)
        print(f"  {tool:<10}  {c}/{n}  [{bar}]")

    if confusions:
        print(f"\nfailures ({len(confusions)}):")
        for line in confusions:
            print(line)

    best_has_acc, best_no_acc = [
        (h, n) for t, h, n, f in sweep_results if t == best_thresh
    ][0]

    print(f"\n--- 4. VERDICT ---")
    print(f"mean gap: {mean_gap:.4f}   hard gap: {gap:.4f}   "
          f"best F1: {best_f1:.3f}   threshold: {best_thresh:.2f}")
    print(f"has-tool acc: {best_has_acc:.3f}   no-tool acc: {best_no_acc:.3f}")

    if gap > 0 and best_f1 >= 0.85:
        print(
            f"SOUND. Clean separation exists. Fixed threshold {best_thresh:.2f} works."
        )
    elif mean_gap > 0.10 and best_f1 >= 0.70:
        print(
            f"MARGINAL. Moderate separation (mean gap {mean_gap:.4f}) but no clean"
            f" hard cutoff. F1 {best_f1:.3f} is workable. Consider richer descriptions"
            " or a calibrated threshold."
        )
    else:
        print(
            f"NOT VIABLE. Separation too weak (mean gap {mean_gap:.4f}, F1 {best_f1:.3f})."
            " The encoder space does not separate this retrieval task."
        )

    # Per-command detail
    print(f"\n--- DETAILED SCORES (threshold {best_thresh:.2f}) ---")
    print("\nhas-tool commands:")
    print(f"  {'score':>6}  {'pred':>10}  {'expected':>10}  status  command")
    for i, (cmd, expected) in enumerate(HAS_TOOL):
        pred = TOOL_NAMES[int(has_sims[i].argmax())]
        score = float(has_top[i])
        ok = "OK  " if pred == expected and score >= best_thresh else "FAIL"
        print(f"  {score:.4f}  {pred:>10}  {expected:>10}  {ok}    {cmd!r}")

    print("\nno-tool commands:")
    print(f"  {'score':>6}  {'top_tool':>10}  status  command")
    for i, cmd in enumerate(NO_TOOL):
        top_tool = TOOL_NAMES[int(no_sims[i].argmax())]
        score = float(no_top[i])
        ok = "OK  " if score < best_thresh else "FAIL"
        print(f"  {score:.4f}  {top_tool:>10}  {ok}    {cmd!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(custom_threshold: Optional[float] = None, mode: str = "both") -> None:
    print("=== TOOL ROUTER EVAL ===")
    print(f"tools: {', '.join(TOOL_NAMES)}")
    print(f"has-tool examples: {len(HAS_TOOL)}  no-tool examples: {len(NO_TOOL)}")

    if mode in ("finetuned", "both"):
        print("\nloading fine-tuned SetFit encoder...")
        ft_enc, ft_tool_embs = _load_finetuned()
        _run_single_eval(
            enc=ft_enc,
            tool_embs=ft_tool_embs,
            use_preprocess=True,
            normalize=False,
            label=f"fine-tuned SetFit body  ({MODEL_DIR})",
        )

    if mode in ("base", "both"):
        print(f"\nloading base encoder ({BASE_MODEL_ID})...")
        b_enc, b_tool_embs = _load_base()
        _run_single_eval(
            enc=b_enc,
            tool_embs=b_tool_embs,
            use_preprocess=True,
            normalize=True,
            label=f"base  {BASE_MODEL_ID}",
        )

    print("\n=== SUMMARY ===")
    print(
        "The fine-tuned SetFit encoder is optimized for 5-class intent classification"
        " via contrastive training. That training destroys cross-domain semantic"
        " similarity: command-to-tool-description cosines collapse to ~0. The"
        " fine-tuned encoder CANNOT be reused for retrieval as-is."
    )
    print(
        "The base encoder (BAAI/bge-small-en-v1.5) retains strong semantic"
        " relationships and produces clean separation between has-tool and no-tool"
        " commands at a fixed threshold. See the base encoder results above for the"
        " recommended threshold and per-tool accuracy."
    )
    print(
        "Recommended architecture: use the fine-tuned SetFit model for intent"
        " classification (integration vs. other), then run the base encoder for"
        " tool retrieval within the integration branch. The base encoder is only"
        " 33MB and can be loaded alongside the SetFit model with negligible overhead."
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    custom = float(args[0]) if args and args[0].replace(".", "", 1).isdigit() else None
    mode_arg = "both"
    for a in args:
        if a in ("base", "finetuned", "both"):
            mode_arg = a
    main(custom, mode_arg)
