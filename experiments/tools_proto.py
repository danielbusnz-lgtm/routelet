"""Approach A prototype: classify integration commands to tool labels.

Variant trained: FINETUNE only (fresh SetFit on bge-small base).
The frozen-head variant is skipped; it failed in earlier runs.

Data source: all 211 rows of data/integration.jsonl, relabeled by keyword rules.
Split: stratified 80/20 by tool label, seeded at 42.
The main evals/holdout.jsonl is NOT used here.

Usage:
    .venv/bin/python -m routelet.tools_proto
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from routelet.preprocess import preprocess

PROJECT_ROOT = Path(__file__).parent.parent.parent
INTEGRATION_DATA = PROJECT_ROOT / "data" / "integration.jsonl"
PROTO_MODEL_OUT = PROJECT_ROOT / "models" / "setfit_tools_proto"

TOOLS = ["spotify", "gmail", "github", "youtube", "no_tool"]

# ---------------------------------------------------------------------------
# Keyword-based relabeler
# Rules (applied in priority order):
#   youtube  - explicit "youtube" mention
#   github   - github/PR/pull request/issue/repo/branch/commit/CI/fork/star/clone
#   gmail    - email/inbox/compose/reply/forward/archive in email context
#   spotify  - music playback controls, play <artist/song>, spotify, podcast playback
#   no_tool  - everything else
#
# Tricky cases preserved from old comments:
#   "play the next episode on netflix" -> no_tool (no netflix integration)
#   "queue up the next episode of the office" -> no_tool (TV, not spotify)
#   "skip ahead like a minute in this podcast" -> spotify (playback control)
#   "crank the bass up on this song" -> spotify (playback control)
#   "play that song on youtube" -> youtube (explicit youtube)
# ---------------------------------------------------------------------------

import re

_YOUTUBE_KW = re.compile(r"\byoutube\b", re.I)

_GITHUB_KW = re.compile(
    r"\b(github|pull request|pull requests|open pr|open prs|my prs|my pr|merge pr|"
    r"draft pr|draft pull|approve.*pr|pr from|pr for|"
    r"pull request from|pull request for|"
    r"repo\b|repos\b|branch\b|branches\b|commit\b|commits\b|"
    r"issue\b|issues\b|fork\b|clone\b|CI\b|workflow\b|"
    r"github action|push.*change|push.*github|latest from github|"
    r"github build|github star|github notification)\b",
    re.I,
)
# "star" alone is too broad (hits "star the email"); only match "star" when paired with github context
_GITHUB_STAR = re.compile(r"\bstar\b.*\b(repo|github)\b|\b(github|repo)\b.*\bstar\b", re.I)

# Email keywords: cover all the ways people say "email/inbox" tasks
_GMAIL_KW = re.compile(
    r"\b(email\b|inbox\b|unread emails?|new emails?|read.*email|check.*email|"
    r"send.*email|compose.*email|search.*email|search.*inbox|"
    r"reply.*email|forward.*email|archive.*email|delete.*email|"
    r"mark.*email|star.*email|draft.*email|newest email|latest email|"
    r"email to|email from|do i have.*email|any.*email|"
    r"my emails|promotional email|newsletter email|archive.*email|"
    r"emails?$)\b",
    re.I,
)

# Spotify: play/skip/pause/queue/shuffle/volume controls for music/podcast
# Explicit "spotify" mention, or music-playback verbs with music targets
_SPOTIFY_EXPLICIT = re.compile(r"\bspotify\b", re.I)
_SPOTIFY_PLAY = re.compile(
    r"\b(play|queue up|throw on|put on|blast|fire up)\b", re.I
)
_SPOTIFY_MUSIC_TARGET = re.compile(
    r"\b(song|album|track|music|playlist|artist|band|jazz|lofi|"
    r"hip hop|beats|shuffle|podcast|episode of my podcast|"
    r"chill|liked songs)\b",
    re.I,
)
# Match "play/queue up <anything>" at sentence start (likely a song/artist name).
# "add ... to my playlist/queue" is also spotify if it has a playlist/queue target.
_SPOTIFY_PLAY_NAMED = re.compile(
    r"^(play|queue up)\b",
    re.I,
)
_SPOTIFY_ADD_QUEUE = re.compile(
    r"\badd\b.+\b(playlist|queue)\b",
    re.I,
)
_SPOTIFY_CONTROL = re.compile(
    r"\b(pause|skip|next track|previous track|go back to the previous|"
    r"rewind|fast forward|volume|raise the volume|lower the volume|"
    r"turn.*volume|set.*volume|increase.*volume|crank.*up|"
    r"what.s playing|shuffle my|repeat)\b",
    re.I,
)
# Hard-exclude: netflix episode, tv show episode (not podcast)
_SPOTIFY_EXCLUDE = re.compile(
    r"\b(netflix|episode of the|next episode of|tv show|hulu)\b", re.I
)


def _relabel(text: str) -> str:
    """Return tool label for a single integration.jsonl row text."""
    t = text.strip()

    # Priority 1: explicit youtube
    if _YOUTUBE_KW.search(t):
        return "youtube"

    # Priority 2: github signals (star needs explicit github context)
    if _GITHUB_KW.search(t) or _GITHUB_STAR.search(t):
        return "github"

    # Priority 3: gmail signals
    if _GMAIL_KW.search(t):
        return "gmail"

    # Priority 4: spotify
    if _SPOTIFY_EXCLUDE.search(t):
        return "no_tool"
    if _SPOTIFY_EXPLICIT.search(t):
        return "spotify"
    if _SPOTIFY_CONTROL.search(t):
        return "spotify"
    if _SPOTIFY_PLAY.search(t) and _SPOTIFY_MUSIC_TARGET.search(t):
        return "spotify"
    # "play/queue up <anything>" at start -> likely music
    if _SPOTIFY_PLAY_NAMED.match(t) and not _SPOTIFY_EXCLUDE.search(t):
        return "spotify"
    # "add <x> to my playlist/queue" -> spotify
    if _SPOTIFY_ADD_QUEUE.search(t):
        return "spotify"

    return "no_tool"


def load_and_relabel() -> list[tuple[str, str]]:
    """Load integration.jsonl and return (text, tool_label) pairs."""
    rows = []
    with open(INTEGRATION_DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"]
            label = _relabel(text)
            rows.append((text, label))
    return rows


def _counts(labels: list[str]) -> str:
    return "  ".join(f"{t}={labels.count(t)}" for t in TOOLS)


def _print_report(name: str, te_labels: list[str], preds: list[str], te_texts: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"VARIANT: {name}")
    print(f"{'=' * 60}")
    print(classification_report(te_labels, preds, labels=TOOLS, zero_division=0))

    cm = confusion_matrix(te_labels, preds, labels=TOOLS)
    col_w = max(len(t) for t in TOOLS) + 2
    header = " " * col_w + "".join(f"{t:>{col_w}}" for t in TOOLS)
    print("confusion matrix (rows=true, cols=pred):")
    print(header)
    for i, row_lbl in enumerate(TOOLS):
        row_str = f"{row_lbl:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in cm[i])
        print(row_str)

    nt_true = [lbl == "no_tool" for lbl in te_labels]
    nt_pred_as_nt = [p == "no_tool" for p in preds]
    nt_correct = sum(a and b for a, b in zip(nt_true, nt_pred_as_nt))
    nt_total = sum(nt_true)
    nt_recall = nt_correct / nt_total if nt_total else 0.0
    print(f"\nno_tool recall: {nt_recall:.3f}  ({nt_correct}/{nt_total})")

    acc = sum(p == t for p, t in zip(preds, te_labels)) / len(te_labels)
    print(f"overall accuracy: {acc:.3f}  ({sum(p == t for p, t in zip(preds, te_labels))}/{len(te_labels)})")

    errors = [
        (te_labels[i], preds[i], te_texts[i])
        for i in range(len(te_labels))
        if preds[i] != te_labels[i]
    ]
    if errors:
        print(f"\nmisclassifications ({len(errors)}):")
        for true_lbl, pred_lbl, text in errors:
            print(f"  true={true_lbl:<10} pred={pred_lbl:<10} text={text!r}")


# ---------------------------------------------------------------------------
# Variant: FINETUNE SetFit with tool labels (bge-small base)
# ---------------------------------------------------------------------------

def run_finetune(
    tr_texts: list[str],
    tr_labels: list[str],
    te_texts: list[str],
    te_labels: list[str],
) -> None:
    print("\n" + "=" * 60)
    print("VARIANT: FINETUNE SetFit (bge-small base) with tool labels")
    print("=" * 60)

    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    BASE = "BAAI/bge-small-en-v1.5"

    print(f"train: {len(tr_texts)}  |  {_counts(tr_labels)}")
    print(f"test:  {len(te_texts)}  |  {_counts(te_labels)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}")

    proc_tr = [preprocess(t) for t in tr_texts]
    proc_te = [preprocess(t) for t in te_texts]

    train_ds = Dataset.from_dict({"text": proc_tr, "label": tr_labels})

    model = SetFitModel.from_pretrained(
        BASE,
        labels=TOOLS,
        device=device,
        head_params={"class_weight": "balanced"},
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=2,
        sampling_strategy="unique",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    preds_raw = model.predict(proc_te)
    preds = [str(p) for p in preds_raw]

    _print_report("FINETUNE", te_labels, preds, te_texts)

    PROTO_MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(PROTO_MODEL_OUT))
    print(f"\nsaved proto model to {PROTO_MODEL_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== APPROACH A PROTOTYPE: TOOL CLASSIFICATION (stratified split) ===")
    print(f"tools: {TOOLS}")
    print(f"data:  {INTEGRATION_DATA}")

    # Step 1: relabel
    all_data = load_and_relabel()
    all_texts = [t for t, _ in all_data]
    all_labels = [lbl for _, lbl in all_data]

    print(f"\nRelabel distribution ({len(all_data)} total rows):")
    dist = Counter(all_labels)
    for tool in TOOLS:
        print(f"  {tool:<12} {dist[tool]:>4}")

    # Spot-check: print any rows that might be misfiring
    print("\nSpot-check: first 3 per class from relabeled set")
    seen: dict[str, int] = {t: 0 for t in TOOLS}
    for text, lbl in all_data:
        if seen[lbl] < 3:
            print(f"  {lbl:<12} {text!r}")
            seen[lbl] += 1

    # Check for thin classes
    thin = [t for t in TOOLS if dist[t] < 10]
    if thin:
        print(f"\nWARNING: thin classes (< 10 examples): {thin}")
    else:
        print("\nAll classes have >= 10 examples. OK to split.")

    # Step 2: stratified 80/20 split
    tr_texts, te_texts, tr_labels, te_labels = train_test_split(
        all_texts, all_labels,
        test_size=0.20,
        stratify=all_labels,
        random_state=42,
    )

    print(f"\nStratified split (seed=42):")
    print(f"  train {len(tr_texts)} rows: {_counts(tr_labels)}")
    print(f"  test  {len(te_texts)} rows: {_counts(te_labels)}")

    # Step 3 + 4: train and eval finetune variant
    run_finetune(tr_texts, tr_labels, te_texts, te_labels)

    print("\n" + "=" * 60)
    print("PROTOTYPE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
