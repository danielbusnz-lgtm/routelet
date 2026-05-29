"""Approach A prototype: classify integration commands directly to tool labels.

Two variants on the same data:
  (a) FROZEN  - embed with existing models/setfit encoder, fit fresh LR head.
  (b) RETRAIN - fine-tune a fresh SetFit model with tool-level labels.

Usage:
    .venv/bin/python -m routelet.tool_classifier_proto

Saves retrained model to models/setfit_tools_proto (separate from models/setfit).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from routelet.preprocess import preprocess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
SETFIT_MODEL_DIR = PROJECT_ROOT / "models" / "setfit"
PROTO_MODEL_OUT = PROJECT_ROOT / "models" / "setfit_tools_proto"

TOOLS = ["spotify", "gmail", "github", "youtube", "no_tool"]

# ---------------------------------------------------------------------------
# Dataset: hand-label each integration.jsonl row by tool
# ---------------------------------------------------------------------------

# Rules used for labeling:
#   spotify   - music playback control (play/pause/skip/volume/what's playing)
#   gmail     - read/send/search email
#   github    - PRs / issues / repos / push / pull
#   youtube   - search or play youtube videos
#   no_tool   - everything else (no matching integration)

_RAW_LABELS: list[tuple[str, str]] = [
    # --- from integration.jsonl (102 rows) ---
    ("pause the music", "spotify"),
    ("turn off the living room lights", "no_tool"),
    ("set a timer for ten minutes", "no_tool"),
    ("what's the weather today", "no_tool"),
    ("text mom i'm on my way", "no_tool"),
    ("set an alarm for 7am", "no_tool"),
    ("add milk to my shopping list", "no_tool"),
    ("turn the volume up", "spotify"),
    ("call the dentist", "no_tool"),
    ("play the next episode on netflix", "no_tool"),
    ("mute my microphone", "no_tool"),
    ("take a screenshot", "no_tool"),
    ("send a slack message to the team channel", "no_tool"),
    ("start a 25 minute focus timer", "no_tool"),
    ("open spotify", "spotify"),
    ("add a meeting to my calendar for 3pm", "no_tool"),
    ("turn on do not disturb", "no_tool"),
    ("search google for italian restaurants nearby", "no_tool"),
    ("start recording my screen", "no_tool"),
    ("lower the brightness", "no_tool"),
    ("play some jazz", "spotify"),
    ("snooze the alarm", "no_tool"),
    ("send an email to my boss saying i'll be late", "gmail"),
    ("connect to my bluetooth headphones", "no_tool"),
    ("set the thermostat to 70 degrees", "no_tool"),
    ("share my location with sarah", "no_tool"),
    ("turn on the porch light", "no_tool"),
    ("add a reminder to call the bank tomorrow", "no_tool"),
    ("what's on my calendar today", "no_tool"),
    ("skip to the next track", "spotify"),
    ("play my workout playlist", "spotify"),
    ("turn off the tv", "no_tool"),
    ("text dad happy birthday", "no_tool"),
    ("dim the bedroom lights", "no_tool"),
    ("increase the volume to max", "spotify"),
    ("pause the podcast", "spotify"),
    ("send a whatsapp to alex", "no_tool"),
    ("turn on airplane mode", "no_tool"),
    ("play the latest taylor swift album on spotify", "spotify"),
    ("set my status to away", "no_tool"),
    ("play remember the name by fort minor", "spotify"),
    ("play the song memories by maroon 5", "spotify"),
    ("add remember me to my playlist", "spotify"),
    ("play what's my age again by blink 182", "spotify"),
    ("play the night we met", "spotify"),
    ("play unforgettable by nat king cole", "spotify"),
    ("skip to the song called my way", "spotify"),
    ("play try to remember", "spotify"),
    ("play memory from the musical cats", "spotify"),
    ("queue up remember when by alan jackson", "spotify"),
    ("play the song called my name is", "spotify"),
    ("play i will remember you by sarah mclachlan", "spotify"),
    ("play check yes juliet", "spotify"),
    ("play the chainsmokers song memories", "spotify"),
    ("play don't you forget about me", "spotify"),
    ("add my favorite song to the queue", "spotify"),
    ("book a table for two at an italian place saturday night", "no_tool"),
    ("find me a thai place nearby", "no_tool"),
    ("show me my unread emails", "gmail"),
    ("what's playing right now", "spotify"),
    ("search google for how to fix a leaky faucet", "no_tool"),
    ("show me the best sushi spots around here", "no_tool"),
    ("find a gas station close by", "no_tool"),
    ("what are the top rated burgers near me", "no_tool"),
    ("post this to twitter", "no_tool"),
    ("tweet good morning everyone", "no_tool"),
    ("send a text to jenny saying running late", "no_tool"),
    ("rewind the podcast thirty seconds", "spotify"),
    ("turn the volume down a bit", "spotify"),
    ("go back to the previous track", "spotify"),
    ("what's the weather like in chicago", "no_tool"),
    ("show me directions to the airport", "no_tool"),
    ("find me a parking garage near the stadium", "no_tool"),
    ("what restaurants are open right now near me", "no_tool"),
    ("look up the address for the nearest pharmacy", "no_tool"),
    ("play that new drake song", "spotify"),
    ("shuffle my liked songs", "spotify"),
    ("raise the volume", "spotify"),
    ("post a photo to my instagram story", "no_tool"),
    ("send an email to the landlord about the rent", "gmail"),
    ("what time does the closest target close", "no_tool"),
    ("find me a hotel in seattle for tonight", "no_tool"),
    ("google the showtimes for the new batman movie", "no_tool"),
    ("fast forward to the chorus", "spotify"),
    ("show me my recent transactions", "no_tool"),
    ("whats the traffic like on my way home", "no_tool"),
    ("find me a vet that's open on sundays", "no_tool"),
    ("play the album on repeat", "spotify"),
    ("send a dm to mark on instagram", "no_tool"),
    ("look up flights to miami", "no_tool"),
    ("set the volume to fifty percent", "spotify"),
    ("show me the menu for the pizza place down the street", "no_tool"),
    ("play something chill", "spotify"),
    ("shoot a text to ryan saying happy friday", "no_tool"),
    ("throw on my road trip playlist", "spotify"),
    ("ping the marketing channel that the deck is ready", "no_tool"),
    ("bump the brightness down", "no_tool"),
    ("fire off an email to support about my broken order", "gmail"),
    ("drop a pin at my current location and send it to dad", "no_tool"),
    ("queue up that new sza track", "spotify"),
    ("remind me to take out the trash tonight", "no_tool"),
    ("pull up directions to the nearest hospital", "no_tool"),
    # --- synthetic gmail examples (not in integration.jsonl) ---
    ("check my email", "gmail"),
    ("do i have any new emails", "gmail"),
    ("search my inbox for the receipt from amazon", "gmail"),
    ("find the email with the conference details", "gmail"),
    ("compose an email to the team about the deadline", "gmail"),
    ("reply to the last email from sarah", "gmail"),
    ("any unread messages in my gmail", "gmail"),
    ("send a quick email to hr asking about my leave balance", "gmail"),
    ("look for emails from my landlord", "gmail"),
    ("forward that email to jake", "gmail"),
    # --- synthetic github examples (not in integration.jsonl) ---
    ("show my open pull requests", "github"),
    ("list open issues on my repo", "github"),
    ("any new prs filed today", "github"),
    ("check the status of my pull request", "github"),
    ("what repos do i have on github", "github"),
    ("show unreviewed pull requests", "github"),
    ("are there any failing checks on my pr", "github"),
    ("list the issues assigned to me", "github"),
    ("what branches do i have on that repo", "github"),
    ("did anyone comment on my pr", "github"),
    # --- synthetic youtube examples (not in integration.jsonl) ---
    ("play lofi beats on youtube", "youtube"),
    ("search youtube for that lex fridman interview", "youtube"),
    ("find a rust async tutorial on youtube", "youtube"),
    ("play the latest veritasium video", "youtube"),
    ("look up that cooking video on youtube", "youtube"),
    ("search youtube for how to make sourdough bread", "youtube"),
    ("put on some ambient music on youtube", "youtube"),
    ("show me that python tutorial on youtube", "youtube"),
    ("find the mr beast challenge video", "youtube"),
    ("play that ted talk on youtube", "youtube"),
    # --- synthetic no_tool examples (25-30) ---
    ("find me restaurants nearby", "no_tool"),
    ("set a timer for 10 minutes", "no_tool"),
    ("whats the weather right now", "no_tool"),
    ("turn up the brightness", "no_tool"),
    ("order an uber", "no_tool"),
    ("search google for something", "no_tool"),
    ("pull up the rust docs", "no_tool"),
    ("what's 15% of 80", "no_tool"),
    ("open the terminal", "no_tool"),
    ("translate this to spanish", "no_tool"),
    ("lock my screen", "no_tool"),
    ("copy that to the clipboard", "no_tool"),
    ("remind me at 8pm", "no_tool"),
    ("what's the score of the game", "no_tool"),
    ("how many calories in a banana", "no_tool"),
    ("set my wallpaper to this image", "no_tool"),
    ("call mom", "no_tool"),
    ("navigate to the coffee shop", "no_tool"),
    ("book a flight to new york", "no_tool"),
    ("what time is it in london", "no_tool"),
    ("convert 100 dollars to euros", "no_tool"),
    ("schedule a meeting for tomorrow at 2pm", "no_tool"),
    ("close all the open windows", "no_tool"),
    ("take a screenshot of the screen", "no_tool"),
    ("turn off wifi", "no_tool"),
    ("show me the news headlines", "no_tool"),
    ("check the stock price of apple", "no_tool"),
    ("send a text to mom", "no_tool"),
]


def _load_dataset() -> tuple[list[str], list[str]]:
    texts = [preprocess(t) for t, _ in _RAW_LABELS]
    labels = [lbl for _, lbl in _RAW_LABELS]
    return texts, labels


def _stratified_split(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_te = next(sss.split(texts, labels))
    tr_texts = [texts[i] for i in idx_tr]
    tr_labels = [labels[i] for i in idx_tr]
    te_texts = [texts[i] for i in idx_te]
    te_labels = [labels[i] for i in idx_te]
    return tr_texts, tr_labels, te_texts, te_labels


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _print_report(
    name: str,
    te_labels: list[str],
    preds: list[str],
    proba: np.ndarray | None = None,
) -> None:
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

    # Focus: no_tool recall
    nt_true = [1 if lbl == "no_tool" else 0 for lbl in te_labels]
    nt_pred = [1 if p == "no_tool" else 0 for p in preds]
    nt_correct = sum(a == b for a, b in zip(nt_true, nt_pred) if a == 1)
    nt_total = sum(nt_true)
    nt_recall = nt_correct / nt_total if nt_total > 0 else 0.0
    print(f"\nno_tool recall: {nt_recall:.3f}  ({nt_correct}/{nt_total})")


# ---------------------------------------------------------------------------
# Variant (a): FROZEN encoder + fresh LR head
# ---------------------------------------------------------------------------


def run_frozen() -> None:
    print("\n" + "=" * 60)
    print("VARIANT (a): FROZEN encoder + fresh LR head")
    print("=" * 60)
    print(f"loading SetFit encoder from {SETFIT_MODEL_DIR} ...")

    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(SETFIT_MODEL_DIR))
    enc = model.model_body

    texts, labels = _load_dataset()
    label_counts = {t: labels.count(t) for t in TOOLS}
    print(f"dataset: {len(texts)} examples  |  " + "  ".join(f"{k}={v}" for k, v in label_counts.items()))

    tr_texts, tr_labels, te_texts, te_labels = _stratified_split(texts, labels)
    print(f"train: {len(tr_texts)}  test: {len(te_texts)}")

    print("embedding training set ...")
    tr_embs = enc.encode(tr_texts, convert_to_numpy=True, show_progress_bar=False)
    print("embedding test set ...")
    te_embs = enc.encode(te_texts, convert_to_numpy=True, show_progress_bar=False)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(tr_embs, tr_labels)

    preds = clf.predict(te_embs)
    proba = clf.predict_proba(te_embs)

    _print_report("(a) FROZEN", te_labels, list(preds), proba)


# ---------------------------------------------------------------------------
# Variant (b): RETRAIN SetFit with tool labels
# ---------------------------------------------------------------------------


def run_retrain() -> None:
    print("\n" + "=" * 60)
    print("VARIANT (b): RETRAIN SetFit with tool labels")
    print("=" * 60)

    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    BASE = "BAAI/bge-small-en-v1.5"
    texts, labels = _load_dataset()
    label_counts = {t: labels.count(t) for t in TOOLS}
    print(f"dataset: {len(texts)} examples  |  " + "  ".join(f"{k}={v}" for k, v in label_counts.items()))

    tr_texts, tr_labels, te_texts, te_labels = _stratified_split(texts, labels)
    print(f"train: {len(tr_texts)}  test: {len(te_texts)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}")

    train_ds = Dataset.from_dict({"text": tr_texts, "label": tr_labels})

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

    preds_raw = model.predict(te_texts)
    preds = [str(p) for p in preds_raw]

    _print_report("(b) RETRAIN", te_labels, preds)

    PROTO_MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(PROTO_MODEL_OUT))
    print(f"\nsaved proto model to {PROTO_MODEL_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== APPROACH A PROTOTYPE: TOOL CLASSIFICATION ===")
    print(f"tools: {TOOLS}")
    texts, labels = _load_dataset()
    print(f"total labeled examples: {len(texts)}")
    label_counts = {t: labels.count(t) for t in TOOLS}
    print("class distribution: " + "  ".join(f"{k}={v}" for k, v in label_counts.items()))

    run_frozen()
    run_retrain()

    print("\n" + "=" * 60)
    print("APPROACH A PROTOTYPE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
