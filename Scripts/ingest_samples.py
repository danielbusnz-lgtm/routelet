"""Turn pulled distillation samples into labeled training data.

Reads the raw samples produced by ``pull_samples.py`` and resolves a single
training label for each:

  * If the sample carries ``claude_label`` (the Claude fallback fired on-device
    that turn), use it. Free teacher signal, already paid for.
  * Otherwise call the offline Claude teacher to label it (docs/distillation.md).

Then it normalizes the text through ``preprocess`` (the same pass training and
inference use, so there is no skew), drops anything already in the training pool
or in the eval set (keeping the eval honest), and appends the survivors to
``data/collected.jsonl`` as ``{"text", "intent"}`` lines that ``data.load`` reads.

Needs ANTHROPIC_API_KEY only when some samples lack a ``claude_label``. The
teacher model is claude-opus-4-8 by default; override with ROUTELET_TEACHER_MODEL.

Usage:
    uv run python Scripts/ingest_samples.py
    uv run python Scripts/ingest_samples.py --dry-run
    uv run python Scripts/ingest_samples.py --raw data/raw/today.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from routelet.data import Intent, load, load_dir  # noqa: E402
from routelet.preprocess import preprocess  # noqa: E402

DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "samples.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "collected.jsonl"
TRAIN_DIR = PROJECT_ROOT / "data"
EVAL_FILE = PROJECT_ROOT / "evals" / "holdout.jsonl"


def _norm(text: str) -> str:
    """The dedup key: normalized text, case-folded and stripped. Mirrors the
    leakage check in checkdata.py so 'same command' means the same thing here."""
    return preprocess(text).lower().strip()


def load_raw(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  skip {path}:{n}: bad json", file=sys.stderr)
    return rows


def existing_keys() -> tuple[set[str], set[str]]:
    """Texts already in the training pool and in the eval set, both normalized.
    A sample matching either is dropped: the first is a duplicate, the second
    would leak the eval into training."""
    train = {_norm(e.text) for e in load_dir(TRAIN_DIR)}
    evalset = {_norm(e.text) for e in load(EVAL_FILE)} if EVAL_FILE.exists() else set()
    return train, evalset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="raw samples JSONL")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="labeled output JSONL")
    parser.add_argument("--limit", type=int, help="stop after this many raw samples")
    parser.add_argument("--model", help="teacher model override (else ROUTELET_TEACHER_MODEL)")
    parser.add_argument("--dry-run", action="store_true", help="don't call the teacher or write")
    args = parser.parse_args()

    if not args.raw.exists():
        sys.exit(f"no raw samples at {args.raw}; run pull_samples.py first")

    raw = load_raw(args.raw)
    if args.limit:
        raw = raw[: args.limit]

    train_keys, eval_keys = existing_keys()
    # Dedup against the existing pool, the output file (so re-runs are idempotent
    # even when --out lives outside data/), and within this batch.
    seen = set(train_keys)
    if args.out.exists():
        seen |= {_norm(r["text"]) for r in load_raw(args.out) if r.get("text")}

    client = None  # created lazily; only needed for teacher labeling
    labeled: list[dict] = []
    stats = {"client_label": 0, "teacher_label": 0, "dup": 0, "leak": 0, "empty": 0, "bad": 0}

    for row in raw:
        text = (row.get("text") or "").strip()
        if not text:
            stats["empty"] += 1
            continue

        key = _norm(text)
        if key in eval_keys:
            stats["leak"] += 1
            continue
        if key in seen:
            stats["dup"] += 1
            continue

        # Prefer the free in-turn Claude label; fall back to the offline teacher.
        label_str = row.get("claude_label")
        if label_str:
            try:
                intent = Intent(label_str)
            except ValueError:
                stats["bad"] += 1
                continue
            stats["client_label"] += 1
        elif args.dry_run:
            # Count what a real run would send to the teacher without spending.
            stats["teacher_label"] += 1
            seen.add(key)
            continue
        else:
            if client is None:
                import anthropic

                client = anthropic.Anthropic()
            from routelet.teacher import classify

            intent = classify(client, preprocess(text), model=args.model)
            stats["teacher_label"] += 1

        seen.add(key)
        labeled.append({"text": preprocess(text), "intent": intent.value})

    if args.dry_run:
        would = stats["client_label"] + stats["teacher_label"]
        print(f"dry run: {would} new samples ({stats['teacher_label']} need the teacher)")
        print(f"  skipped: {stats['dup']} dup, {stats['leak']} eval-leak, "
              f"{stats['empty']} empty, {stats['bad']} bad-label")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for rec in labeled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"appended {len(labeled)} samples to {args.out}")
    print(f"  {stats['client_label']} client-labeled, {stats['teacher_label']} teacher-labeled")
    print(f"  skipped: {stats['dup']} dup, {stats['leak']} eval-leak, "
          f"{stats['empty']} empty, {stats['bad']} bad-label")


if __name__ == "__main__":
    main()
