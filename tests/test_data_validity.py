"""Data integrity tests: JSONL schema, intent values, dedup, and holdout disjointness.

No heavy deps: scikit-learn is pulled in by data.py but torch/setfit are not.

Run with: .venv/bin/python -m unittest tests.test_data_validity
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from routelet.data import Intent, load, load_dir

REPO = Path(__file__).parent.parent
DATA_DIR = REPO / "data"
HOLDOUT = REPO / "evals" / "holdout.jsonl"

VALID_INTENTS = {i.value for i in Intent}
DATA_FILES = sorted(DATA_DIR.glob("*.jsonl"))


class TestJsonlSchema(unittest.TestCase):
    """Every line in every file has exactly {"text","intent"} with a valid intent."""

    def _check_file(self, path: Path) -> None:
        with path.open() as f:
            for n, raw in enumerate(f, start=1):
                if not raw.strip():
                    continue
                with self.subTest(file=path.name, line=n):
                    row = json.loads(raw)
                    self.assertEqual(
                        set(row.keys()),
                        {"text", "intent"},
                        msg=f"{path.name}:{n} unexpected keys: {set(row.keys())}",
                    )
                    self.assertIn(
                        row["intent"],
                        VALID_INTENTS,
                        msg=f"{path.name}:{n} unknown intent {row['intent']!r}",
                    )

    def test_data_files(self) -> None:
        self.assertTrue(DATA_FILES, "no .jsonl files found in data/")
        for path in DATA_FILES:
            self._check_file(path)

    def test_holdout_file(self) -> None:
        self._check_file(HOLDOUT)


class TestTrainingPoolDedup(unittest.TestCase):
    """No two rows in the combined training pool share the same text."""

    def test_no_duplicate_texts(self) -> None:
        examples = load_dir(DATA_DIR)
        seen: dict[str, str] = {}  # text -> first filename
        duplicates: list[str] = []
        for ex in examples:
            if ex.text in seen:
                duplicates.append(ex.text)
            else:
                seen[ex.text] = ex.intent.value
        self.assertEqual(
            duplicates,
            [],
            msg=f"Duplicate texts in training pool ({len(duplicates)}):\n"
            + "\n".join(f"  {t!r}" for t in duplicates[:20]),
        )


class TestHoldoutDisjointness(unittest.TestCase):
    """Zero overlap between holdout texts and training texts (normalized)."""

    @staticmethod
    def _norm(text: str) -> str:
        return text.strip().lower()

    def test_no_train_eval_overlap(self) -> None:
        train_norms = {self._norm(e.text) for e in load_dir(DATA_DIR)}
        holdout_norms = {self._norm(e.text) for e in load(HOLDOUT)}
        overlap = train_norms & holdout_norms
        self.assertEqual(
            overlap,
            set(),
            msg=f"Holdout texts found in training pool ({len(overlap)} texts):\n"
            + "\n".join(f"  {t!r}" for t in sorted(overlap)),
        )


if __name__ == "__main__":
    unittest.main()
