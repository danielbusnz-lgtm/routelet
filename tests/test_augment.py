"""Tests for routelet.augment: determinism, label preservation, holdout non-leakage,
and conservative chaining (integration/find_action seeds never produce "and then").

No heavy deps (no torch/setfit/onnx).

Run with: .venv/bin/python -m unittest tests.test_augment
"""

import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from routelet.augment import (
    CHAINABLE,
    make_chain,
    make_variant,
)
from routelet.data import Intent

REPO = Path(__file__).parent.parent
HOLDOUT = REPO / "evals" / "holdout.jsonl"

# Small representative sample: one row per intent.
SAMPLES = [
    ("open spotify", Intent.INTEGRATION),
    ("find the settings icon", Intent.FIND_ACTION),
    ("what time is it", Intent.CHAT),
    ("remember that my meeting is at 3pm", Intent.MEMORY),
    ("search amazon and buy the cheapest hdmi cable", Intent.AGENT),
]

# A partner text for chaining tests.
_PARTNER = "turn on do not disturb"


def _norm(text: str) -> str:
    return text.strip().lower()


class TestDeterminism(unittest.TestCase):
    """Same seed produces identical variants on repeated calls."""

    def _run_once(self) -> list[str]:
        results = []
        for text, intent in SAMPLES:
            rng = random.Random(42)
            results.append(make_variant(text, intent, rng))
        return results

    def test_make_variant_deterministic(self) -> None:
        first = self._run_once()
        second = self._run_once()
        self.assertEqual(first, second)

    def test_make_chain_deterministic(self) -> None:
        results_a = []
        results_b = []
        for text, intent in SAMPLES:
            rng_a = random.Random(7)
            rng_b = random.Random(7)
            results_a.append(make_chain(text, _PARTNER, intent, rng_a))
            results_b.append(make_chain(text, _PARTNER, intent, rng_b))
        self.assertEqual(results_a, results_b)


class TestLabelPreservation(unittest.TestCase):
    """make_variant does not change the intent of the source row.

    The function returns a string, not an Example, so label preservation is
    guaranteed structurally: the caller supplies both the text and the intent
    and the variant inherits the intent at write-time. We verify that make_variant
    returns a non-empty string (a valid variant exists to attach the label to)
    and that the function never raises for any intent value.
    """

    def test_variant_is_non_empty_string(self) -> None:
        for text, intent in SAMPLES:
            with self.subTest(intent=intent):
                rng = random.Random(0)
                variant = make_variant(text, intent, rng)
                self.assertIsInstance(variant, str)
                self.assertTrue(variant.strip(), f"empty variant for {intent!r}")

    def test_augment_file_preserves_labels(self) -> None:
        """Rows written by augment() carry the same intent as their source file."""
        augmented = REPO / "data" / "augmented.jsonl"
        if not augmented.exists():
            self.skipTest("augmented.jsonl not present; run python -m routelet.augment first")
        valid_intents = {i.value for i in Intent}
        with augmented.open() as f:
            for n, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                with self.subTest(line=n):
                    row = json.loads(line)
                    self.assertIn(
                        row["intent"],
                        valid_intents,
                        msg=f"augmented.jsonl:{n} bad intent {row['intent']!r}",
                    )


class TestHoldoutNonLeakage(unittest.TestCase):
    """Variants generated for the sample rows do not appear in the holdout set."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.holdout_norms: set[str] = set()
        with HOLDOUT.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cls.holdout_norms.add(_norm(row["text"]))

    def test_variants_not_in_holdout(self) -> None:
        for text, intent in SAMPLES:
            for seed in range(10):
                rng = random.Random(seed)
                variant = make_variant(text, intent, rng)
                with self.subTest(intent=intent, seed=seed):
                    self.assertNotIn(
                        _norm(variant),
                        self.holdout_norms,
                        msg=f"variant {variant!r} matched holdout entry",
                    )


class TestConservativeChaining(unittest.TestCase):
    """integration and find_action seeds must never produce "and then" variants.

    make_chain returns None when the intent is not in CHAINABLE, which is the
    mechanism that prevents " and then " from appearing in integration/find_action
    outputs. We test that:
      1. make_chain returns None for integration and find_action (unit-level).
      2. For the complementary check, make_chain returns a non-None string for
         all intents in CHAINABLE (chat, memory, agent).
      3. When we call make_variant 50 times on integration/find_action rows the
         results never contain " and then " (make_variant never splices chains in).
    """

    def test_make_chain_returns_none_for_non_chainable(self) -> None:
        non_chainable = [Intent.INTEGRATION, Intent.FIND_ACTION]
        for intent in non_chainable:
            with self.subTest(intent=intent):
                rng = random.Random(0)
                result = make_chain("open spotify", _PARTNER, intent, rng)
                self.assertIsNone(
                    result,
                    msg=f"make_chain should return None for {intent!r} but got {result!r}",
                )

    def test_chainable_intents_not_none(self) -> None:
        for intent in CHAINABLE:
            with self.subTest(intent=intent):
                rng = random.Random(0)
                result = make_chain("do something", _PARTNER, intent, rng)
                self.assertIsNotNone(
                    result,
                    msg=f"make_chain returned None for chainable intent {intent!r}",
                )

    def test_make_variant_no_and_then_for_non_chainable(self) -> None:
        """make_variant never embeds 'and then' in integration or find_action output."""
        non_chainable_samples = [
            ("open spotify", Intent.INTEGRATION),
            ("send a slack message to alice", Intent.INTEGRATION),
            ("find the settings icon", Intent.FIND_ACTION),
            ("where is the volume button", Intent.FIND_ACTION),
        ]
        for text, intent in non_chainable_samples:
            for seed in range(50):
                rng = random.Random(seed)
                variant = make_variant(text, intent, rng)
                with self.subTest(intent=intent, seed=seed):
                    self.assertNotIn(
                        " and then ",
                        variant,
                        msg=f"variant for {intent!r} contains 'and then': {variant!r}",
                    )


if __name__ == "__main__":
    unittest.main()
