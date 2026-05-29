"""Conformance tests for preprocess() against the shared fixture.

The same vectors live in the aegis repo; the Rust implementation must also
satisfy them. Do not edit the fixture; fix preprocess() if a vector fails.

Run with: .venv/bin/python -m unittest tests.test_preprocess
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from routelet.preprocess import preprocess

FIXTURE = Path(__file__).parent / "preprocess_vectors.json"


class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vectors = json.loads(FIXTURE.read_text())["vectors"]

    def test_all_vectors(self) -> None:
        for v in self.vectors:
            with self.subTest(input=v["in"]):
                self.assertEqual(preprocess(v["in"]), v["out"])


if __name__ == "__main__":
    unittest.main()
