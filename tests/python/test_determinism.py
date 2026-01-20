"""tests/python/test_determinism.py

Determinism checks for short runs.

This requires loading a real model, so it is treated as an integration
test and is skipped unless enabled.

Enable integration tests with:
  IE_TEST_INTEGRATION=1 make test
"""

import json
import os
import subprocess
import unittest
from pathlib import Path

BIN = str(Path("build/inference-engine").resolve())


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


class DeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _env_truthy("IE_TEST_INTEGRATION"):
            self.skipTest("integration tests disabled (set IE_TEST_INTEGRATION=1 to enable)")

    def test_same_prompt_same_output_short_run(self):
        args = ["--prompt", "once upon a time", "--max-new", "8"]

        a = json.loads(
            subprocess.run([BIN] + args, text=True, capture_output=True, check=True, timeout=240).stdout
        )
        b = json.loads(
            subprocess.run([BIN] + args, text=True, capture_output=True, check=True, timeout=240).stdout
        )

        self.assertEqual(a["tokens_generated"], b["tokens_generated"])

        # Prefix determinism: identical first few generated tokens.
        self.assertEqual(a.get("tokens", [])[0:4], b.get("tokens", [])[0:4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
