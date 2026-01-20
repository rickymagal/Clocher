"""tests/python/test_harness_more.py

Additional harness checks.

This requires a working model, so it is treated as an integration test and
is skipped unless enabled.

Enable integration tests with:
  IE_TEST_INTEGRATION=1 make test
"""

import os
import subprocess
import unittest
from pathlib import Path

HARNESS = str(Path("benchmarks/harness.py").resolve())


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


class HarnessMoreTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _env_truthy("IE_TEST_INTEGRATION"):
            self.skipTest("integration tests disabled (set IE_TEST_INTEGRATION=1 to enable)")

    def test_custom_samples_and_promptlen(self):
        cp = subprocess.run(
            ["python3", HARNESS, "--samples", "4", "--prompt", "abcdefghij" * 3],
            text=True,
            capture_output=True,
            check=True,
            timeout=600,
        )
        self.assertIn("wrote:", cp.stdout)

        lines = [ln for ln in cp.stdout.splitlines() if "summary.json" in ln]
        self.assertTrue(lines, msg=cp.stdout + "\n" + cp.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
