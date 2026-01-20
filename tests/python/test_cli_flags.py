"""tests/python/test_cli_flags.py

Feature-flag acceptance tests for the CLI (stdlib-only).

These tests require a working model load and therefore are treated as
integration tests. They are skipped unless enabled.

Enable integration tests with:
  IE_TEST_INTEGRATION=1 make test
"""

import os
import subprocess
import unittest
from pathlib import Path

BIN = str(Path("build/inference-engine").resolve())


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


class CLIFeatureFlagsTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _env_truthy("IE_TEST_INTEGRATION"):
            self.skipTest("integration tests disabled (set IE_TEST_INTEGRATION=1 to enable)")

    def run_cli(self, args):
        cp = subprocess.run([BIN] + args, text=True, capture_output=True, check=True, timeout=180)
        return cp.stdout

    def test_threads_flag_accepts_values(self):
        out = self.run_cli(["--prompt", "hello", "--max-new", "4", "--threads", "1"])
        self.assertIn('"tokens_generated"', out)
        out2 = self.run_cli(["--prompt", "hello", "--max-new", "4", "--threads", "2"])
        self.assertIn('"tokens_generated"', out2)

    def test_precision_flag(self):
        out = self.run_cli(["--prompt", "hello", "--max-new", "2", "--precision", "fp32"])
        self.assertIn('"tokens_generated"', out)
        out = self.run_cli(["--prompt", "hello", "--max-new", "2", "--precision", "bf16"])
        self.assertIn('"tokens_generated"', out)

    def test_affinity_flag_no_crash(self):
        # Accepted values, even if ignored by default.
        for mode in ["auto", "compact", "scatter"]:
            out = self.run_cli(["--prompt", "x", "--max-new", "1", "--affinity", mode])
            self.assertIn('"tokens_generated"', out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
