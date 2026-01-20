"""tests/python/test_cli.py

CLI smoke tests for the inference engine (stdlib-only).

These tests execute the compiled binary and expect it to load a model.
To keep `make test` reliable when model artifacts are not present or are
under development, this module is skipped unless integration tests are
explicitly enabled.

Enable integration tests with:
  IE_TEST_INTEGRATION=1 make test
"""

import json
import os
import subprocess
import unittest
from pathlib import Path


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


def _require_integration(testcase: unittest.TestCase) -> None:
    if not _env_truthy("IE_TEST_INTEGRATION"):
        testcase.skipTest("integration tests disabled (set IE_TEST_INTEGRATION=1 to enable)")


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.bin_path = self.repo_root / "build" / "inference-engine"

    def test_cli_runs_and_emits_json(self) -> None:
        _require_integration(self)

        self.assertTrue(self.bin_path.exists(), msg="Binary not found. Run: make build")

        cp = subprocess.run(
            [str(self.bin_path), "test prompt"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )

        line = cp.stdout.strip()
        self.assertTrue(line.startswith("{") and line.endswith("}"), msg=f"Not JSON: {line}")
        m = json.loads(line)

        for key in (
            "tokens_generated",
            "wall_time_s",
            "tps_true",
            "latency_p50_ms",
            "latency_p95_ms",
            "rss_peak_mb",
            "kv_hits",
            "kv_misses",
        ):
            self.assertIn(key, m, msg=f"Missing field: {key}")

        self.assertIsInstance(m["tokens_generated"], int)
        self.assertGreaterEqual(m["tokens_generated"], 0)
        self.assertIsInstance(m["tps_true"], (int, float))
        self.assertGreater(m["tps_true"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
