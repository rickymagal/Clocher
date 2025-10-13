"""
CLI smoke tests for the inference engine (stdlib-only).

This test executes the compiled binary with a small prompt and asserts that:
- The process exits successfully.
- The output is a single JSON object with expected numeric fields.
"""

import json
import subprocess
import unittest
from pathlib import Path


class CLITests(unittest.TestCase):
    def setUp(self):
        # Binary built by `make build`
        self.repo_root = Path(__file__).resolve().parents[2]
        self.bin_path = self.repo_root / "build" / "inference-engine"

    def test_cli_runs_and_emits_json(self):
        self.assertTrue(self.bin_path.exists(), msg="Binary not found. Run: make build")
        cp = subprocess.run(
            [str(self.bin_path), "test prompt"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        line = cp.stdout.strip()
        self.assertTrue(line.startswith("{") and line.endswith("}"), msg=f"Not JSON: {line}")
        m = json.loads(line)

        # Minimal schema checks
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
