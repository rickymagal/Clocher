"""tests/python/test_metrics_ring.py

Metrics ring / latency quantile checks.

This requires running the engine with a model loaded, so it is treated as an
integration test and is skipped unless enabled.

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


class MetricsRingTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _env_truthy("IE_TEST_INTEGRATION"):
            self.skipTest("integration tests disabled (set IE_TEST_INTEGRATION=1 to enable)")

    def test_p50_p95_present_and_consistent(self):
        cp = subprocess.run(
            [BIN, "--prompt", "abc def ghi", "--max-new", "16"],
            text=True,
            capture_output=True,
            check=True,
            timeout=240,
        )
        data = json.loads(cp.stdout)
        self.assertIn("latency_p50_ms", data)
        self.assertIn("latency_p95_ms", data)
        self.assertGreaterEqual(data["latency_p95_ms"], data["latency_p50_ms"])

        if data["latency_p50_ms"] > 0:
            self.assertGreater(data["tps_true"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
