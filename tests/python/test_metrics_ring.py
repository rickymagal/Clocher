import json
import subprocess
import unittest
from pathlib import Path

BIN = str(Path("build/inference-engine").resolve())

class MetricsRingTests(unittest.TestCase):
    def test_p50_p95_present_and_consistent(self):
        cp = subprocess.run([BIN, "--prompt", "abc def ghi", "--max-new", "16"],
                            text=True, capture_output=True, check=True)
        data = json.loads(cp.stdout)
        self.assertIn("latency_p50_ms", data)
        self.assertIn("latency_p95_ms", data)
        self.assertGreaterEqual(data["latency_p95_ms"], data["latency_p50_ms"])
        # p50 > 0 implies TPS computed
        if data["latency_p50_ms"] > 0:
            self.assertGreater(data["tps_true"], 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
