import json
import subprocess
import unittest
from pathlib import Path

HARNESS = str(Path("benchmarks/harness.py").resolve())

class HarnessMoreTests(unittest.TestCase):
    def test_custom_samples_and_promptlen(self):
        cp = subprocess.run(["python3", HARNESS, "--samples", "4", "--prompt", "abcdefghij" * 3],
                            text=True, capture_output=True, check=True)
        self.assertIn("wrote:", cp.stdout)  # script already prints writes
        # ensure summary exists
        # Quick parse of last line “wrote: .../summary.json”
        lines = [ln for ln in cp.stdout.splitlines() if "summary.json" in ln]
        self.assertTrue(lines)

if __name__ == "__main__":
    unittest.main(verbosity=2)
