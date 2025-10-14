import subprocess
import unittest
from pathlib import Path

BIN = str(Path("build/inference-engine").resolve())

class CLIErrorPathsTests(unittest.TestCase):
    def test_zero_tokens_graceful(self):
        cp = subprocess.run([BIN, "--prompt", "x", "--max-new", "0"], text=True, capture_output=True)
        self.assertEqual(cp.returncode, 0)
        self.assertIn('"tokens_generated": 0', cp.stdout)

    def test_invalid_flag_is_rejected(self):
        cp = subprocess.run([BIN, "--this-flag-does-not-exist"], text=True, capture_output=True)
        self.assertNotEqual(cp.returncode, 0)
        self.assertIn("usage", cp.stderr.lower())

if __name__ == "__main__":
    unittest.main(verbosity=2)
