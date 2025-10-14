import json
import os
import subprocess
import unittest
from pathlib import Path

BIN = str(Path("build/inference-engine").resolve())

class CLIFeatureFlagsTests(unittest.TestCase):
    def run_cli(self, args):
        cp = subprocess.run([BIN] + args, text=True, capture_output=True, check=True)
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
        # accepted values, even se ignorado por default
        for mode in ["auto", "compact", "scatter"]:
            out = self.run_cli(["--prompt", "x", "--max-new", "1", "--affinity", mode])
            self.assertIn('"tokens_generated"', out)

if __name__ == "__main__":
    unittest.main(verbosity=2)
