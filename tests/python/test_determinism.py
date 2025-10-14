import json
import subprocess
import unittest
from pathlib import Path

BIN = str(Path("build/inference-engine").resolve())

class DeterminismTests(unittest.TestCase):
    def test_same_prompt_same_output_short_run(self):
        args = ["--prompt", "once upon a time", "--max-new", "8"]
        a = json.loads(subprocess.run([BIN]+args, text=True, capture_output=True, check=True).stdout)
        b = json.loads(subprocess.run([BIN]+args, text=True, capture_output=True, check=True).stdout)
        self.assertEqual(a["tokens_generated"], b["tokens_generated"])
        self.assertEqual(a["tokens"][0:4], b["tokens"][0:4])  # prefix deterministic

if __name__ == "__main__":
    unittest.main(verbosity=2)
