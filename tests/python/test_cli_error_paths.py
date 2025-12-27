import json
import os
import subprocess
import unittest
from pathlib import Path


def _engine_bin() -> str:
    # Repo root is assumed to be the current working directory when running `make test`.
    # Keep it simple and explicit.
    return str(Path("build") / "inference-engine")


class CLIErrorPathsTests(unittest.TestCase):
    def test_invalid_flag_is_rejected(self) -> None:
        bin_path = _engine_bin()
        if not os.path.exists(bin_path):
            self.skipTest(f"engine binary not found: {bin_path}")

        cp = subprocess.run(
            [bin_path, "--this-flag-does-not-exist"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(cp.returncode, 0)
        # Message text may vary; just ensure it didn't silently succeed.
        self.assertTrue(cp.stderr.strip() or cp.stdout.strip())

    def test_zero_tokens_graceful(self) -> None:
        bin_path = _engine_bin()
        if not os.path.exists(bin_path):
            self.skipTest(f"engine binary not found: {bin_path}")

        cp = subprocess.run(
            [bin_path, "--prompt", "hi", "--max-new", "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(cp.returncode, 0)

        # Parse JSON instead of string-matching spacing.
        out = cp.stdout.strip()
        self.assertTrue(out, "expected JSON on stdout")
        obj = json.loads(out)

        self.assertIn("tokens_generated", obj)
        self.assertEqual(int(obj["tokens_generated"]), 0)


if __name__ == "__main__":
    unittest.main()
