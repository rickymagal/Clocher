"""tests/python/test_cli_error_paths.py

Error-path tests for the CLI (stdlib-only).

Some behaviors are model-independent (e.g. rejecting unknown flags).
Other behaviors require a working model load. Those are treated as
integration tests and are skipped unless enabled.

Enable integration tests with:
  IE_TEST_INTEGRATION=1 make test
"""

import json
import os
import subprocess
import unittest
from pathlib import Path


def _engine_bin() -> str:
    # Repo root is assumed to be the current working directory when running `make test`.
    return str(Path("build") / "inference-engine")


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


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
            timeout=60,
        )
        self.assertNotEqual(cp.returncode, 0)
        self.assertTrue(cp.stderr.strip() or cp.stdout.strip())

    def test_zero_tokens_graceful(self) -> None:
        if not _env_truthy("IE_TEST_INTEGRATION"):
            self.skipTest("integration tests disabled (set IE_TEST_INTEGRATION=1 to enable)")

        bin_path = _engine_bin()
        if not os.path.exists(bin_path):
            self.skipTest(f"engine binary not found: {bin_path}")

        cp = subprocess.run(
            [bin_path, "--prompt", "hi", "--max-new", "0"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(cp.returncode, 0, msg=cp.stderr.strip() or cp.stdout.strip())

        out = cp.stdout.strip()
        self.assertTrue(out, "expected JSON on stdout")
        obj = json.loads(out)

        self.assertIn("tokens_generated", obj)
        self.assertEqual(int(obj["tokens_generated"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
