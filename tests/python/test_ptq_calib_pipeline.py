# -*- coding: utf-8 -*-
"""
Unit tests for the INT8 PTQ calibration pipeline (benchmarks/ptq_calib.py).

Covers:
- End-to-end run on a small synthetic matrix
- Output artifacts existence (int8/scales/report)
- Report metrics sanity (cosine >= threshold; small MSE)
- Per-tensor and per-row modes
"""
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

THIS = Path(__file__).resolve().parent.parent.parent
PTQ = THIS / "benchmarks" / "ptq_calib.py"


def write_f32_bin(path: Path, rows: int, cols: int):
    """Write a simple 2D ramp into a float32 .bin file (row-major)."""
    n = rows * cols
    with open(path, "wb") as f:
        for i in range(n):
            # spread across a moderate range to avoid all-zero or saturations
            val = (i - n / 2) / 123.0
            f.write(struct.pack("<f", float(val)))


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class PTQCalibPipelineTests(unittest.TestCase):
    def run_ptq(self, rows: int, cols: int, mode: str, out_prefix: Path):
        w = out_prefix.with_suffix(".f32.bin")
        w.parent.mkdir(parents=True, exist_ok=True)
        write_f32_bin(w, rows, cols)

        cp = subprocess.run(
            ["python3", str(PTQ),
             "--weights", str(w),
             "--rows", str(rows),
             "--cols", str(cols),
             "--mode", mode,
             "--out-prefix", str(out_prefix),
             "--accuracy-threshold", "0.99"],
            text=True, capture_output=True
        )
        self.assertEqual(cp.returncode, 0, msg=f"stderr:\n{cp.stderr}\nstdout:\n{cp.stdout}")

        int8_bin = Path(str(out_prefix) + ".int8.bin")
        scales_bin = Path(str(out_prefix) + ".scales.bin")
        report_json = Path(str(out_prefix) + ".report.json")

        for p in (int8_bin, scales_bin, report_json):
            self.assertTrue(p.exists(), msg=f"missing artifact: {p}")

        rpt = read_json(report_json)
        self.assertIn("cosine_sim", rpt)
        self.assertIn("mse", rpt)
        self.assertGreaterEqual(rpt["cosine_sim"], 0.99)
        self.assertLess(rpt["mse"], 1e-2)

    def test_per_row_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            outp = Path(td) / "perrow"
            self.run_ptq(rows=16, cols=32, mode="per_row", out_prefix=outp)

    def test_per_tensor_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            outp = Path(td) / "pertensor"
            self.run_ptq(rows=16, cols=32, mode="per_tensor", out_prefix=outp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
