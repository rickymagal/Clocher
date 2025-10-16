#!/usr/bin/env python3
"""
INT8 weight-only PTQ calibration (stdlib-only).

This tool reads FP32 weights in row-major order, computes scaling factors
(min-max rule, per-tensor or per-row), quantizes to INT8, reconstructs FP32,
and writes artifacts + a JSON report with error metrics.

Artifacts:
  - <out_prefix>.int8.bin     (INT8 weights)
  - <out_prefix>.scales.bin   (float32 scales)
  - <out_prefix>.report.json  (summary with MSE, cosine similarity, pass/fail)
"""

import argparse
import json
import math
import os
from array import array
from typing import List


def read_f32(path: str, rows: int, cols: int) -> array:
    """
    Read a binary file containing float32 values and return an 'array("f")'.

    Args:
        path: Path to the .bin file.
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        An array("f") of length rows*cols in row-major order.

    Raises:
        ValueError: If the element count does not match rows*cols.
    """
    count = rows * cols
    with open(path, "rb") as f:
        data = f.read()
    floats = array("f")
    floats.frombytes(data)
    if len(floats) != count:
        raise ValueError(f"Expected {count} floats, got {len(floats)}")
    return floats


def compute_scales_minmax(w: array, rows: int, cols: int, mode: str) -> array:
    """
    Compute PTQ scales using a symmetric min-max rule.

    Args:
        w: FP32 weights (array('f')) in row-major.
        rows: Number of rows.
        cols: Number of columns.
        mode: 'per_tensor' or 'per_row'.

    Returns:
        array('f') with 1 scale (per_tensor) or 'rows' scales (per_row).
    """
    if mode == "per_tensor":
        lo = min(w)
        hi = max(w)
        amax = max(abs(lo), abs(hi))
        s = amax / 127.0 if amax > 0.0 else 1.0
        return array("f", [s])
    elif mode == "per_row":
        out = array("f", [0.0] * rows)
        for r in range(rows):
            row = w[r * cols : (r + 1) * cols]
            lo = min(row)
            hi = max(row)
            amax = max(abs(lo), abs(hi))
            s = amax / 127.0 if amax > 0.0 else 1.0
            out[r] = s
        return out
    else:
        raise ValueError("mode must be per_tensor or per_row")


def quantize_int8(w: array, rows: int, cols: int, mode: str, scales: array) -> array:
    """
    Quantize FP32 weights to INT8 using provided scales.

    Args:
        w: FP32 weights in row-major.
        rows: Number of rows.
        cols: Number of columns.
        mode: 'per_tensor' or 'per_row'.
        scales: Scale buffer from compute_scales_minmax.

    Returns:
        array('b') with quantized INT8 values in [-127, 127].
    """
    q = array("b", [0] * (rows * cols))
    if mode == "per_tensor":
        inv = 1.0 / scales[0] if scales[0] > 0.0 else 1.0
        for i, v in enumerate(w):
            x = round(v * inv)
            if x > 127:
                x = 127
            elif x < -127:
                x = -127
            q[i] = int(x)
        return q
    # per_row
    for r in range(rows):
        s = scales[r]
        inv = 1.0 / s if s > 0.0 else 1.0
        base = r * cols
        for c in range(cols):
            v = w[base + c]
            x = round(v * inv)
            if x > 127:
                x = 127
            elif x < -127:
                x = -127
            q[base + c] = int(x)
    return q


def dequant_int8(q: array, rows: int, cols: int, mode: str, scales: array) -> array:
    """
    Dequantize INT8 weights to FP32 using provided scales.

    Args:
        q: INT8 weights (array('b')) in row-major.
        rows: Number of rows.
        cols: Number of columns.
        mode: 'per_tensor' or 'per_row'.
        scales: Scales used during quantization.

    Returns:
        array('f') reconstructed in FP32.
    """
    out = array("f", [0.0] * (rows * cols))
    if mode == "per_tensor":
        s = scales[0] if scales[0] > 0.0 else 1.0
        for i, v in enumerate(q):
            out[i] = float(v) * s
        return out
    # per_row
    for r in range(rows):
        s = scales[r] if scales[r] > 0.0 else 1.0
        base = r * cols
        for c in range(cols):
            out[base + c] = float(q[base + c]) * s
    return out


def mse(a: array, b: array) -> float:
    """
    Compute mean squared error between two float arrays.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Mean squared error value.
    """
    s = 0.0
    n = len(a)
    for i in range(n):
        d = float(a[i]) - float(b[i])
        s += d * d
    return s / float(n) if n else 0.0


def cosine_sim(a: array, b: array) -> float:
    """
    Compute cosine similarity between two float arrays.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [0, 1] (1 when vectors are identical direction).
    """
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(len(a)):
        x = float(a[i])
        y = float(b[i])
        num += x * y
        da += x * x
        db += y * y
    denom = math.sqrt(da) * math.sqrt(db)
    return (num / denom) if denom > 0.0 else 1.0


def main():
    """
    Entry point: perform calibration and write artifacts + report.
    """
    ap = argparse.ArgumentParser(description="INT8 PTQ calibrator (stdlib-only)")
    ap.add_argument("--weights", required=True, help="Path to FP32 weights .bin (row-major)")
    ap.add_argument("--rows", type=int, required=True, help="Rows (R)")
    ap.add_argument("--cols", type=int, required=True, help="Cols (C)")
    ap.add_argument("--mode", choices=["per_tensor", "per_row"], default="per_row")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for artifacts")
    ap.add_argument("--accuracy-threshold", type=float, default=0.995,
                    help="Minimum cosine similarity vs FP32")
    args = ap.parse_args()

    w = read_f32(args.weights, args.rows, args.cols)
    scales = compute_scales_minmax(w, args.rows, args.cols, args.mode)
    q8 = quantize_int8(w, args.rows, args.cols, args.mode, scales)
    w_rec = dequant_int8(q8, args.rows, args.cols, args.mode, scales)

    rep = {
        "rows": args.rows,
        "cols": args.cols,
        "mode": args.mode,
        "mse": mse(w, w_rec),
        "cosine_sim": cosine_sim(w, w_rec),
        "threshold": args.accuracy_threshold,
        "passed": cosine_sim(w, w_rec) >= args.accuracy_threshold,
    }

    with open(args.out_prefix + ".int8.bin", "wb") as f:
        f.write(q8.tobytes())
    with open(args.out_prefix + ".scales.bin", "wb") as f:
        f.write(scales.tobytes())
    with open(args.out_prefix + ".report.json", "w") as f:
        json.dump(rep, f, indent=2)

    print(json.dumps(rep))


if __name__ == "__main__":
    main()
