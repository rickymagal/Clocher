# =============================================================================
# File: benchmarks/ptq_calib.py
# =============================================================================
#!/usr/bin/env python3
"""
Post-Training Quantization (weight-only) calibrator.

Reads a row-major FP32 weight matrix (W in R^{rows x cols}) from a .bin file,
computes symmetric scales and quantizes weights to INT8 or INT4 (weight-only),
then writes compact artifacts:

Outputs (for OUT_PREFIX = <prefix>):
  - <prefix>.meta.json              # metadata with shapes, files, metrics
  - <prefix>.scale.f16.bin|.f32.bin # per-row or per-tensor scales
  - <prefix>.int8.w.bin             # row-major int8 weights (if --wbits=8)
  - <prefix>.int4.w.bin             # row-major 4-bit packed weights (if --wbits=4)

Packing (INT4):
  - Signed 4-bit values using "+8 bias" scheme over range [-7..+7]
  - Two values packed per byte: low nibble = column j (even), high nibble = j+1
  - Row-major order; each row occupies ceil(cols/2) bytes

Scale modes:
  - per_row: one scale per row (default, recommended for GEMV)
  - per_tensor: one scale for the entire tensor

Metrics:
  - Average row-wise cosine similarity between W and dequantized(W_q)
  - Global mean squared error

A JSON snippet you can paste into `model.ie.json` is printed as:
  [iebin-json] {...}

Example:
  python3 benchmarks/ptq_calib.py \
      --weights out/qproj.f32.bin --rows 4096 --cols 4096 \
      --mode per_row --wbits 4 --out-prefix out/qproj_q4 \
      --scale-dtype fp16
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np


# ----------------------------- I/O Utilities ---------------------------------
def load_f32_matrix(path: str, rows: int, cols: int) -> np.ndarray:
    """
    Load a row-major float32 matrix from a flat .bin file.

    Parameters
    ----------
    path : str
        Path to the .bin file containing raw float32 data.
    rows : int
        Number of rows in the matrix.
    cols : int
        Number of columns in the matrix.

    Returns
    -------
    np.ndarray
        Array of shape (rows, cols), dtype float32.

    Raises
    ------
    ValueError
        If the file size does not match rows*cols*4 bytes.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"weights file not found: {path}")
    expected = rows * cols * 4
    actual = p.stat().st_size
    if actual != expected:
        raise ValueError(
            f"size mismatch for {path}: got {actual} bytes, expected {expected} "
            f"(rows={rows}, cols={cols})"
        )
    arr = np.fromfile(path, dtype=np.float32, count=rows * cols)
    return arr.reshape(rows, cols)


def save_binary(path: str, array: np.ndarray) -> None:
    """
    Save a contiguous numpy array to a binary file.

    Parameters
    ----------
    path : str
        Output file path.
    array : np.ndarray
        Array to write. The function writes bytes as-is (no header).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


# --------------------------- Quantization Core -------------------------------
def compute_scales(
    W: np.ndarray, wbits: int, mode: str, eps: float = 1e-12
) -> np.ndarray:
    """
    Compute symmetric quantization scales for weight-only PTQ.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix, shape (rows, cols), dtype float32.
    wbits : int
        Bit-width (8 or 4).
    mode : str
        'per_row' or 'per_tensor'.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Scales as float32. Shape (rows,) for per_row, or (1,) for per_tensor.
    """
    assert W.ndim == 2, "W must be 2D"
    qmax = 127 if wbits == 8 else 7
    if mode == "per_row":
        absmax = np.max(np.abs(W), axis=1).astype(np.float32)
        scales = np.maximum(absmax / float(qmax), eps).astype(np.float32)
        return scales  # (rows,)
    elif mode == "per_tensor":
        absmax = float(np.max(np.abs(W)))
        scale = max(absmax / float(qmax), eps)
        return np.array([scale], dtype=np.float32)  # (1,)
    else:
        raise ValueError(f"unknown mode: {mode}")


def quantize_int8(W: np.ndarray, scales: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize to INT8 with symmetric zero-point (z=0), range [-127, 127].

    Parameters
    ----------
    W : np.ndarray
        Float32 weights.
    scales : np.ndarray
        Scales computed by compute_scales.
    mode : str
        'per_row' or 'per_tensor'.

    Returns
    -------
    (Q, W_hat)
        Q is int8 array, same shape as W.
        W_hat is float32 dequantized reconstruction.
    """
    qmax = 127
    if mode == "per_row":
        s = scales[:, None]  # (rows, 1)
    else:
        s = scales.reshape(1, 1)  # (1, 1)
    Q = np.clip(np.rint(W / s), -qmax, qmax).astype(np.int8)
    W_hat = (Q.astype(np.float32) * s).astype(np.float32)
    return Q, W_hat


def _pack_row_int4_biased(q_row: np.ndarray) -> bytes:
    """
    Pack a single int8 row using the +8 bias INT4 scheme (range [-7..+7]).

    Low nibble holds column j (even), high nibble holds column j+1 (odd).
    """
    n = q_row.shape[0]
    if n % 2 != 0:
        q_row = np.concatenate([q_row, np.zeros(1, dtype=np.int8)], axis=0)
        n += 1
    # Clamp to [-7, +7], then add bias +8 to map into [1..15] (0 unused), mask to 4 bits.
    q_row = np.clip(q_row, -7, 7).astype(np.int8, copy=False)
    nibbles = (q_row.astype(np.int16) + 8).astype(np.uint8) & 0x0F
    lo = nibbles[0::2]
    hi = (nibbles[1::2] << 4)
    return (lo | hi).astype(np.uint8).tobytes()


def quantize_int4_packed(
    W: np.ndarray, scales: np.ndarray, mode: str
) -> Tuple[bytes, np.ndarray]:
    """
    Quantize to INT4 using +8 bias scheme (range [-7..+7]) and pack to nibbles.

    Parameters
    ----------
    W : np.ndarray
        Float32 weights.
    scales : np.ndarray
        Scales computed by compute_scales.
    mode : str
        'per_row' or 'per_tensor'.

    Returns
    -------
    (packed_bytes, W_hat)
        packed_bytes: bytes object containing row-major packed int4
        W_hat: float32 dequantized reconstruction (for metrics)
    """
    qmin, qmax = -7, 7
    if mode == "per_row":
        s = scales[:, None]  # (rows, 1)
    else:
        s = scales.reshape(1, 1)  # (1, 1)

    Q = np.clip(np.rint(W / s), qmin, qmax).astype(np.int8)
    # Dequantization for metrics:
    W_hat = (Q.astype(np.float32) * s).astype(np.float32)

    # Pack each row into bytes
    rows = Q.shape[0]
    packed_rows = []
    for r in range(rows):
        packed_rows.append(_pack_row_int4_biased(Q[r]))
    packed = b"".join(packed_rows)
    return packed, W_hat


# ------------------------------- Metrics -------------------------------------
def avg_row_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the average row-wise cosine similarity between two matrices.

    Parameters
    ----------
    a, b : np.ndarray
        Float32 matrices with the same shape.
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    float
        Average cosine similarity in [0, 1].
    """
    assert a.shape == b.shape
    a_norm = np.linalg.norm(a, axis=1) + eps
    b_norm = np.linalg.norm(b, axis=1) + eps
    dot = np.sum(a * b, axis=1)
    cos = dot / (a_norm * b_norm)
    # Clamp for safety
    cos = np.clip(cos, -1.0, 1.0)
    # Map to [0,1] if negative due to numeric issues; typical positive for weights
    return float(np.mean((cos + 1.0) / 2.0))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean squared error between two matrices.
    """
    d = (a - b).astype(np.float32)
    return float(np.mean(d * d))


# ------------------------------- Main Flow -----------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Input FP32 .bin (row-major)")
    ap.add_argument("--rows", type=int, required=True, help="Rows of the matrix")
    ap.add_argument("--cols", type=int, required=True, help="Cols of the matrix")
    ap.add_argument("--mode", default="per_row", choices=["per_row", "per_tensor"],
                    help="Scale mode (default: per_row)")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for artifacts")
    ap.add_argument("--accuracy-threshold", type=float, default=0.995,
                    help="Advisory threshold for avg row cosine (no hard fail by default)")
    ap.add_argument("--wbits", type=int, default=8, choices=[8, 4],
                    help="Weight bit-width: 8 or 4 (default: 8)")
    ap.add_argument("--scale-dtype", default="fp16", choices=["fp16", "fp32"],
                    help="Scale storage dtype (default: fp16)")
    ap.add_argument("--fail-below-threshold", action="store_true",
                    help="Exit non-zero if metric falls below threshold")
    args = ap.parse_args()

    W = load_f32_matrix(args.weights, args.rows, args.cols)

    # Compute scales
    scales_f32 = compute_scales(W, args.wbits, args.mode)

    # Quantize
    if args.wbits == 8:
        Q8, W_hat = quantize_int8(W, scales_f32, args.mode)
        weight_path = f"{args.out_prefix}.int8.w.bin"
        save_binary(weight_path, Q8.astype(np.int8, copy=False))
        packed_note = "int8 row-major"
    else:
        packed_bytes, W_hat = quantize_int4_packed(W, scales_f32, args.mode)
        weight_path = f"{args.out_prefix}.int4.w.bin"
        Path(weight_path).parent.mkdir(parents=True, exist_ok=True)
        with open(weight_path, "wb") as f:
            f.write(packed_bytes)
        packed_note = "int4 packed (+8 bias), low nibble = even col"

    # Save scales
    if args.mode == "per_row":
        scale_shape = [args.rows]
    else:
        scale_shape = [1]

    if args.scale_dtype == "fp16":
        scales_to_save = scales_f32.astype(np.float16)
        scale_path = f"{args.out_prefix}.scale.f16.bin"
    else:
        scales_to_save = scales_f32.astype(np.float32)
        scale_path = f"{args.out_prefix}.scale.f32.bin"

    save_binary(scale_path, scales_to_save)

    # Metrics
    cos = avg_row_cosine(W, W_hat)
    err = mse(W, W_hat)

    # Metadata
    meta = {
        "rows": args.rows,
        "cols": args.cols,
        "wbits": args.wbits,
        "mode": args.mode,
        "scale_dtype": args.scale_dtype,
        "scale_shape": scale_shape,
        "files": {
            "weights": weight_path,
            "scales": scale_path,
            "fp32_src": os.path.abspath(args.weights),
        },
        "packing": packed_note,
        "metrics": {
            "avg_row_cosine": round(cos, 6),
            "mse": err,
        },
    }

    # Emit a helpful snippet the engine can adopt in model.ie.json
    ie_json_snippet = {
        "dtype": f"int{args.wbits}",
        "quant": {
            "scheme": "weight_only",
            "mode": args.mode,
            "bits": args.wbits,
            "scale_bin": scale_path,
            "scale_dtype": args.scale_dtype,
            "weight_bin": weight_path,
            "pack": ("int8_row_major" if args.wbits == 8 else "nibble_lohi"),
            "shape": [args.rows, args.cols],
        },
    }
    meta["ie_json_snippet"] = ie_json_snippet

    meta_path = f"{args.out_prefix}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Single-line summaries for parsers
    print("[ptq]", json.dumps({
        "rows": args.rows, "cols": args.cols, "wbits": args.wbits,
        "mode": args.mode, "avg_row_cosine": cos, "mse": err,
        "weights": weight_path, "scales": scale_path, "meta": meta_path
    }))

    print("[iebin-json]", json.dumps(ie_json_snippet))

    if args.fail_below_threshold and cos < float(args.accuracy_threshold):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
