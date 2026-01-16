#!/usr/bin/env python3
"""
build_hybrid_moe_int4.py

Repack an existing IEBIN tensor map + weights binary into a new aligned binary,
and emit a compatible tensor map that points to the new weights bin.

This tool is intentionally tolerant of older tensor-map JSON formats that do not
include a top-level "weights_bin" field. In that case, it defaults to
"model.ie.bin" inside --model-dir (or --in-bin if provided).

Typical usage (matches your invocation):

  MODEL_DIR=".../models/gpt-oss-20b"
  python3 tools/build_hybrid_moe_int4.py \
    --model-dir "$MODEL_DIR" \
    --out-bin "$MODEL_DIR/model.q4.bin" \
    --out-map "$MODEL_DIR/model.ie.compat.json" \
    --align 256

Notes:
- This script does not perform quantization. It repacks whatever tensors already
  exist in the input weights bin (BF16 weights, INT4 blocks/scales, etc.).
- The engine selects the INT4 artifact pair via --precision int4, which maps to
  model.ie.compat.json + model.q4.bin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TensorRec:
    name: str
    dtype: str
    shape: List[int]
    offset: int
    nbytes: int
    raw: Dict[str, Any]


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _align_up(x: int, a: int) -> int:
    if a <= 0:
        return x
    r = x % a
    return x if r == 0 else x + (a - r)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def _infer_in_bin(model_dir: str, m: Dict[str, Any], in_bin_override: Optional[str]) -> str:
    if in_bin_override:
        p = in_bin_override
        if not os.path.isabs(p):
            p = os.path.join(model_dir, p)
        if not os.path.exists(p):
            _die(f"--in-bin {p} does not exist")
        return p

    wb = m.get("weights_bin")
    if isinstance(wb, str) and wb.strip():
        p = wb
        if not os.path.isabs(p):
            p = os.path.join(model_dir, p)
        if os.path.exists(p):
            return p

    # Back-compat default.
    p = os.path.join(model_dir, "model.ie.bin")
    if os.path.exists(p):
        return p

    # Last resort: find the largest *.bin file in model_dir.
    best = None
    best_sz = -1
    try:
        for fn in os.listdir(model_dir):
            if not fn.endswith(".bin"):
                continue
            fp = os.path.join(model_dir, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            if sz > best_sz:
                best_sz = sz
                best = fp
    except FileNotFoundError:
        pass

    if best:
        return best

    _die(f"could not infer input weights bin (no weights_bin, no model.ie.bin, no *.bin in {model_dir})")
    return ""


def _parse_tensors(m: Dict[str, Any]) -> List[TensorRec]:
    ts = m.get("tensors")
    if not isinstance(ts, list) or not ts:
        _die("tensor map missing/invalid tensors[]")

    out: List[TensorRec] = []
    for t in ts:
        if not isinstance(t, dict):
            _die("tensor map tensors[] contains non-object")
        name = t.get("name")
        dtype = t.get("dtype")
        shape = t.get("shape")
        offset = t.get("offset")
        nbytes = t.get("nbytes")
        if not isinstance(name, str) or not name:
            _die("tensor missing/invalid name")
        if not isinstance(dtype, str) or not dtype:
            _die(f"{name}: tensor missing/invalid dtype")
        if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape):
            _die(f"{name}: tensor missing/invalid shape")
        if not isinstance(offset, int) or offset < 0:
            _die(f"{name}: tensor missing/invalid offset")
        if not isinstance(nbytes, int) or nbytes <= 0:
            _die(f"{name}: tensor missing/invalid nbytes")

        out.append(TensorRec(
            name=name,
            dtype=dtype,
            shape=shape,
            offset=offset,
            nbytes=nbytes,
            raw=t,
        ))

    # Sort by file offset so we can stream copies.
    out.sort(key=lambda r: r.offset)
    return out


def _copy_range(fin, fout, src_off: int, nbytes: int) -> None:
    fin.seek(src_off, os.SEEK_SET)
    remaining = nbytes
    buf_sz = 4 * 1024 * 1024
    while remaining > 0:
        take = buf_sz if remaining > buf_sz else remaining
        chunk = fin.read(take)
        if not chunk or len(chunk) != take:
            _die(f"short read at offset={src_off} (wanted {nbytes} bytes)")
        fout.write(chunk)
        remaining -= take


def _repack(model_dir: str,
            in_map_path: str,
            in_bin_path: str,
            out_bin_path: str,
            out_map_path: str,
            align: int) -> None:
    m = _load_json(in_map_path)
    tensors = _parse_tensors(m)

    os.makedirs(os.path.dirname(out_bin_path) or ".", exist_ok=True)

    # Stream-copy each tensor into the new bin with alignment.
    new_tensors: List[Dict[str, Any]] = []
    cur = 0

    with open(in_bin_path, "rb") as fin, open(out_bin_path, "wb") as fout:
        for r in tensors:
            cur = _align_up(cur, align)
            dst_off = cur

            # Ensure output file is positioned correctly.
            fout.seek(dst_off, os.SEEK_SET)

            _copy_range(fin, fout, r.offset, r.nbytes)

            cur = dst_off + r.nbytes

            t2 = dict(r.raw)
            t2["offset"] = dst_off
            t2["nbytes"] = r.nbytes
            new_tensors.append(t2)

    out_map = dict(m)
    out_map["weights_bin"] = os.path.basename(out_bin_path)
    out_map["tensors"] = new_tensors

    _write_json(out_map_path, out_map)

    out_sz = os.path.getsize(out_bin_path)
    print(f"[int4-artifacts] in_map={in_map_path}", file=sys.stderr)
    print(f"[int4-artifacts] in_bin={in_bin_path}", file=sys.stderr)
    print(f"[int4-artifacts] out_bin={out_bin_path} ({out_sz} bytes)", file=sys.stderr)
    print(f"[int4-artifacts] out_map={out_map_path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Model directory containing model.ie.json/bin.")
    ap.add_argument("--out-bin", required=True, help="Output weights bin path (e.g., model.q4.bin).")
    ap.add_argument("--out-map", required=True, help="Output map path (e.g., model.ie.compat.json).")
    ap.add_argument("--align", type=int, default=256, help="Byte alignment for each tensor start.")
    ap.add_argument("--in-map", default=None, help="Override input map path (default: model.ie.json in model-dir).")
    ap.add_argument("--in-bin", default=None, help="Override input weights bin (default: infer).")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)

    in_map_path = args.in_map
    if in_map_path is None:
        in_map_path = os.path.join(model_dir, "model.ie.json")
    elif not os.path.isabs(in_map_path):
        in_map_path = os.path.join(model_dir, in_map_path)

    if not os.path.exists(in_map_path):
        _die(f"input map does not exist: {in_map_path}")

    m = _load_json(in_map_path)
    in_bin_path = _infer_in_bin(model_dir, m, args.in_bin)

    out_bin_path = args.out_bin
    if not os.path.isabs(out_bin_path):
        out_bin_path = os.path.join(model_dir, out_bin_path)

    out_map_path = args.out_map
    if not os.path.isabs(out_map_path):
        out_map_path = os.path.join(model_dir, out_map_path)

    if args.align < 1 or (args.align & (args.align - 1)) != 0:
        _die("--align must be a positive power of two (e.g., 256)")

    _repack(
        model_dir=model_dir,
        in_map_path=in_map_path,
        in_bin_path=in_bin_path,
        out_bin_path=out_bin_path,
        out_map_path=out_map_path,
        align=args.align,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
