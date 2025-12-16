#!/usr/bin/env python3
"""
dedup_verify_int4.py

Lossless verifier for byte-oriented dedup artifacts.

This tool:
  - Loads model.dedup.json
  - Reads defaults/masks/exceptions binaries
  - Reconstructs each target tensor
  - Reads the original target tensor bytes from weights_bin using tensor map
  - Byte-compares reconstructed vs original

Inputs:
  - --model-dir: model directory containing weights_bin
  - --tensor-map: JSON mapping tensor indices -> (offset,nbytes)
  - --spec: dedup spec JSON (model.dedup.json)
  - --out-prefix: prefix used by extractor (to locate bin files if spec uses basenames)

Usage:
  python3 tools/dedup_verify_int4.py \
    --model-dir models/gpt-oss-20b \
    --tensor-map models/gpt-oss-20b/model.tensor_map.json \
    --spec models/gpt-oss-20b/model.dedup.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TensorRec:
    index: int
    name: str
    offset: int
    nbytes: int


def _load_json(path: str) -> object:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_all(path: str) -> bytes:
    """Read entire binary file into memory."""
    with open(path, "rb") as f:
        return f.read()


def _read_slice(path: str, offset: int, nbytes: int) -> bytes:
    """Read a slice from a file at a given offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise RuntimeError(f"Short read from {path}: expected {nbytes}, got {len(data)}")
    return data


def _ceil_div(a: int, b: int) -> int:
    """Compute ceil(a/b) for non-negative integers."""
    return (a + b - 1) // b


def _mask_nbytes(nbytes: int) -> int:
    """Return the number of mask bytes needed for a tensor of size nbytes."""
    return _ceil_div(nbytes, 8)


def _parse_tensor_map(tensor_map: object) -> Tuple[str, Dict[int, TensorRec]]:
    """Parse the tensor map JSON object into a lookup table."""
    if not isinstance(tensor_map, dict):
        raise ValueError("tensor map JSON must be an object")

    weights_bin = tensor_map.get("weights_bin")
    tensors = tensor_map.get("tensors")
    if not isinstance(weights_bin, str) or not isinstance(tensors, list):
        raise ValueError("tensor map JSON must contain 'weights_bin' (str) and 'tensors' (list)")

    out: Dict[int, TensorRec] = {}
    for t in tensors:
        idx = int(t["index"])
        name = str(t.get("name", f"tensor_{idx}"))
        off = int(t["offset"])
        nb = int(t["nbytes"])
        out[idx] = TensorRec(index=idx, name=name, offset=off, nbytes=nb)

    return weights_bin, out


def _reconstruct(default_bytes: bytes, mask: bytes, exc: bytes) -> bytes:
    """
    Reconstruct a target blob from (default_bytes, mask, exc).

    Mask semantics:
      - 1 bit per byte in default_bytes
      - if bit is 1 at byte index i, output byte at i comes from exc stream (in ascending i order)
    """
    n = len(default_bytes)
    if len(mask) != _mask_nbytes(n):
        raise ValueError("mask length mismatch")
    out = bytearray(default_bytes)

    exc_i = 0
    byte_i = 0

    for m in mask:
        if m == 0:
            byte_i += 8
            continue
        for bit in range(8):
            idx = byte_i + bit
            if idx >= n:
                break
            if (m >> bit) & 1:
                out[idx] = exc[exc_i]
                exc_i += 1
        byte_i += 8

    if exc_i != len(exc):
        raise RuntimeError(f"exception stream not fully consumed: used {exc_i}, have {len(exc)}")

    return bytes(out)


def main() -> int:
    """Program entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Model directory containing weights bin")
    ap.add_argument("--tensor-map", required=True, help="Tensor map JSON file")
    ap.add_argument("--spec", required=True, help="Dedup spec JSON file")
    args = ap.parse_args()

    tensor_map_obj = _load_json(args.tensor_map)
    weights_bin_rel, tensors = _parse_tensor_map(tensor_map_obj)
    weights_bin_path = os.path.join(args.model_dir, weights_bin_rel)

    spec = _load_json(args.spec)
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    if spec.get("magic") != "DUDP" or int(spec.get("version", 0)) != 1:
        raise ValueError("unsupported spec magic/version")

    files = spec.get("files", {})
    defaults_path = os.path.join(os.path.dirname(args.spec), files["defaults"])
    masks_path = os.path.join(os.path.dirname(args.spec), files["masks"])
    exc_path = os.path.join(os.path.dirname(args.spec), files["exceptions"])

    defaults = _read_all(defaults_path)
    masks = _read_all(masks_path)
    exc = _read_all(exc_path)

    groups = spec.get("groups")
    if not isinstance(groups, list):
        raise ValueError("spec.groups must be a list")

    checked = 0
    for g in groups:
        d = g["default"]
        default_index = int(d["tensor_index"])
        default_nbytes = int(d["nbytes"])
        default_off = int(d["default_off"])

        if default_index not in tensors:
            raise KeyError(f"default tensor_index {default_index} missing from tensor map")
        if tensors[default_index].nbytes != default_nbytes:
            raise ValueError("default nbytes mismatch vs tensor map")

        default_bytes = defaults[default_off : default_off + default_nbytes]
        if len(default_bytes) != default_nbytes:
            raise RuntimeError("default slice out of bounds in defaults.bin")

        for t in g["targets"]:
            tidx = int(t["tensor_index"])
            nbytes = int(t["nbytes"])
            mask_off = int(t["mask_off"])
            mask_n = int(t["mask_nbytes"])
            exc_off = int(t["exc_off"])
            exc_n = int(t["exc_nbytes"])

            if tidx not in tensors:
                raise KeyError(f"target tensor_index {tidx} missing from tensor map")
            if tensors[tidx].nbytes != nbytes:
                raise ValueError(f"target nbytes mismatch vs tensor map for {tidx}")
            if nbytes != default_nbytes:
                raise ValueError("target nbytes mismatch vs default nbytes")

            mask_bytes = masks[mask_off : mask_off + mask_n]
            if len(mask_bytes) != mask_n:
                raise RuntimeError("mask slice out of bounds in masks.bin")

            exc_bytes = exc[exc_off : exc_off + exc_n]
            if len(exc_bytes) != exc_n:
                raise RuntimeError("exception slice out of bounds in exceptions.bin")

            rec = _reconstruct(default_bytes, mask_bytes, exc_bytes)
            orig = _read_slice(weights_bin_path, tensors[tidx].offset, nbytes)

            if rec != orig:
                raise RuntimeError(f"mismatch for tensor_index={tidx} name={tensors[tidx].name}")

            checked += 1

    print(f"OK: verified {checked} tensors losslessly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
