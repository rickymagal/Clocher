#!/usr/bin/env python3
"""
dedup_extract_int4.py

Lossless dedup extractor for byte-oriented tensors (including packed INT4 weights).

This tool creates:
  - model.defaults.bin
  - model.masks.bin
  - model.exceptions.bin
  - model.dedup.json

The core contract is byte-level and therefore dtype-agnostic:
  - A tensor is reconstructed exactly as raw bytes.
  - The mask has 1 bit per tensor byte.
  - Exceptions store the exact original bytes at masked indices.

Inputs:
  1) A tensor map JSON describing where each tensor lives in the binary file.
     Expected minimal schema:
     {
       "weights_bin": "model.ie.bin",
       "tensors": [
         { "index": 0, "name": "...", "offset": 123, "nbytes": 456 },
         ...
       ]
     }

  2) A groups JSON describing dedup groups:
     [
       {
         "default_index": 17,
         "targets": [18, 19, 20]
       },
       ...
     ]

Usage:
  python3 tools/dedup_extract_int4.py \
    --model-dir models/gpt-oss-20b \
    --tensor-map models/gpt-oss-20b/model.tensor_map.json \
    --groups tools/dedup_groups.json \
    --out-prefix models/gpt-oss-20b/model

Notes:
  - This script intentionally does not parse safetensors or HF formats.
  - It expects a simple (offset,nbytes) map to an existing binary file.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


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


def _write_json(path: str, obj: object) -> None:
    """Write a JSON file with stable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _ceil_div(a: int, b: int) -> int:
    """Compute ceil(a/b) for non-negative integers."""
    return (a + b - 1) // b


def _mask_bytes(nbytes: int) -> int:
    """Return the number of mask bytes needed for a tensor of size nbytes."""
    return _ceil_div(nbytes, 8)


def _bitset_set(mask: bytearray, bit_index: int) -> None:
    """Set one bit in a little-endian bitset (bit 0 is LSB of byte 0)."""
    byte_i = bit_index // 8
    bit_i = bit_index % 8
    mask[byte_i] |= (1 << bit_i)


def _bitset_popcount(mask: bytes) -> int:
    """Count set bits in a bitset."""
    return sum(int(b).bit_count() for b in mask)


def _read_slice(path: str, offset: int, nbytes: int) -> bytes:
    """Read a byte slice from a file at a given offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise RuntimeError(f"Short read from {path}: expected {nbytes}, got {len(data)}")
    return data


def _diff_default_target(default_bytes: bytes, target_bytes: bytes) -> Tuple[bytes, bytes]:
    """
    Compute (mask, exceptions) between default_bytes and target_bytes.

    The mask has 1 bit per byte; a 1 means target differs from default at that byte.
    Exceptions are the raw target bytes at set-bit positions in ascending index order.
    """
    if len(default_bytes) != len(target_bytes):
        raise ValueError("default_bytes and target_bytes must have same length")

    n = len(target_bytes)
    mask = bytearray(_mask_bytes(n))
    exc = bytearray()

    for i in range(n):
        if target_bytes[i] != default_bytes[i]:
            _bitset_set(mask, i)
            exc.append(target_bytes[i])

    return bytes(mask), bytes(exc)


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
        if not isinstance(t, dict):
            raise ValueError("tensor entry must be an object")
        idx = int(t["index"])
        name = str(t.get("name", f"tensor_{idx}"))
        off = int(t["offset"])
        nb = int(t["nbytes"])
        out[idx] = TensorRec(index=idx, name=name, offset=off, nbytes=nb)

    return weights_bin, out


def main() -> int:
    """Program entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Model directory containing weights_bin")
    ap.add_argument("--tensor-map", required=True, help="JSON mapping tensor indices to (offset,nbytes)")
    ap.add_argument("--groups", required=True, help="JSON describing dedup groups")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (e.g., models/.../model)")
    args = ap.parse_args()

    tensor_map_obj = _load_json(args.tensor_map)
    weights_bin_rel, tensors = _parse_tensor_map(tensor_map_obj)

    weights_bin_path = os.path.join(args.model_dir, weights_bin_rel)
    if not os.path.exists(weights_bin_path):
        raise FileNotFoundError(weights_bin_path)

    groups_obj = _load_json(args.groups)
    if not isinstance(groups_obj, list):
        raise ValueError("groups JSON must be a list")

    defaults_path = args.out_prefix + ".defaults.bin"
    masks_path = args.out_prefix + ".masks.bin"
    exceptions_path = args.out_prefix + ".exceptions.bin"
    spec_path = args.out_prefix + ".dedup.json"

    defaults_off = 0
    masks_off = 0
    exc_off = 0

    spec_groups: List[dict] = []

    with open(defaults_path, "wb") as f_def, open(masks_path, "wb") as f_mask, open(exceptions_path, "wb") as f_exc:
        for g in groups_obj:
            if not isinstance(g, dict):
                raise ValueError("each group must be an object")
            default_index = int(g["default_index"])
            target_indices = [int(x) for x in g["targets"]]

            if default_index not in tensors:
                raise KeyError(f"default_index {default_index} not in tensor map")

            drec = tensors[default_index]
            default_bytes = _read_slice(weights_bin_path, drec.offset, drec.nbytes)

            group_entry = {
                "default": {
                    "tensor_index": default_index,
                    "nbytes": drec.nbytes,
                    "default_off": defaults_off,
                },
                "targets": [],
            }

            # Write default blob.
            f_def.write(default_bytes)
            defaults_off += drec.nbytes

            for tidx in target_indices:
                if tidx not in tensors:
                    raise KeyError(f"target tensor_index {tidx} not in tensor map")

                trec = tensors[tidx]
                if trec.nbytes != drec.nbytes:
                    raise ValueError(
                        f"Size mismatch: default {default_index} nbytes={drec.nbytes} vs target {tidx} nbytes={trec.nbytes}"
                    )

                target_bytes = _read_slice(weights_bin_path, trec.offset, trec.nbytes)
                mask, exc = _diff_default_target(default_bytes, target_bytes)

                # Write mask and exceptions.
                f_mask.write(mask)
                f_exc.write(exc)

                t_entry = {
                    "tensor_index": tidx,
                    "nbytes": trec.nbytes,
                    "mask_off": masks_off,
                    "mask_nbytes": len(mask),
                    "exc_off": exc_off,
                    "exc_nbytes": len(exc),
                    "exc_popcount": _bitset_popcount(mask),
                }

                if t_entry["exc_popcount"] != t_entry["exc_nbytes"]:
                    raise RuntimeError("Internal error: popcount(mask) != exc_nbytes")

                group_entry["targets"].append(t_entry)

                masks_off += len(mask)
                exc_off += len(exc)

            spec_groups.append(group_entry)

    spec = {
        "magic": "DUDP",
        "version": 1,
        "files": {
            "defaults": os.path.basename(defaults_path),
            "masks": os.path.basename(masks_path),
            "exceptions": os.path.basename(exceptions_path),
        },
        "totals": {
            "defaults_size": defaults_off,
            "masks_size": masks_off,
            "exceptions_size": exc_off,
        },
        "groups": spec_groups,
    }

    _write_json(spec_path, spec)

    print(f"Wrote:\n  {spec_path}\n  {defaults_path}\n  {masks_path}\n  {exceptions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
