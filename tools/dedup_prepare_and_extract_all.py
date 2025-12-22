#!/usr/bin/env python3
"""
Prepare inputs for the lossless byte-level dedup extractor.

This script converts:
  - an IE weight manifest (model.ie*.json) into tensor_map.for_dedup.json
  - a dedup pairing manifest (manifest.dedup.json) into groups.for_dedup.json

It is intentionally tolerant of multiple IE JSON layouts:
  - weights_bin vs bin
  - tensors as a list vs a dict keyed by tensor name
  - dtype spelled as torch.* or short forms (BF16/F16/F32/U8/I32)

Outputs match tools/dedup_extract_int4.py:
  groups: [{"default_index": int, "targets": [int, ...]}, ...]
  tensor_map: {"weights_bin": "file.bin", "tensors": [{"index": i, "name": ..., "offset": ..., "nbytes": ...}, ...]}
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


_DTYPE_SIZES: Dict[str, int] = {
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


def _dtype_size(dtype: str) -> int:
    if dtype in _DTYPE_SIZES:
        return _DTYPE_SIZES[dtype]
    raise SystemExit(f"ERROR: unsupported dtype '{dtype}'")


def _as_list_tensors(ie: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Return (weights_bin_rel, tensors_list) where each tensor has:
      name, dtype, shape, offset, nbytes
    """
    weights_bin = ie.get("weights_bin") or ie.get("bin")
    if not isinstance(weights_bin, str) or not weights_bin:
        weights_bin = "model.ie.bin"

    tensors = ie.get("tensors")
    if tensors is None:
        raise SystemExit("ERROR: IE json missing 'tensors'")

    out: List[Dict[str, Any]] = []
    if isinstance(tensors, list):
        for t in tensors:
            if not isinstance(t, dict):
                continue
            out.append(dict(t))
    elif isinstance(tensors, dict):
        for name, meta in tensors.items():
            if not isinstance(meta, dict):
                continue
            d = dict(meta)
            d.setdefault("name", name)
            out.append(d)
    else:
        raise SystemExit(f"ERROR: unsupported tensors container type: {type(tensors)}")

    norm: List[Dict[str, Any]] = []
    for t in out:
        name = t.get("name")
        dtype = t.get("dtype")
        shape = t.get("shape")
        offset = t.get("offset")
        nbytes = t.get("nbytes")

        if not isinstance(name, str) or not name:
            continue
        if not isinstance(dtype, str) or not dtype:
            raise SystemExit(f"ERROR: tensor '{name}' missing dtype")
        if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape):
            raise SystemExit(f"ERROR: tensor '{name}' has invalid shape")
        if not isinstance(offset, int):
            raise SystemExit(f"ERROR: tensor '{name}' missing offset")

        if not isinstance(nbytes, int):
            esz = _dtype_size(dtype)
            elems = 1
            for d in shape:
                elems *= int(d)
            nbytes = elems * esz

        norm.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "offset": int(offset),
                "nbytes": int(nbytes),
            }
        )

    norm.sort(key=lambda x: (x["offset"], x["name"]))
    for i, t in enumerate(norm):
        t["index"] = i
    return weights_bin, norm


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def _build_name_to_index(tensors: List[Dict[str, Any]]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for t in tensors:
        m[t["name"]] = int(t["index"])
    return m


def _make_virtual_slices(base: Dict[str, Any], slice_count: int) -> List[Dict[str, Any]]:
    """
    Create virtual tensors representing slicing along dimension 0 of a fused tensor.

    Slice size is derived from nbytes/slice_count (byte-accurate).
    Slice shape is base.shape[1:].
    """
    base_shape = base["shape"]
    if not base_shape or not isinstance(base_shape[0], int):
        raise SystemExit(f"ERROR: base tensor '{base['name']}' has invalid shape")
    if base_shape[0] != slice_count:
        raise SystemExit(
            f"ERROR: fused tensor '{base['name']}' shape[0]={base_shape[0]} "
            f"does not match expected expert_count={slice_count}"
        )

    total_bytes = int(base["nbytes"])
    if total_bytes % slice_count != 0:
        raise SystemExit(
            f"ERROR: fused tensor '{base['name']}' nbytes={total_bytes} is not divisible by expert_count={slice_count}"
        )
    slice_nbytes = total_bytes // slice_count

    slice_shape = list(base_shape[1:])
    if not slice_shape:
        slice_shape = [1]

    slices: List[Dict[str, Any]] = []
    for i in range(slice_count):
        slices.append(
            {
                "name": f"{base['name']}#slice{i}",
                "dtype": base["dtype"],
                "shape": slice_shape,
                "offset": int(base["offset"]) + i * slice_nbytes,
                "nbytes": slice_nbytes,
            }
        )
    return slices


def _prepare_groups_and_tensor_map(
    ie_json_path: str,
    manifest_path: str,
    out_tensor_map_path: str,
    out_groups_path: str,
) -> None:
    ie = _load_json(ie_json_path)
    weights_bin, base_tensors = _as_list_tensors(ie)

    name_to_index = _build_name_to_index(base_tensors)
    compat_tensors = list(base_tensors)
    compat_name_to_index = dict(name_to_index)

    out_groups: List[Dict[str, Any]] = []

    man = _load_json(manifest_path)
    groups = man.get("groups", man)
    if not isinstance(groups, list):
        raise SystemExit("ERROR: manifest is neither a list nor an object with 'groups'")

    def ensure_virtual_slice(base_name: str, slice_idx: int, expected_slices: int) -> int:
        vname = f"{base_name}#slice{slice_idx}"
        if vname in compat_name_to_index:
            return compat_name_to_index[vname]
        base_idx = name_to_index.get(base_name)
        if base_idx is None:
            raise SystemExit(f"ERROR: fused tensor '{base_name}' not found in IE json")
        base = base_tensors[base_idx]
        new_slices = _make_virtual_slices(base, expected_slices)
        start_index = len(compat_tensors)
        for j, s in enumerate(new_slices):
            s["index"] = start_index + j
            compat_tensors.append(s)
            compat_name_to_index[s["name"]] = s["index"]
        return compat_name_to_index[vname]

    for g in groups:
        if not isinstance(g, dict):
            continue
        kind = g.get("kind")
        if kind == "kv_pair":
            members = g.get("members")
            if not isinstance(members, list) or len(members) < 2:
                continue
            member_indices: List[int] = []
            for name in members:
                if not isinstance(name, str):
                    continue
                idx = name_to_index.get(name)
                if idx is None:
                    raise SystemExit(f"ERROR: kv_pair member '{name}' not found in IE json")
                member_indices.append(idx)
            if len(member_indices) >= 2:
                out_groups.append({"default_index": member_indices[0], "targets": member_indices[1:]})

        elif kind == "expert_pair":
            fused_name = g.get("fused_tensor")
            expert_indices = g.get("expert_indices")
            if not isinstance(fused_name, str) or not fused_name:
                continue
            if not isinstance(expert_indices, list) or len(expert_indices) < 2:
                continue
            base_idx = name_to_index.get(fused_name)
            if base_idx is None:
                raise SystemExit(f"ERROR: expert_pair fused_tensor '{fused_name}' not found in IE json")
            base = base_tensors[base_idx]
            expert_count = int(base["shape"][0])

            slice_indices: List[int] = []
            for ei in expert_indices:
                if not isinstance(ei, int):
                    continue
                slice_indices.append(ensure_virtual_slice(fused_name, ei, expert_count))
            if len(slice_indices) >= 2:
                out_groups.append({"default_index": slice_indices[0], "targets": slice_indices[1:]})

    tensor_map = {
        "weights_bin": weights_bin,
        "tensors": [
            {
                "index": int(t["index"]),
                "name": t["name"],
                "offset": int(t["offset"]),
                "nbytes": int(t["nbytes"]),
            }
            for t in compat_tensors
        ],
    }

    _save_json(out_tensor_map_path, tensor_map)
    _save_json(out_groups_path, out_groups)

    print(f"Wrote: {out_tensor_map_path}")
    print(f"Wrote: {out_groups_path}")
    print(f"Tensors: base={len(base_tensors)} total(with slices)={len(compat_tensors)} groups={len(out_groups)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie-json", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-tensor-map", required=True)
    ap.add_argument("--out-groups", required=True)
    args = ap.parse_args()

    _prepare_groups_and_tensor_map(
        ie_json_path=args.ie_json,
        manifest_path=args.manifest,
        out_tensor_map_path=args.out_tensor_map,
        out_groups_path=args.out_groups,
    )


if __name__ == "__main__":
    main()
