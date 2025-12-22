#!/usr/bin/env python3
"""
Build a dedup/quant source bin+json from:
  - a base IE manifest/bin (model.ie.json + model.ie.bin)
  - a dedup pairing manifest (manifest.dedup.json)

It will:
  1) Copy all tensors referenced by kv_pair groups into a compact output bin.
  2) Ensure fused expert tensors referenced by expert_pair groups exist in the output.
     If missing in the base IE, it will synthesize them from per-expert weight tensors by:
       - symmetric int4 quantization (group-wise, default group size 64)
       - writing int4 blocks (packed nibbles) and per-group scales

This script does not require torch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


_DTYPE_TO_NP: Dict[str, np.dtype] = {
    "torch.float32": np.float32,
    "torch.float16": np.float16,
    "torch.bfloat16": np.uint16,  # stored as bf16 bits, decode manually
    "torch.uint8": np.uint8,
    "torch.int8": np.int8,
    "torch.int16": np.int16,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "U8": np.uint8,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def _parse_ie(ie_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    ie = _load_json(ie_path)
    weights_bin = ie.get("weights_bin") or ie.get("bin") or "model.ie.bin"
    tensors = ie.get("tensors")
    if tensors is None:
        raise SystemExit("ERROR: IE json missing 'tensors'")

    out: List[Dict[str, Any]] = []
    if isinstance(tensors, list):
        for t in tensors:
            if isinstance(t, dict):
                out.append(dict(t))
    elif isinstance(tensors, dict):
        for name, meta in tensors.items():
            if isinstance(meta, dict):
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
            npdt = _DTYPE_TO_NP.get(dtype)
            if npdt is None:
                raise SystemExit(f"ERROR: tensor '{name}' has unknown dtype '{dtype}' and missing nbytes")
            esz = int(np.dtype(npdt).itemsize)
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
    return weights_bin, norm


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _read_tensor_bytes(bin_path: str, t: Dict[str, Any]) -> bytes:
    with open(bin_path, "rb") as f:
        f.seek(int(t["offset"]))
        return f.read(int(t["nbytes"]))


def _read_weight_matrix_f32(bin_path: str, t: Dict[str, Any]) -> np.ndarray:
    shape = t["shape"]
    if len(shape) != 2:
        raise SystemExit(f"ERROR: expected 2D weight for '{t['name']}', got shape={shape}")
    raw = _read_tensor_bytes(bin_path, t)
    dtype = t["dtype"]
    npdt = _DTYPE_TO_NP.get(dtype)
    if npdt is None:
        raise SystemExit(f"ERROR: unsupported dtype '{dtype}' for '{t['name']}'")
    if dtype in ("torch.bfloat16", "BF16"):
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        return _bf16_to_f32(u16)
    arr = np.frombuffer(raw, dtype=npdt).reshape(shape)
    return arr.astype(np.float32, copy=False)


def _pack_int4_nibbles(q: np.ndarray) -> np.ndarray:
    if q.dtype != np.uint8:
        q = q.astype(np.uint8, copy=False)
    if q.shape[-1] % 2 == 1:
        pad = np.zeros(q.shape[:-1] + (1,), dtype=np.uint8)
        q = np.concatenate([q, pad], axis=-1)
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8, copy=False)


def _quant_int4_groupwise(w: np.ndarray, group_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if w.dtype != np.float32:
        w = w.astype(np.float32, copy=False)
    rows, cols = w.shape
    groups = int(math.ceil(cols / group_size))
    pad_cols = groups * group_size
    if pad_cols != cols:
        wpad = np.zeros((rows, pad_cols), dtype=np.float32)
        wpad[:, :cols] = w
        w = wpad
        cols = pad_cols

    w3 = w.reshape(rows, groups, group_size)
    absmax = np.max(np.abs(w3), axis=2)
    scales = absmax / 7.0
    scales = np.where(scales == 0.0, 1.0, scales).astype(np.float32)

    q = np.rint(w3 / scales[:, :, None]).astype(np.int32)
    q = np.clip(q, -7, 7).astype(np.int16)
    q_u8 = (q + 8).astype(np.uint8)

    packed = _pack_int4_nibbles(q_u8)
    packed = packed.reshape(rows, -1)
    return packed, scales


def _encode_scales_fp16(scales_f32: np.ndarray) -> bytes:
    return scales_f32.astype(np.float16).tobytes(order="C")


def _encode_scales_log2_u8_q3(scales_f32: np.ndarray, bias: int = 128, step: float = 0.125) -> bytes:
    s = np.maximum(scales_f32, np.float32(2 ** -30))
    l = np.log2(s) / np.float32(step)
    code = np.rint(l).astype(np.int32) + int(bias)
    code = np.clip(code, 0, 255).astype(np.uint8)
    return code.tobytes(order="C")


@dataclass
class OutTensor:
    name: str
    dtype: str
    shape: List[int]
    offset: int
    nbytes: int


def _manifest_required_tensors(manifest: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    groups = manifest.get("groups", manifest)
    if not isinstance(groups, list):
        raise SystemExit("ERROR: manifest is neither a list nor an object with 'groups'")

    kv: List[str] = []
    fused: List[str] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        if g.get("kind") == "kv_pair":
            members = g.get("members")
            if isinstance(members, list):
                for m in members:
                    if isinstance(m, str):
                        kv.append(m)
        elif g.get("kind") == "expert_pair":
            ft = g.get("fused_tensor")
            if isinstance(ft, str):
                fused.append(ft)
    return sorted(set(kv)), sorted(set(fused))


def _infer_expert_weight_names(fused_tensor: str, expert_count: int) -> Tuple[str, List[str]]:
    base_prefix, last = fused_tensor.rsplit(".", 1)
    if last.endswith("_scales"):
        proj = last[: -len("_scales")]
    elif last.endswith("_blocks"):
        proj = last[: -len("_blocks")]
    else:
        raise SystemExit(f"ERROR: cannot infer per-expert weights from '{fused_tensor}'")

    per = [f"{base_prefix}.{e}.{proj}.weight" for e in range(expert_count)]
    return proj, per


def _discover_expert_count(all_names: List[str], fused_tensor: str) -> int:
    base_prefix, last = fused_tensor.rsplit(".", 1)
    if last.endswith("_scales"):
        proj = last[: -len("_scales")]
    elif last.endswith("_blocks"):
        proj = last[: -len("_blocks")]
    else:
        raise SystemExit(f"ERROR: cannot infer expert_count from '{fused_tensor}'")

    pat = re.compile(re.escape(base_prefix) + r"\.(\d+)\." + re.escape(proj) + r"\.weight$")
    idxs: List[int] = []
    for name in all_names:
        m = pat.match(name)
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        raise SystemExit(f"ERROR: no per-expert weights found for '{fused_tensor}'")
    max_idx = max(idxs)
    missing = [i for i in range(max_idx + 1) if i not in set(idxs)]
    if missing:
        raise SystemExit(f"ERROR: missing expert indices for '{fused_tensor}': missing={missing[:10]} total_missing={len(missing)}")
    return max_idx + 1


def _build_dedup_source(
    base_ie_json: str,
    base_bin_path: str,
    manifest_path: str,
    out_bin_path: str,
    out_ie_json_path: str,
    group_size: int,
    scale_encoding: str,
    include_blocks: bool,
) -> None:
    weights_bin_rel, tensors = _parse_ie(base_ie_json)
    base_dir = os.path.dirname(os.path.abspath(base_ie_json))
    base_bin = base_bin_path or os.path.join(base_dir, weights_bin_rel)

    by_name: Dict[str, Dict[str, Any]] = {t["name"]: t for t in tensors}
    man = _load_json(manifest_path)
    kv_names, fused_names = _manifest_required_tensors(man)

    out_tensors: List[OutTensor] = []
    cursor = 0

    os.makedirs(os.path.dirname(os.path.abspath(out_bin_path)), exist_ok=True)
    with open(out_bin_path, "wb") as out_f:
        for name in kv_names:
            t = by_name.get(name)
            if t is None:
                raise SystemExit(f"ERROR: kv tensor '{name}' not found in base IE json")
            blob = _read_tensor_bytes(base_bin, t)
            out_f.write(blob)
            out_tensors.append(OutTensor(name=name, dtype=t["dtype"], shape=list(t["shape"]), offset=cursor, nbytes=len(blob)))
            cursor += len(blob)

        all_names = list(by_name.keys())
        for fused in fused_names:
            if fused in by_name:
                t = by_name[fused]
                blob = _read_tensor_bytes(base_bin, t)
                out_f.write(blob)
                out_tensors.append(OutTensor(name=fused, dtype=t["dtype"], shape=list(t["shape"]), offset=cursor, nbytes=len(blob)))
                cursor += len(blob)
                continue

            expert_count = _discover_expert_count(all_names, fused)
            proj, per_names = _infer_expert_weight_names(fused, expert_count)

            blocks_list: List[np.ndarray] = []
            scales_list: List[np.ndarray] = []
            rows = None
            cols = None

            for pn in per_names:
                wt = by_name.get(pn)
                if wt is None:
                    raise SystemExit(f"ERROR: per-expert weight '{pn}' not found (needed for '{fused}')")
                w = _read_weight_matrix_f32(base_bin, wt)
                if rows is None:
                    rows, cols = w.shape
                elif w.shape != (rows, cols):
                    raise SystemExit(f"ERROR: shape mismatch '{pn}': {w.shape} vs {(rows, cols)}")
                b, s = _quant_int4_groupwise(w, group_size=group_size)
                blocks_list.append(b)
                scales_list.append(s)

            blocks_fused = np.stack(blocks_list, axis=0)
            scales_fused = np.stack(scales_list, axis=0)

            base_prefix, last = fused.rsplit(".", 1)
            if last.endswith("_scales"):
                fused_scales_name = fused
                fused_blocks_name = f"{base_prefix}.{proj}_blocks"
            elif last.endswith("_blocks"):
                fused_blocks_name = fused
                fused_scales_name = f"{base_prefix}.{proj}_scales"
            else:
                raise SystemExit(f"ERROR: unsupported fused name '{fused}'")

            if scale_encoding == "fp16":
                scales_bytes = _encode_scales_fp16(scales_fused)
                scales_dtype = "torch.float16"
            elif scale_encoding == "log2_u8_q3":
                scales_bytes = _encode_scales_log2_u8_q3(scales_fused)
                scales_dtype = "torch.uint8"
            else:
                raise SystemExit(f"ERROR: unknown scale encoding '{scale_encoding}'")

            out_f.write(scales_bytes)
            out_tensors.append(
                OutTensor(
                    name=fused_scales_name,
                    dtype=scales_dtype,
                    shape=[expert_count, int(rows), int(scales_fused.shape[2])],
                    offset=cursor,
                    nbytes=len(scales_bytes),
                )
            )
            cursor += len(scales_bytes)

            if include_blocks:
                blocks_bytes = blocks_fused.astype(np.uint8).tobytes(order="C")
                out_f.write(blocks_bytes)
                out_tensors.append(
                    OutTensor(
                        name=fused_blocks_name,
                        dtype="torch.uint8",
                        shape=[expert_count, int(rows), int(blocks_fused.shape[2])],
                        offset=cursor,
                        nbytes=len(blocks_bytes),
                    )
                )
                cursor += len(blocks_bytes)

    out_ie = {
        "version": 1,
        "dtype": "mixed",
        "weights_bin": os.path.basename(out_bin_path),
        "tensors": [
            {"name": t.name, "dtype": t.dtype, "shape": t.shape, "offset": t.offset, "nbytes": t.nbytes}
            for t in out_tensors
        ],
        "quant": {
            "int4": {
                "group_size": int(group_size),
                "scale_encoding": scale_encoding,
                "pack": "nibble_lohi",
                "range": [-7, 7],
            }
        },
    }
    _save_json(out_ie_json_path, out_ie)

    print(f"Wrote: {out_bin_path}")
    print(f"Wrote: {out_ie_json_path}")
    print(f"Tensors: {len(out_tensors)}  Bytes: {sum(t.nbytes for t in out_tensors)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ie-json", required=True)
    ap.add_argument("--base-bin", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-bin", required=True)
    ap.add_argument("--out-ie-json", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--scale-encoding", choices=["fp16", "log2_u8_q3"], default="fp16")
    ap.add_argument("--include-blocks", action="store_true")
    args = ap.parse_args()

    _build_dedup_source(
        base_ie_json=args.base_ie_json,
        base_bin_path=args.base_bin,
        manifest_path=args.manifest,
        out_bin_path=args.out_bin,
        out_ie_json_path=args.out_ie_json,
        group_size=args.group_size,
        scale_encoding=args.scale_encoding,
        include_blocks=bool(args.include_blocks),
    )


if __name__ == "__main__":
    main()
