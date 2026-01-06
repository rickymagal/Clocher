#!/usr/bin/env python3
import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

ALIGN = 64
BLOCK_K = 32  # must match infer_gptoss.c

def align_up(x: int, a: int = ALIGN) -> int:
    return (x + (a - 1)) & ~(a - 1)

def read_safetensors_header(f):
    hdr_len_bytes = f.read(8)
    if len(hdr_len_bytes) != 8:
        raise RuntimeError("Failed to read safetensors header length")
    (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
    hdr_json = f.read(hdr_len)
    if len(hdr_json) != hdr_len:
        raise RuntimeError("Failed to read safetensors header JSON")
    header = json.loads(hdr_json.decode("utf-8"))
    return header, 8 + hdr_len

def bf16_to_f32_from_u16(u16: np.ndarray) -> np.ndarray:
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32)

def f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)

class HFIndex:
    def __init__(self, hf_dir: Path):
        idx_path = hf_dir / "model.safetensors.index.json"
        if not idx_path.is_file():
            raise SystemExit(f"ERROR: missing {idx_path}")
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        wm = idx.get("weight_map", {})
        if not wm:
            raise SystemExit("ERROR: model.safetensors.index.json has empty weight_map")
        self.hf_dir = hf_dir
        self.weight_map = wm
        self._shard_cache = {}  # shard_name -> (header, data_base)

    def _load_shard_header(self, shard_name: str):
        if shard_name in self._shard_cache:
            return self._shard_cache[shard_name]
        shard_path = self.hf_dir / shard_name
        if not shard_path.is_file():
            raise SystemExit(f"ERROR: missing shard {shard_path}")
        with open(shard_path, "rb") as f:
            header, data_base = read_safetensors_header(f)
        self._shard_cache[shard_name] = (header, data_base)
        return header, data_base

    def get_meta(self, tensor_name: str):
        shard_name = self.weight_map.get(tensor_name)
        if shard_name is None:
            raise KeyError(tensor_name)
        header, data_base = self._load_shard_header(shard_name)
        meta = header.get(tensor_name)
        if meta is None:
            raise SystemExit(f"ERROR: tensor '{tensor_name}' not found in header of {shard_name}")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offs = meta.get("data_offsets")
        if dtype is None or shape is None or offs is None:
            raise SystemExit(f"ERROR: malformed meta for '{tensor_name}' in {shard_name}")
        start, end = int(offs[0]), int(offs[1])
        nbytes = end - start
        if nbytes <= 0:
            raise SystemExit(f"ERROR: bad data_offsets for '{tensor_name}': {offs}")
        return shard_name, data_base, dtype, [int(x) for x in shape], start, nbytes

    def read_tensor_f32(self, tensor_name: str) -> np.ndarray:
        shard_name, data_base, dtype, shape, start, nbytes = self.get_meta(tensor_name)
        shard_path = self.hf_dir / shard_name
        with open(shard_path, "rb") as f:
            f.seek(data_base + start, os.SEEK_SET)
            raw = f.read(nbytes)
            if len(raw) != nbytes:
                raise SystemExit(f"ERROR: unexpected EOF reading '{tensor_name}' from {shard_name}")

        dt = str(dtype).upper()
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = bf16_to_f32_from_u16(u16)
        elif dt in ("F16", "FLOAT16"):
            f16 = np.frombuffer(raw, dtype=np.float16)
            f32 = f16.astype(np.float32)
        elif dt in ("F32", "FLOAT32"):
            f32 = np.frombuffer(raw, dtype=np.float32)
        else:
            raise SystemExit(f"ERROR: unsupported dtype '{dtype}' for '{tensor_name}' (expected BF16/F16/F32)")

        if np.prod(shape, dtype=np.int64) * 4 != f32.nbytes:
            raise SystemExit(f"ERROR: size mismatch for '{tensor_name}': shape={shape} dtype={dtype} bytes={nbytes}")

        return f32.reshape(shape)

def infer_num_experts_from_index(weight_map: dict, layer: int) -> int:
    prefix = f"model.layers.{layer}.mlp.experts."
    max_id = -1
    for k in weight_map.keys():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        parts = rest.split(".", 1)
        if not parts:
            continue
        try:
            eid = int(parts[0])
        except ValueError:
            continue
        if eid > max_id:
            max_id = eid
    if max_id < 0:
        raise SystemExit(f"ERROR: could not infer num_experts from index for layer {layer}")
    return max_id + 1

def orient_linear_weight(w: np.ndarray, in_features: int, out_features: int, name: str) -> np.ndarray:
    if w.ndim != 2:
        raise SystemExit(f"ERROR: '{name}' expected 2D weight, got shape={list(w.shape)}")
    r, c = int(w.shape[0]), int(w.shape[1])
    if r == out_features and c == in_features:
        return w
    if r == in_features and c == out_features:
        return w.T
    raise SystemExit(
        f"ERROR: '{name}' shape={list(w.shape)} does not match "
        f"either [{out_features},{in_features}] or [{in_features},{out_features}]"
    )

def quantize_q4_0_blocks(mat_f32: np.ndarray, block_k: int = BLOCK_K):
    mat = np.asarray(mat_f32, dtype=np.float32)
    if mat.ndim != 2:
        raise SystemExit(f"ERROR: quantize expects 2D, got {mat.ndim}D")
    rows, cols = int(mat.shape[0]), int(mat.shape[1])
    if cols % block_k != 0:
        raise SystemExit(f"ERROR: cols={cols} not divisible by block_k={block_k}")
    nb = cols // block_k

    scales_u16 = np.empty((rows, nb), dtype=np.uint16)
    blocks_u8 = np.empty((rows, nb, block_k // 2), dtype=np.uint8)  # 16 bytes per 32 cols

    for r in range(rows):
        row = mat[r]
        for b in range(nb):
            x = row[b * block_k : (b + 1) * block_k]
            absmax = float(np.max(np.abs(x))) if x.size else 0.0
            if absmax == 0.0:
                scale = np.float32(1.0)
                q = np.zeros((block_k,), dtype=np.int32)
            else:
                scale = np.float32(absmax / 7.0)
                q = np.rint(x / scale).astype(np.int32)
                q = np.clip(q, -7, 7)

            scales_u16[r, b] = f32_to_bf16_u16(scale)[()]
