#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np

ALIGN = 64
BLOCK_K = 32


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
        self._shard_cache = {}

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
            raise SystemExit(f"ERROR: unsupported dtype '{dtype}' for '{tensor_name}'")

        if np.prod(shape, dtype=np.int64) * 4 != f32.nbytes:
            raise SystemExit(f"ERROR: size mismatch for '{tensor_name}': shape={shape} dtype={dtype} bytes={nbytes}")

        return f32.reshape(shape)


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
        f"[{out_features},{in_features}] or [{in_features},{out_features}]"
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
    blocks_u8 = np.empty((rows, nb, block_k // 2), dtype=np.uint8)

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
            q_u = (q + 8).astype(np.uint8)
            lo = q_u[0::2]
            hi = q_u[1::2] << 4
            blocks_u8[r, b, :] = (lo | hi)

    return blocks_u8, scales_u16


def infer_layers(weight_map: dict) -> int:
    max_layer = -1
    prefix = "model.layers."
    for k in weight_map.keys():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        parts = rest.split(".", 1)
        if not parts:
            continue
        try:
            lid = int(parts[0])
        except ValueError:
            continue
        if lid > max_layer:
            max_layer = lid
    if max_layer < 0:
        raise SystemExit("ERROR: could not infer layer count from safetensors index")
    return max_layer + 1


def find_d_model(compat_json: dict, hf: HFIndex) -> int:
    for name in ("model.embed_tokens.weight", "tok_embeddings.weight", "transformer.wte.weight"):
        try:
            desc = next(t for t in compat_json.get("tensors", []) if t.get("name") == name)
        except StopIteration:
            continue
        shape = desc.get("shape")
        if shape and len(shape) == 2:
            return int(shape[1])
    # fallback: try layer0 q_proj weight
    w = hf.read_tensor_f32("model.layers.0.self_attn.q_proj.weight")
    r, c = int(w.shape[0]), int(w.shape[1])
    if r == c:
        return r
    raise SystemExit("ERROR: cannot infer d_model from compat json or q_proj")


def infer_attn_dims(hf: HFIndex, d_model: int) -> tuple[int, int]:
    wq = hf.read_tensor_f32("model.layers.0.self_attn.q_proj.weight")
    wk = hf.read_tensor_f32("model.layers.0.self_attn.k_proj.weight")
    wv = hf.read_tensor_f32("model.layers.0.self_attn.v_proj.weight")
    wq = np.asarray(wq)
    wk = np.asarray(wk)
    wv = np.asarray(wv)
    r_q, c_q = int(wq.shape[0]), int(wq.shape[1])
    r_k, c_k = int(wk.shape[0]), int(wk.shape[1])
    r_v, c_v = int(wv.shape[0]), int(wv.shape[1])

    if c_q == d_model:
        q_dim = r_q
    elif r_q == d_model:
        q_dim = c_q
    else:
        raise SystemExit(f"ERROR: q_proj shape {list(wq.shape)} does not match d_model={d_model}")

    if c_k == d_model:
        kv_dim = r_k
    elif r_k == d_model:
        kv_dim = c_k
    else:
        raise SystemExit(f"ERROR: k_proj shape {list(wk.shape)} does not match d_model={d_model}")

    if (c_v == d_model and r_v == kv_dim) or (r_v == d_model and c_v == kv_dim):
        return q_dim, kv_dim

    raise SystemExit(
        f"ERROR: v_proj shape {list(wv.shape)} does not match d_model={d_model} kv_dim={kv_dim}"
    )


def append_tensor(bin_f, data: bytes, align: int) -> tuple[int, int]:
    off = bin_f.tell()
    aligned = align_up(off, align)
    if aligned > off:
        bin_f.write(b"\x00" * (aligned - off))
    bin_f.write(data)
    return aligned, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--compat-json", required=True)
    ap.add_argument("--q4-bin", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-bin", default="")
    ap.add_argument("--align", type=int, default=ALIGN)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    compat_path = Path(args.compat_json)
    bin_path = Path(args.q4_bin)

    compat = json.loads(compat_path.read_text(encoding="utf-8"))
    tensors = list(compat.get("tensors", []))
    existing = {t.get("name"): t for t in tensors if t.get("name")}

    hf = HFIndex(hf_dir)
    n_layers = infer_layers(hf.weight_map)
    d_model = find_d_model(compat, hf)
    q_dim, kv_dim = infer_attn_dims(hf, d_model)

    out_bin = Path(args.out_bin) if args.out_bin else bin_path
    out_json = Path(args.out_json) if args.out_json else compat_path

    if out_bin != bin_path:
        shutil.copyfile(bin_path, out_bin)

    with open(out_bin, "r+b") as fbin:
        fbin.seek(0, os.SEEK_END)
        for l in range(n_layers):
            for proj, rows, cols in (
                ("q", q_dim, d_model),
                ("k", kv_dim, d_model),
                ("v", kv_dim, d_model),
                ("o", d_model, q_dim),
            ):
                w_name = f"model.layers.{l}.self_attn.{proj}_proj.weight"
                blocks_name = f"model.layers.{l}.self_attn.{proj}_proj.weight_blocks"
                scales_name = f"model.layers.{l}.self_attn.{proj}_proj.weight_scales"
                if blocks_name in existing or scales_name in existing:
                    raise SystemExit(f"ERROR: q4 tensors already present for {w_name}")
                w = hf.read_tensor_f32(w_name)
                w = orient_linear_weight(w, cols, rows, w_name)

                blocks_u8, scales_u16 = quantize_q4_0_blocks(w, BLOCK_K)
                blocks_bytes = blocks_u8.tobytes(order="C")
                scales_bytes = scales_u16.tobytes(order="C")

                blocks_off, blocks_n = append_tensor(fbin, blocks_bytes, args.align)
                scales_off, scales_n = append_tensor(fbin, scales_bytes, args.align)

                blocks_entry = {
                    "name": blocks_name,
                    "dtype": "U8",
                    "shape": [int(rows), int(cols // BLOCK_K), int(BLOCK_K // 2)],
                    "nbytes": int(blocks_n),
                    "offset": int(blocks_off),
                }
                scales_entry = {
                    "name": scales_name,
                    "dtype": "BF16",
                    "shape": [int(rows), int(cols // BLOCK_K)],
                    "nbytes": int(scales_n),
                    "offset": int(scales_off),
                }
                tensors.append(blocks_entry)
                tensors.append(scales_entry)
                existing[blocks_name] = blocks_entry
                existing[scales_name] = scales_entry

    compat["tensors"] = tensors

    if out_json == compat_path:
        backup = compat_path.with_suffix(compat_path.suffix + ".bak")
        shutil.copyfile(compat_path, backup)

    out_json.write_text(json.dumps(compat, indent=2), encoding="utf-8")

    print(f"[ok] layers={n_layers} d_model={d_model} kv_dim={kv_dim}")
    print(f"[ok] q4 bin -> {out_bin}")
    print(f"[ok] compat -> {out_json}")


if __name__ == "__main__":
    main()
