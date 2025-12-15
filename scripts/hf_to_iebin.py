#!/usr/bin/env python3
"""
Pack a Hugging Face checkpoint directory into the Inference Engine container:
  - model.ie.json
  - model.ie.bin

Supports:
  - PyTorch .bin (pytorch_model.bin or pytorch_model-*-of-*.bin)
  - safetensors (model.safetensors or model-*-of-*.safetensors)

Mixed precision:
  - FP32 (default): tensors stored as row-major fp32.
  - INT4 weight-only: tensors copied from pre-packed INT4 artifacts and
    annotated with a `quant` object in JSON.

Optional lossless dedup (whole-tensor):
  - If enabled, tensors with identical byte content (after fp32 conversion for fp32 tensors)
    are stored once, and subsequent duplicates reuse the same offset/nbytes.

Inputs
------
Required:
  --hf-dir     : HF directory with config.json + weights files
  --out-dir    : Output directory for model.ie.json / model.ie.bin

Optional (INT4 integration):
  --q4-map     : Manifest listing INT4 artifacts per tensor (json/jsonl/dir).
  --align      : Byte alignment for each tensor in the aggregate bin (default: 64).
  --prefer     : Which format to prefer when both exist: "safetensors"|"bin" (default: safetensors).

Optional (dedup):
  --dedup              : Enable whole-tensor byte-identical deduplication.
  --dedup-min-bytes    : Only dedup tensors >= this many bytes (default: 1048576).
"""

import argparse
import gc
import json
import os
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable


def log(msg: str) -> None:
    print(f"[hf_to_iebin] {msg}", file=sys.stderr)


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


@dataclass(frozen=True)
class TensorRef:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    shard_idx: int


@dataclass(frozen=True)
class Shard:
    path: Path
    fmt: str  # "bin" or "safetensors"


def align_up(x: int, a: int) -> int:
    if a <= 1:
        return x
    return (x + (a - 1)) & ~(a - 1)


def _read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def discover_tokenizer(hf_dir: Path) -> Dict[str, Any]:
    def _p(rel: str) -> Optional[str]:
        p = hf_dir / rel
        return str(p.resolve()) if p.exists() else None

    return {
        "vocab": _p("vocab.json"),
        "merges": _p("merges.txt"),
        "tokenizer_json": _p("tokenizer.json"),
        "tokenizer_config": _p("tokenizer_config.json"),
    }


def find_shards(hf_dir: Path, prefer: str) -> List[Shard]:
    if prefer not in ("safetensors", "bin"):
        die("--prefer must be 'safetensors' or 'bin'")

    st_single = sorted(hf_dir.glob("model.safetensors"))
    st_shards = sorted(hf_dir.glob("model-*-of-*.safetensors"))
    bin_single = sorted(hf_dir.glob("pytorch_model.bin"))
    bin_shards = sorted(hf_dir.glob("pytorch_model-*-of-*.bin"))

    st = st_single or st_shards
    bn = bin_single or bin_shards

    if not st and not bn:
        die("no weights found (expected *.safetensors or *.bin).")

    def as_shards(paths: List[Path], fmt: str) -> List[Shard]:
        return [Shard(path=p, fmt=fmt) for p in paths]

    if prefer == "safetensors":
        return as_shards(st, "safetensors") if st else as_shards(bn, "bin")
    return as_shards(bn, "bin") if bn else as_shards(st, "safetensors")


def load_q4_map(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        die(f"--q4-map not found: {path}")

    items: List[Dict[str, Any]] = []
    if path.is_dir():
        for jp in sorted(path.glob("*.json")):
            obj = _read_json(jp)
            if isinstance(obj, list):
                items.extend(obj)
            else:
                items.append(obj)
    else:
        suf = path.suffix.lower()
        if suf == ".jsonl":
            items.extend(_read_jsonl(path))
        else:
            obj = _read_json(path)
            if isinstance(obj, list):
                items.extend(obj)
            elif isinstance(obj, dict):
                if "name" in obj:
                    items.append(obj)
                else:
                    for name, spec in obj.items():
                        if isinstance(spec, dict):
                            d = dict(spec)
                            d["name"] = name
                            items.append(d)
            else:
                die(f"--q4-map has unsupported top-level type: {type(obj)}")

    q4: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict) or "name" not in it:
            die("every INT4 manifest item must be an object with a 'name'")
        name = str(it["name"])
        spec = dict(it)

        if "int4_bin" not in spec and "packed_bin" in spec:
            spec["int4_bin"] = spec.pop("packed_bin")
        if "int4_bin" not in spec:
            die(f"INT4 spec for '{name}' missing 'int4_bin' (or 'packed_bin')")
        if "scale_bin" not in spec:
            die(f"INT4 spec for '{name}' missing 'scale_bin'")
        if "per" not in spec:
            die(f"INT4 spec for '{name}' missing 'per' ('row'|'tensor')")
        if "rows" not in spec or "cols" not in spec:
            die(f"INT4 spec for '{name}' must include 'rows' and 'cols'")

        spec.setdefault("scale_dtype", "fp16")
        spec.setdefault("pack", "nibble_lohi")
        spec.setdefault("zp", 0)
        spec.setdefault("symmetric", True)

        spec["int4_bin"] = str(Path(spec["int4_bin"]).resolve())
        spec["scale_bin"] = str(Path(spec["scale_bin"]).resolve())
        q4[name] = spec

    log(f"Loaded INT4 manifest entries: {len(q4)}")
    return q4


def pass1_index(shards: List[Shard]) -> Tuple[List[str], Dict[str, TensorRef]]:
    order: List[str] = []
    meta: Dict[str, TensorRef] = {}

    for si, sh in enumerate(shards):
        if sh.fmt == "safetensors":
            try:
                from safetensors import safe_open
            except Exception as e:
                die(f"safetensors is required to read {sh.path.name}: {e}")

            log(f"Indexing keys in safetensors shard: {sh.path.name}")
            with safe_open(str(sh.path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k in meta:
                        continue
                    t = f.get_tensor(k)
                    shape = tuple(int(x) for x in t.shape)
                    dtype = str(t.dtype).replace("torch.", "")
                    meta[k] = TensorRef(name=k, shape=shape, dtype=dtype, shard_idx=si)
                    order.append(k)
                    del t
            gc.collect()
        else:
            import torch

            log(f"Indexing keys in bin shard: {sh.path.name}")
            try:
                sd = torch.load(sh.path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
            except TypeError:
                sd = torch.load(sh.path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            if not isinstance(sd, dict):
                die(f"unexpected shard object type: {type(sd)}")

            for k, v in sd.items():
                if k in meta:
                    continue
                if not hasattr(v, "shape"):
                    continue
                shape = tuple(int(x) for x in v.shape)
                dtype = str(getattr(v, "dtype", "float32")).replace("torch.", "")
                meta[k] = TensorRef(name=k, shape=shape, dtype=dtype, shard_idx=si)
                order.append(k)
            del sd
            gc.collect()

    if not order:
        die("no tensor keys found in checkpoint.")
    return order, meta


def _sha256_of_file(path: Path, expected_nbytes: int, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            n += len(b)
            h.update(b)
    if n != expected_nbytes:
        die(f"INT4 bytes length mismatch for '{path.name}': got {n}, want {expected_nbytes}")
    return h.hexdigest()


def _copy_file_into(fout, dst_off: int, path: Path, expected_nbytes: int, chunk_size: int = 8 * 1024 * 1024) -> None:
    fout.seek(dst_off)
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            fout.write(b)
            n += len(b)
    if n != expected_nbytes:
        die(f"INT4 bytes length mismatch while copying '{path.name}': got {n}, want {expected_nbytes}")


def stream_write_blob_dedup(
    out_bin: Path,
    shards: List[Shard],
    order: List[str],
    meta: Dict[str, TensorRef],
    q4_map: Optional[Dict[str, Dict[str, Any]]],
    align: int,
    dedup: bool,
    dedup_min_bytes: int,
) -> List[Dict[str, Any]]:
    import torch

    out_bin.parent.mkdir(parents=True, exist_ok=True)

    # Hash -> (offset, nbytes)
    seen: Dict[Tuple[str, int], Tuple[int, int]] = {}

    # We'll write sequentially and fill tensor metadata as we go.
    tensors: List[Dict[str, Any]] = []

    # Group keys by shard to reduce re-open overhead
    shard2keys: Dict[int, List[str]] = {}
    for k in order:
        shard2keys.setdefault(meta[k].shard_idx, []).append(k)

    cur_off = 0

    def reserve(nbytes: int) -> int:
        nonlocal cur_off
        cur_off = align_up(cur_off, align)
        off = cur_off
        cur_off += nbytes
        return off

    log("Writing model.ie.bin (streaming, with optional dedup)")
    with open(out_bin, "wb") as fout:
        for si, sh in enumerate(shards):
            keys = shard2keys.get(si, [])
            if not keys:
                continue

            if sh.fmt == "safetensors":
                from safetensors import safe_open

                log(f"Streaming tensors from safetensors shard: {sh.path.name}")
                with safe_open(str(sh.path), framework="pt", device="cpu") as f:
                    for k in keys:
                        if q4_map and k in q4_map:
                            spec = q4_map[k]
                            rows = int(spec["rows"])
                            cols = int(spec["cols"])
                            nbytes = rows * ((cols + 1) // 2)

                            srcp = Path(spec["int4_bin"])
                            if not srcp.is_file():
                                die(f"INT4 bytes file not found: {srcp}")

                            do_dedup = dedup and (nbytes >= dedup_min_bytes)
                            if do_dedup:
                                hx = _sha256_of_file(srcp, nbytes)
                                key = (hx, nbytes)
                                if key in seen:
                                    off0, nb0 = seen[key]
                                    tensors.append(
                                        {
                                            "name": k,
                                            "shape": [rows, cols],
                                            "dtype": "int4",
                                            "offset": off0,
                                            "nbytes": nb0,
                                            "quant": {
                                                "per": str(spec["per"]),
                                                "scale_bin": spec["scale_bin"],
                                                "scale_dtype": str(spec.get("scale_dtype", "fp16")),
                                                "pack": str(spec.get("pack", "nibble_lohi")),
                                                "zp": int(spec.get("zp", 0)),
                                                "symmetric": bool(spec.get("symmetric", True)),
                                            },
                                        }
                                    )
                                    continue

                            off = reserve(nbytes)
                            _copy_file_into(fout, off, srcp, nbytes)
                            if do_dedup:
                                hx = _sha256_of_file(srcp, nbytes)
                                seen[(hx, nbytes)] = (off, nbytes)

                            tensors.append(
                                {
                                    "name": k,
                                    "shape": [rows, cols],
                                    "dtype": "int4",
                                    "offset": off,
                                    "nbytes": nbytes,
                                    "quant": {
                                        "per": str(spec["per"]),
                                        "scale_bin": spec["scale_bin"],
                                        "scale_dtype": str(spec.get("scale_dtype", "fp16")),
                                        "pack": str(spec.get("pack", "nibble_lohi")),
                                        "zp": int(spec.get("zp", 0)),
                                        "symmetric": bool(spec.get("symmetric", True)),
                                    },
                                }
                            )
                            continue

                        ten = f.get_tensor(k)
                        ten = ten.detach().to(dtype=torch.float32, device="cpu").contiguous()
                        arr = ten.numpy()
                        mv = memoryview(arr).cast("B")
                        nbytes = mv.nbytes

                        do_dedup = dedup and (nbytes >= dedup_min_bytes)
                        if do_dedup:
                            hx = hashlib.sha256(mv).hexdigest()
                            key = (hx, nbytes)
                            if key in seen:
                                off0, nb0 = seen[key]
                                tensors.append(
                                    {
                                        "name": k,
                                        "shape": list(arr.shape),
                                        "dtype": "fp32",
                                        "offset": off0,
                                        "nbytes": nb0,
                                    }
                                )
                                del ten, arr, mv
                                continue

                        off = reserve(nbytes)
                        fout.seek(off)
                        fout.write(mv)
                        if do_dedup:
                            hx = hashlib.sha256(mv).hexdigest()
                            seen[(hx, nbytes)] = (off, nbytes)

                        tensors.append(
                            {
                                "name": k,
                                "shape": list(arr.shape),
                                "dtype": "fp32",
                                "offset": off,
                                "nbytes": nbytes,
                            }
                        )
                        del ten, arr, mv
                gc.collect()
            else:
                log(f"Streaming tensors from bin shard: {sh.path.name}")
                try:
                    sd = torch.load(sh.path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
                except TypeError:
                    sd = torch.load(sh.path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                if not isinstance(sd, dict):
                    die(f"unexpected shard object type: {type(sd)}")

                for k in keys:
                    if q4_map and k in q4_map:
                        spec = q4_map[k]
                        rows = int(spec["rows"])
                        cols = int(spec["cols"])
                        nbytes = rows * ((cols + 1) // 2)

                        srcp = Path(spec["int4_bin"])
                        if not srcp.is_file():
                            die(f"INT4 bytes file not found: {srcp}")

                        do_dedup = dedup and (nbytes >= dedup_min_bytes)
                        if do_dedup:
                            hx = _sha256_of_file(srcp, nbytes)
                            key = (hx, nbytes)
                            if key in seen:
                                off0, nb0 = seen[key]
                                tensors.append(
                                    {
                                        "name": k,
                                        "shape": [rows, cols],
                                        "dtype": "int4",
                                        "offset": off0,
                                        "nbytes": nb0,
                                        "quant": {
                                            "per": str(spec["per"]),
                                            "scale_bin": spec["scale_bin"],
                                            "scale_dtype": str(spec.get("scale_dtype", "fp16")),
                                            "pack": str(spec.get("pack", "nibble_lohi")),
                                            "zp": int(spec.get("zp", 0)),
                                            "symmetric": bool(spec.get("symmetric", True)),
                                        },
                                    }
                                )
                                continue

                        off = reserve(nbytes)
                        _copy_file_into(fout, off, srcp, nbytes)
                        if do_dedup:
                            hx = _sha256_of_file(srcp, nbytes)
                            seen[(hx, nbytes)] = (off, nbytes)

                        tensors.append(
                            {
                                "name": k,
                                "shape": [rows, cols],
                                "dtype": "int4",
                                "offset": off,
                                "nbytes": nbytes,
                                "quant": {
                                    "per": str(spec["per"]),
                                    "scale_bin": spec["scale_bin"],
                                    "scale_dtype": str(spec.get("scale_dtype", "fp16")),
                                    "pack": str(spec.get("pack", "nibble_lohi")),
                                    "zp": int(spec.get("zp", 0)),
                                    "symmetric": bool(spec.get("symmetric", True)),
                                },
                            }
                        )
                        continue

                    t = sd[k]
                    ten = t.detach().to(dtype=torch.float32, device="cpu").contiguous()
                    arr = ten.numpy()
                    mv = memoryview(arr).cast("B")
                    nbytes = mv.nbytes

                    do_dedup = dedup and (nbytes >= dedup_min_bytes)
                    if do_dedup:
                        hx = hashlib.sha256(mv).hexdigest()
                        key = (hx, nbytes)
                        if key in seen:
                            off0, nb0 = seen[key]
                            tensors.append(
                                {
                                    "name": k,
                                    "shape": list(arr.shape),
                                    "dtype": "fp32",
                                    "offset": off0,
                                    "nbytes": nb0,
                                }
                            )
                            del t, ten, arr, mv
                            continue

                    off = reserve(nbytes)
                    fout.seek(off)
                    fout.write(mv)
                    if do_dedup:
                        hx = hashlib.sha256(mv).hexdigest()
                        seen[(hx, nbytes)] = (off, nbytes)

                    tensors.append(
                        {
                            "name": k,
                            "shape": list(arr.shape),
                            "dtype": "fp32",
                            "offset": off,
                            "nbytes": nbytes,
                        }
                    )
                    del t, ten, arr, mv

                del sd
                gc.collect()

        fout.flush()

    log(f"Final model.ie.bin size: {cur_off / (1024**3):.2f} GiB")
    return tensors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="HF checkpoint dir (config.json + weights)")
    ap.add_argument("--out-dir", required=True, help="Output dir for model.ie.json/bin")
    ap.add_argument("--q4-map", default=None, help="INT4 manifest (json/jsonl or directory with *.json)")
    ap.add_argument("--align", type=int, default=64, help="Alignment in bytes for each tensor (default: 64)")
    ap.add_argument(
        "--prefer",
        type=str,
        default="safetensors",
        choices=["safetensors", "bin"],
        help="Prefer safetensors or bin if both exist (default: safetensors)",
    )
    ap.add_argument("--dedup", action="store_true", help="Enable whole-tensor byte-identical deduplication")
    ap.add_argument(
        "--dedup-min-bytes",
        type=int,
        default=1024 * 1024,
        help="Only dedup tensors >= this many bytes (default: 1048576)",
    )
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not (hf_dir / "config.json").exists():
        die(f"{hf_dir}/config.json not found.")

    shards = find_shards(hf_dir, prefer=args.prefer)
    log(f"Found {len(shards)} shard(s): {shards[0].fmt}")

    q4_map = load_q4_map(Path(args.q4_map)) if args.q4_map else None

    order, meta = pass1_index(shards)

    bin_path = out_dir / "model.ie.bin"
    json_path = out_dir / "model.ie.json"

    tensors = stream_write_blob_dedup(
        bin_path,
        shards,
        order,
        meta,
        q4_map,
        align=args.align,
        dedup=bool(args.dedup),
        dedup_min_bytes=int(args.dedup_min_bytes),
    )

    root_dtype = "mixed" if any(t.get("dtype") == "int4" for t in tensors) else "fp32"

    meta_out = {
        "format": "iebin.v1",
        "version": 1,
        "bin": str(bin_path.name),
        "dtype": root_dtype,
        "tensors": tensors,
        "tokenizer": discover_tokenizer(hf_dir),
        "source": {
            "hf_dir": str(hf_dir),
            "weights": [str(s.path.name) for s in shards],
            "commit": None,
        },
        "dedup": {
            "enabled": bool(args.dedup),
            "kind": "whole_tensor_sha256",
            "min_bytes": int(args.dedup_min_bytes),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    log(f"Wrote {json_path} and {bin_path}")


if __name__ == "__main__":
    main()

