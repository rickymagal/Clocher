#!/usr/bin/env python3
"""
Pack a Hugging Face checkpoint directory (PyTorch shards) into the
Inference Engine container: model.ie.json + model.ie.bin.

Supports mixed precision:
  - FP32 (default): tensors are streamed and stored as row-major FP32.
  - INT4 weight-only: tensors are copied from pre-packed INT4 artifacts and
    annotated with a `quant` object in JSON.

Inputs
------
Required:
  --hf-dir     : HF directory with config.json + pytorch_model-*-of-*.bin
  --out-dir    : Output directory for model.ie.json / model.ie.bin

Optional (INT4 integration):
  --q4-map     : Path to a manifest that lists INT4 artifacts per tensor.
                 Accepted formats:
                   * JSON object: { "<name>": {...}, ... }
                   * JSON list  : [ {"name": "...", ...}, ... ]
                   * JSONL      : one JSON object per line
                   * Directory  : read all *.json files and merge
                 Each item must provide:
                   - "name": state_dict key (exact)
                   - "int4_bin" or "packed_bin": path to packed INT4 bytes
                   - "scale_bin": path to per-row or per-tensor scales
                   - "per": "row" | "tensor"
                   - "rows", "cols": stored shape of the INT4 matrix
                 Optional fields (defaults shown):
                   - "scale_dtype": "fp16"
                   - "pack": "nibble_lohi"
                   - "zp": 0
                   - "symmetric": true

  --align      : Byte alignment for each tensor in the aggregate bin
                 (default: 64). Use 1 to disable alignment.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

def log(msg: str) -> None:
    print(f"[hf_to_iebin] {msg}", file=sys.stderr)

def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def find_shards(hf_dir: Path) -> List[Path]:
    shards = sorted(hf_dir.glob("pytorch_model-*-of-*.bin"))
    if not shards:
        die("no PyTorch shards found (*.bin).")
    return shards

def load_state_keys_from_shard(shard_path: Path) -> Dict[str, Any]:
    import torch
    log(f"Indexing keys in shard: {shard_path.name}")
    try:
        sd = torch.load(shard_path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        sd = torch.load(shard_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise RuntimeError(f"Unexpected shard object type: {type(sd)}")
    return sd

def pass1_index(hf_dir: Path) -> Tuple[List[Path], List[str], Dict[str, Tuple[Tuple[int, ...], str, int]]]:
    """Map key -> (shape, dtype_str, shard_idx) without holding tensors in memory."""
    shards = find_shards(hf_dir)
    key_meta: Dict[str, Tuple[Tuple[int, ...], str, int]] = {}
    order: List[str] = []
    for idx, sh in enumerate(shards):
        sd = load_state_keys_from_shard(sh)
        for k, v in sd.items():
            if k in key_meta:
                continue
            if not hasattr(v, "shape"):
                continue
            shape = tuple(int(x) for x in v.shape)
            dtype = str(getattr(v, "dtype", "float32")).replace("torch.", "")
            key_meta[k] = (shape, dtype, idx)
            order.append(k)
        del sd
        gc.collect()
    return shards, order, key_meta

def _read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    items = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def load_q4_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load an INT4 manifest into a dict keyed by tensor name.

    Canonical item keys:
      name (required)
      int4_bin / packed_bin (required) -> "int4_bin"
      scale_bin (required)
      per ("row"|"tensor", required)
      rows (required), cols (required)
      scale_dtype (optional, default "fp16")
      pack (optional, default "nibble_lohi")
      zp (optional, default 0)
      symmetric (optional, default True)
    """
    if not path.exists():
        die(f"--q4-map not found: {path}")

    items: List[Dict[str, Any]] = []
    if path.is_dir():
        for jp in sorted(path.glob("*.json")):
            try:
                obj = _read_json(jp)
                if isinstance(obj, list):
                    items.extend(obj)
                else:
                    items.append(obj)
            except Exception as e:
                die(f"failed to read {jp}: {e}")
    else:
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
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
                                spec = dict(spec)
                                spec["name"] = name
                                items.append(spec)
                else:
                    die(f"--q4-map has unsupported top-level type: {type(obj)}")
        except Exception as e:
            die(f"failed to parse --q4-map: {e}")

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

        # Normalize paths
        spec["int4_bin"] = str(Path(spec["int4_bin"]).resolve())
        spec["scale_bin"] = str(Path(spec["scale_bin"]).resolve())
        q4[name] = spec

    log(f"Loaded INT4 manifest entries: {len(q4)}")
    return q4

def align_up(x: int, a: int) -> int:
    if a <= 1:
        return x
    return (x + (a - 1)) & ~(a - 1)

def compute_layout(
    order: List[str],
    key_meta: Dict[str, Tuple[Tuple[int, ...], str, int]],
    q4_map: Dict[str, Dict[str, Any]] | None,
    align: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Compute per-tensor layout (dtype, offset, nbytes) for the aggregate bin."""
    tensors: List[Dict[str, Any]] = []
    offset = 0
    for k in order:
        shape, _dtype, _si = key_meta[k]

        if q4_map and k in q4_map:
            spec = q4_map[k]
            rows = int(spec["rows"])
            cols = int(spec["cols"])
            nbytes = rows * ((cols + 1) // 2)  # packed bytes: 2 values per byte
            offset = align_up(offset, align)
            tensors.append({
                "name": k,
                "shape": [rows, cols],
                "dtype": "int4",
                "offset": offset,
                "nbytes": nbytes,
                "quant": {
                    "per": str(spec["per"]),
                    "scale_bin": spec["scale_bin"],
                    "scale_dtype": str(spec.get("scale_dtype", "fp16")),
                    "pack": str(spec.get("pack", "nibble_lohi")),
                    "zp": int(spec.get("zp", 0)),
                    "symmetric": bool(spec.get("symmetric", True))
                }
            })
            offset += nbytes
        else:
            n_elem = 1
            for d in shape:
                n_elem *= int(d)
            nbytes = n_elem * 4  # float32
            offset = align_up(offset, align)
            tensors.append({
                "name": k,
                "shape": list(shape),
                "dtype": "fp32",
                "offset": offset,
                "nbytes": nbytes
            })
            offset += nbytes

    return tensors, offset

def stream_write_blob(
    out_bin: Path,
    shards: List[Path],
    order: List[str],
    key_meta: Dict[str, Tuple[Tuple[int, ...], str, int]],
    tensors: List[Dict[str, Any]],
    q4_map: Dict[str, Dict[str, Any]] | None,
    total_bytes: int,
) -> None:
    """Create and fill model.ie.bin with FP32 and INT4 data in a single pass."""
    import torch
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    log(f"Writing model.ie.bin (~{total_bytes / (1024**3):.2f} GiB)")
    with open(out_bin, "wb") as fout:
        # Pre-size (best effort)
        try:
            if total_bytes > 0:
                fout.seek(total_bytes - 1)
                fout.write(b"\0")
                fout.flush()
        except Exception:
            pass
        fout.seek(0)

        # Build shard -> keys
        shard2keys: Dict[int, List[str]] = {}
        for k in order:
            _shape, _dtype, si = key_meta[k]
            shard2keys.setdefault(si, []).append(k)

        # Index tensor meta by name
        meta_by_name = {t["name"]: t for t in tensors}

        # First, write all INT4 tensors by direct copy (independent of shards)
        if q4_map:
            for name, spec in q4_map.items():
                if name not in meta_by_name:
                    log(f"INT4 spec name not found among HF keys, skipping: {name}")
                    continue
                tmeta = meta_by_name[name]
                if tmeta.get("dtype") != "int4":
                    continue
                srcp = Path(spec["int4_bin"])
                if not srcp.is_file():
                    die(f"INT4 bytes file not found: {srcp}")
                expected = int(tmeta["nbytes"])
                with open(srcp, "rb") as fsrc:
                    data = fsrc.read()
                if len(data) != expected:
                    die(f"INT4 bytes length mismatch for '{name}': got {len(data)}, want {expected}")
                fout.seek(int(tmeta["offset"]))
                fout.write(data)

        # Then, stream remaining FP32 tensors grouped by shard
        for si, sh in enumerate(shards):
            sd = load_state_keys_from_shard(sh)
            for k in shard2keys.get(si, []):
                tmeta = meta_by_name[k]
                if tmeta.get("dtype") == "int4":
                    continue
                fout.seek(int(tmeta["offset"]))
                t = sd[k]
                ten = t.detach().to(dtype=torch.float32, device="cpu").contiguous()
                fout.write(ten.numpy().tobytes(order="C"))
                del t, ten
            del sd
            gc.collect()

def discover_tokenizer(hf_dir: Path) -> Dict[str, Any]:
    tok_vocab  = (hf_dir / "vocab.json");             tok_vocab = str(tok_vocab.resolve()) if tok_vocab.exists() else None
    tok_merges = (hf_dir / "merges.txt");             tok_merges = str(tok_merges.resolve()) if tok_merges.exists() else None
    tok_json   = (hf_dir / "tokenizer.json");         tok_json  = str(tok_json.resolve())  if tok_json.exists()  else None
    tok_cfg    = (hf_dir / "tokenizer_config.json");  tok_cfg   = str(tok_cfg.resolve())   if tok_cfg.exists()   else None
    return {
        "vocab": tok_vocab,
        "merges": tok_merges,
        "tokenizer_json": tok_json,
        "tokenizer_config": tok_cfg,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="HF checkpoint dir (config.json + pytorch_model-*-of-*.bin)")
    ap.add_argument("--out-dir", required=True, help="Output dir for model.ie.json/bin")
    ap.add_argument("--q4-map", default=None, help="INT4 manifest (json/jsonl or directory with *.json)")
    ap.add_argument("--align", type=int, default=64, help="Alignment in bytes for each tensor (default: 64)")
    args = ap.parse_args()

    hf_dir  = Path(args.hf_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not (hf_dir / "config.json").exists():
        die(f"{hf_dir}/config.json not found.")

    # Index HF tensors
    shards, order, key_meta = pass1_index(hf_dir)
    if not order:
        die("no tensor keys found in shards.")

    # Load optional INT4 manifest
    q4_map = load_q4_map(Path(args.q4_map)) if args.q4_map else None

    # Compute aggregate layout
    tensors, total_bytes = compute_layout(order, key_meta, q4_map, args.align)

    # Write combined blob
    bin_path = out_dir / "model.ie.bin"
    json_path = out_dir / "model.ie.json"
    stream_write_blob(bin_path, shards, order, key_meta, tensors, q4_map, total_bytes)

    # Root dtype: "mixed" if any INT4, else "fp32"
    root_dtype = "mixed" if any(t.get("dtype") == "int4" for t in tensors) else "fp32"

    meta = {
        "format": "iebin.v1",
        "version": 1,
        "bin": str(bin_path.name),     # keep relative to JSON for portability
        "dtype": root_dtype,
        "tensors": tensors,
        "tokenizer": discover_tokenizer(hf_dir),
        "source": {
            "hf_dir": str(hf_dir),
            "commit": None
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"Wrote {json_path} and {bin_path}")

if __name__ == "__main__":
    main()
