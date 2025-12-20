# tools/dedup_prepare_and_extract_all.py
import json, subprocess, sys
from pathlib import Path
from typing import Dict, List, Tuple

MD = Path("models/gpt-oss-20b")
IE_JSON = MD / "model.ie.json"
MANIFEST = MD / "model.dedup.json"
OUT_DIR = MD / "dedup_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TENSOR_MAP_OUT = OUT_DIR / "tensor_map.for_dedup.json"
GROUPS_OUT = OUT_DIR / "model.dedup.groups.for_dedup.json"
OUT_PREFIX = OUT_DIR / "model"

DTYPE_SIZES = {
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.int32": 4,
    "torch.int64": 8,
}

def load_ie() -> Tuple[str, List[dict], Dict[str, int]]:
    ie = json.loads(IE_JSON.read_text())
    weights_bin = ie.get("weights_bin")
    tensors = ie.get("tensors")
    if not isinstance(weights_bin, str) or not isinstance(tensors, list):
        raise SystemExit("model.ie.json missing weights_bin or tensors[]")
    name_to_base = {t["name"]: i for i, t in enumerate(tensors) if isinstance(t, dict) and "name" in t}
    return weights_bin, tensors, name_to_base

def dtype_size(dtype: str) -> int:
    s = DTYPE_SIZES.get(dtype)
    if s is None:
        raise SystemExit(f"Unsupported dtype in IE: {dtype}")
    return s

def per_expert_slices(base: dict, expert_count: int) -> int:
    # base["shape"] is expected like [32, ..., ..., ...] where dim0 is experts
    shape = base["shape"]
    if not isinstance(shape, list) or len(shape) < 1:
        return 0
    return expert_count if shape[0] == expert_count else 0

def make_virtual_slices(base: dict, expert_count: int) -> List[dict]:
    # Returns a list of virtual tensor entries (index/name/dtype/shape/offset/nbytes) for each expert slice.
    # We slice along dim0 (experts).
    shape = list(base["shape"])
    dtype = base["dtype"]
    base_offset = int(base["offset"])
    total_bytes = int(base["nbytes"])
    esize = dtype_size(dtype)
    # Elements per expert = total_elements / expert_count
    total_elems = 1
    for d in shape:
        total_elems *= int(d)
    if total_elems * esize != total_bytes:
        raise SystemExit(f"IE tensor size mismatch for {base.get('name')}")

    if shape[0] != expert_count:
        # Not an expert-fused tensor; no slices.
        return []

    elems_per_expert = total_elems // expert_count
    bytes_per_expert = elems_per_expert * esize

    out = []
    # Slice shape drops the first dim
    slice_shape = shape[1:]
    for e in range(expert_count):
        out.append({
            "dtype": dtype,
            "shape": slice_shape,
            "offset": base_offset + e * bytes_per_expert,
            "nbytes": bytes_per_expert,
            # Name is filled by caller for context; index assigned later.
        })
    return out

def main() -> int:
    # 1) Load IE and manifest
    weights_bin, base_tensors, name_to_base = load_ie()
    manifest = json.loads(MANIFEST.read_text())
    groups = manifest["groups"] if isinstance(manifest, dict) else manifest
    if not isinstance(groups, list):
        raise SystemExit("model.dedup.json: expected groups list")

    # 2) Build an extended tensor list for dedup extractor:
    #    - All original tensors with explicit indices
    #    - Dynamic virtual slices for expert fused tensors when needed
    compat_raw: List[dict] = []
    compat_name_to_idx: Dict[str, int] = {}

    # Start with base tensors
    for i, t in enumerate(base_tensors):
        compat_raw.append({
            "index": i,
            "name": t["name"],
            "dtype": t["dtype"],
            "shape": t["shape"],
            "offset": t["offset"],
            "nbytes": t["nbytes"],
        })
        compat_name_to_idx[t["name"]] = i

    next_index = len(compat_raw)

    # 3) Prepare groups (kv_pair + expert_pair)
    out_groups: List[dict] = []
    missing = 0
    converted_kv = 0
    converted_expert = 0

    # Utility to register a virtual slice tensor by name
    def ensure_virtual_slice(base_name: str, suffix: str, expert_idx: int, expert_count: int) -> int:
        nonlocal next_index
        base_idx = name_to_base.get(base_name)
        if base_idx is None:
            return -1
        base = base_tensors[base_idx]
        vslices = make_virtual_slices(base, expert_count)
        if not vslices:
            return -1
        # Assign a stable name for this slice
        virt_name = f"{base_name}#slice{expert_idx}"
        if virt_name in compat_name_to_idx:
            return compat_name_to_idx[virt_name]
        v = vslices[expert_idx]
        entry = {
            "index": next_index,
            "name": virt_name,
            "dtype": v["dtype"],
            "shape": v["shape"],
            "offset": v["offset"],
            "nbytes": v["nbytes"],
        }
        compat_raw.append(entry)
        compat_name_to_idx[virt_name] = next_index
        next_index += 1
        return entry["index"]

    # Walk manifest groups
    for g in groups:
        kind = g.get("kind")
        if kind == "kv_pair":
            members = g.get("members")
            if not (isinstance(members, list) and len(members) >= 2):
                continue
            names = [m for m in members if isinstance(m, str)]
            if len(names) < 2:
                continue
            # Simple: default is first, members are the rest
            try:
                default_index = compat_name_to_idx[names[0]]
                member_indices = [compat_name_to_idx[n] for n in names[1:]]
            except KeyError:
                missing += 1
                continue
            out_groups.append({"default_index": default_index, "member_indices": member_indices})
            converted_kv += 1

        elif kind == "expert_pair":
            # Example fields:
            #   "fused_tensor": "model.layers.0.mlp.experts.gate_up_proj_scales"
            #   "expert_indices": [0, 1]
            #   "component": "gate_up_proj_scales" (or blocks/bias)
            fused_name = g.get("fused_tensor")
            expert_indices = g.get("expert_indices")
            if not (isinstance(fused_name, str) and isinstance(expert_indices, list) and len(expert_indices) >= 2):
                continue
            # We assume 32 experts; if different, infer from IE shape.
            base_idx = name_to_base.get(fused_name)
            if base_idx is None:
                missing += 1
                continue
            base = base_tensors[base_idx]
            expert_count = base["shape"][0]
            # Ensure virtual slices for each expert index
            slice_indices = []
            okay = True
            for eidx in expert_indices:
                if not isinstance(eidx, int) or not (0 <= eidx < expert_count):
                    okay = False
                    break
                virt_idx = ensure_virtual_slice(fused_name, "#slice", eidx, expert_count)
                if virt_idx < 0:
                    okay = False
                    break
                slice_indices.append(virt_idx)
            if not okay or len(slice_indices) < 2:
                missing += 1
                continue
            # default is first expert slice; members are the remainder
            default_index = slice_indices[0]
            member_indices = slice_indices[1:]
            out_groups.append({"default_index": default_index, "member_indices": member_indices})
            converted_expert += 1

        else:
            # Unknown group kind; skip
            continue

    # 4) Write tensor_map and groups for extractor
    tensor_map = {
        "weights_bin": str(Path(weights_bin).name),
        "tensors": compat_raw,  # entries have explicit "index"
    }
    TENSOR_MAP_OUT.write_text(json.dumps(tensor_map, indent=2))
    GROUPS_OUT.write_text(json.dumps(out_groups, indent=2))

    print(f"[dedup] weights_bin={Path(weights_bin).name}")
    print(f"[dedup] ie_tensors={len(base_tensors)} base_indices={len(compat_raw) - (next_index - len(compat_raw))}")
    print(f"[dedup] fused_needed={(next_index - len(base_tensors))} sliced_ok={(next_index - len(base_tensors))} sliced_fail=0")
    print(f"[dedup] groups_in_manifest={len(groups)} converted_expert={converted_expert} converted_kv={converted_kv} skipped=0 missing={missing}")
    print(f"[dedup] wrote tensor_map: {TENSOR_MAP_OUT}")
    print(f"[dedup] wrote groups:    {GROUPS_OUT}")

    # 5) Call your extractor to emit defaults/masks/exceptions
    cmd = [
        sys.executable,
        "tools/dedup_extract_int4.py",
        "--model-dir", str(MD),
        "--tensor-map", str(TENSOR_MAP_OUT),
        "--groups", str(GROUPS_OUT),
        "--out-prefix", str(OUT_PREFIX),
    ]
    print("[dedup] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[dedup] done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
