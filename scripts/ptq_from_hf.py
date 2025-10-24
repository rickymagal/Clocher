#!/usr/bin/env python3
"""
Hugging Face one-shot PTQ:
- Downloads a model repo snapshot via huggingface_hub
- Locates a checkpoint (safetensors or PyTorch .bin/.pth/.ckpt)
- Loads state_dict, extracts the EXACT tensor by key
- (Optional) transposes 2D
- Exports row-major FP32 .bin
- Runs INT8 PTQ with true rows/cols

Example:
  make deps-ptq
  make ptq-from-hf HF_MODEL=facebook/opt-125m \
                   KEY=model.decoder.layers.0.self_attn.q_proj.weight \
                   OUT_PREFIX=out/qproj_int8
"""
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

def find_checkpoint(root: Path, prefer_file: str | None) -> Path:
    # If user specified a filename, use it.
    if prefer_file:
        p = root / prefer_file
        if p.exists():
            return p
        print(f"ERROR: --file '{prefer_file}' not found in snapshot", file=sys.stderr)
        sys.exit(3)
    # Otherwise, try common names, preferring safetensors
    candidates = []
    for pat in ("*.safetensors", "*pytorch_model*.bin", "*.bin", "*.pt", "*.pth", "*.ckpt"):
        candidates += list(root.glob(pat))
    if not candidates:
        print("ERROR: no checkpoint files found in repo snapshot", file=sys.stderr)
        sys.exit(3)
    # Prefer safetensors first
    candidates.sort(key=lambda p: (0 if p.suffix == ".safetensors" else 1, len(p.name)))
    return candidates[0]

def load_state_dict(ckpt_path: Path) -> dict:
    # Try safetensors first
    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as safe_load
            return safe_load(str(ckpt_path))
        except Exception as e:
            print(f"ERROR: failed to load safetensors: {e}", file=sys.stderr)
            sys.exit(2)
    # Fallback to torch
    try:
        import torch
    except Exception:
        print("ERROR: torch is required. pip install torch", file=sys.stderr)
        sys.exit(2)
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    print("ERROR: unsupported checkpoint format", file=sys.stderr)
    sys.exit(2)

def tensor_to_fp32_bin(t, out_bin: Path, transpose: bool) -> tuple[int, int]:
    import numpy as np
    # Convert -> float32
    if hasattr(t, "detach"):
        t = t.detach().cpu()
        arr = t.to(dtype=t.dtype if str(t.dtype).startswith("float") else None).float().numpy()
    else:
        arr = t
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

    if arr.ndim != 2:
        print(f"ERROR: selected tensor is {arr.ndim}D; expected 2D [rows, cols]", file=sys.stderr)
        sys.exit(3)

    if transpose:
        arr = arr.T.copy()

    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(arr.tobytes(order="C"))
    return rows, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face repo id (e.g., org/name)")
    ap.add_argument("--revision", default=None, help="Branch/tag/commit")
    ap.add_argument("--file", default=None, help="Specific checkpoint file inside repo (optional)")
    ap.add_argument("--key", required=True, help="state_dict key to extract")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for PTQ artifacts")
    ap.add_argument("--transpose", action="store_true", help="Transpose 2D tensor before saving")
    ap.add_argument("--mode", default="per_row", choices=["per_row", "per_tensor"])
    ap.add_argument("--accuracy-threshold", type=float, default=0.995)
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("ERROR: huggingface_hub is required. pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)

    # 1) Download snapshot
    local_dir = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        ignore_patterns=["*.md", "*.txt", "*.json", "*.png", "*.jpg", "*.jpeg", "*.gif"],
    )
    root = Path(local_dir)

    # 2) Locate checkpoint file
    ckpt = find_checkpoint(root, args.file)

    # 3) Load state_dict
    sd = load_state_dict(ckpt)
    if args.key not in sd:
        print(f"ERROR: key '{args.key}' not found in state_dict", file=sys.stderr)
        # Offer a quick listing
        keys_sample = list(sd.keys())[:50]
        print(f"Available keys (first 50): {keys_sample}", file=sys.stderr)
        sys.exit(3)

    t = sd[args.key]

    # 4) Export to FP32 row-major .bin and discover true rows/cols
    out_bin = Path(args.out_prefix + ".f32.bin")
    rows, cols = tensor_to_fp32_bin(t, out_bin, args.transpose)

    meta = {
        "repo": args.repo,
        "revision": args.revision,
        "checkpoint_file": str(ckpt.relative_to(root)),
        "key": args.key,
        "shape": [rows, cols],
        "exported_fp32_bin": str(out_bin),
        "mode": args.mode,
        "out_prefix": args.out_prefix,
        "accuracy_threshold": args.accuracy_threshold,
    }
    print("[hf-shape]", json.dumps(meta))

    # 5) Run INT8 PTQ with the TRUE rows/cols
    cmd = [
        sys.executable, "benchmarks/ptq_calib.py",
        "--weights", str(out_bin),
        "--rows", str(rows),
        "--cols", str(cols),
        "--mode", args.mode,
        "--out-prefix", args.out_prefix,
        "--accuracy-threshold", str(args.accuracy_threshold),
    ]
    cp = subprocess.run(cmd)
    sys.exit(cp.returncode)

if __name__ == "__main__":
    main()
