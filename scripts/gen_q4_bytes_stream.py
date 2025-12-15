#!/usr/bin/env python3
import argparse, re, glob, json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file

INC_W = re.compile(r".*\.weight$")

EXC = [re.compile(x) for x in [
    r".*\.bias$",
    r".*(embed|embedding).*",
    r".*(layernorm|layer_norm|rms_norm|ln).*",
    r".*(norm.*weight).*",
    r".*(lm_head|final_linear)\.weight$",
    r".*embed_out\.weight$",
]]

def wanted(k: str) -> bool:
    if not INC_W.match(k):
        return False
    for rx in EXC:
        if rx.match(k):
            return False
    return True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--q4-bytes", required=True)
    p.add_argument("--scales", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--chunk-rows", type=int, default=128)
    return p.parse_args()

def _pack_int4_from_int8(q_int8: np.ndarray) -> bytes:
    rows, cols = q_int8.shape
    if cols % 2 == 1:
        q_int8 = np.pad(q_int8, ((0, 0), (0, 1)), mode="constant")
        cols += 1
    q_u = q_int8.astype(np.int16)
    lo = (q_u[:, 0::2] & 0xF).astype(np.uint8)
    hi = (q_u[:, 1::2] & 0xF).astype(np.uint8)
    return ((hi << 4) | lo).tobytes()

def _iter_shards(hf_dir: Path) -> list[str]:
    pt = sorted(glob.glob(str(hf_dir / "pytorch_model-*.bin")))
    st_sharded = sorted(glob.glob(str(hf_dir / "model-*-of-*.safetensors")))
    st_single = [str(hf_dir / "model.safetensors")] if (hf_dir / "model.safetensors").exists() else []
    return pt or st_sharded or st_single

def _load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        return safe_load_file(path, device="cpu")
    return torch.load(path, map_location="cpu")

def main():
    args = parse_args()
    hf_dir = Path(args.hf_dir)
    q4_path = Path(args.q4_bytes).resolve()
    sc_path = Path(args.scales).resolve()
    man_path = Path(args.manifest)

    man_path.parent.mkdir(parents=True, exist_ok=True)
    q4_path.parent.mkdir(parents=True, exist_ok=True)

    shards = _iter_shards(hf_dir)
    if not shards:
        raise SystemExit(f"No shards in {hf_dir}")

    if q4_path.exists():
        q4_path.unlink()
    if sc_path.exists():
        sc_path.unlink()

    fq = open(q4_path, "ab", buffering=1024 * 1024)
    fs = open(sc_path, "ab", buffering=1024 * 1024)

    manifest = {}
    bytes_q = 0
    bytes_s = 0
    total_keys = 0
    total_rows = 0

    for shard in shards:
        sd = _load_state_dict(shard)

        for k, W in sd.items():
            if not wanted(k):
                continue
            if not hasattr(W, "dim") or W.dim() != 2:
                continue
            if W.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                continue

            rows, cols = int(W.shape[0]), int(W.shape[1])
            q4_nbytes = rows * ((cols + 1) // 2)   # int4 packed: 2 weights per byte
            scale_nbytes = rows * 2               # fp16 per-row scales

            manifest[k] = {
                "q4_offset": bytes_q,
                "q4_nbytes": q4_nbytes,
                "scale_offset": bytes_s,
                "scale_nbytes": scale_nbytes,
                "dtype": "int4",
                "scheme": "weight_only",
                "per": "row",
                "bits": 4,
                "scale_dtype": "fp16",
                "rows": rows,
                "cols": cols,
                "packed_bin": str(q4_path),
                "scale_bin": str(sc_path),
            }

            step = max(1, args.chunk_rows)
            for start in range(0, rows, step):
                end = min(start + step, rows)
                chunk = W[start:end, :].to(dtype=torch.float32, device="cpu")

                maxabs = torch.amax(torch.abs(chunk), dim=1).clamp_min(1e-12)
                scales = (maxabs / 7.0).to(dtype=torch.float16).cpu().numpy()
                fs.write(scales.tobytes())
                bytes_s += scales.nbytes

                inv = (1.0 / (scales.astype(np.float32) + 1e-12)).reshape(-1, 1)
                q = np.rint(chunk.cpu().numpy() * inv).astype(np.int32)
                q = np.clip(q, -8, 7).astype(np.int8)

                packed = _pack_int4_from_int8(q)
                fq.write(packed)
                bytes_q += len(packed)

            if bytes_q - manifest[k]["q4_offset"] != q4_nbytes:
                raise SystemExit(f"q4 size mismatch for {k}: wrote {bytes_q - manifest[k]['q4_offset']}, expected {q4_nbytes}")
            if bytes_s - manifest[k]["scale_offset"] != scale_nbytes:
                raise SystemExit(f"scale size mismatch for {k}: wrote {bytes_s - manifest[k]['scale_offset']}, expected {scale_nbytes}")

            total_keys += 1
            total_rows += rows

        del sd

    fq.close()
    fs.close()

    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] shards={len(shards)} keys={total_keys} rows={total_rows}")
    print(f"[ok] q4={bytes_q/1024/1024:.2f} MiB scales={bytes_s/1024/1024:.2f} MiB")
    print(f"[ok] manifest -> {man_path.resolve()}")
    print(f"[ok] q4 bytes  -> {q4_path}")
    print(f"[ok] scales    -> {sc_path}")

if __name__ == "__main__":
    main()
