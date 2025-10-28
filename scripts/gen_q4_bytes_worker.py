#!/usr/bin/env python3
# Worker: INT4 weight-only per-row, processa UM shard e faz append nos blobs.
# Saídas:
#  - acrescenta bytes em --q4-bytes e --scales
#  - grava um "manifest part" com rows/cols por key (sem offsets; ordem garante consumo)
import argparse, os, re, json, gc
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)

INC_W = re.compile(r".*\.weight$")
EXC = [re.compile(x) for x in [
    r".*\.bias$", r".*(embed|embedding).*", r".*(layernorm|layer_norm|rms_norm|ln).*",
    r".*(norm.*weight).*", r".*(lm_head|final_linear)\.weight$", r".*embed_out\.weight$",
]]

def wanted(k: str) -> bool:
    if not INC_W.match(k): return False
    for rx in EXC:
        if rx.match(k): return False
    return True

def pack_int4_rows(q_int8: np.ndarray) -> bytes:
    rows, cols = q_int8.shape
    if cols % 2:
        q_int8 = np.pad(q_int8, ((0,0),(0,1)), mode="constant")
        cols += 1
    q16 = q_int8.astype(np.int16)
    lo = (q16[:, 0::2] & 0xF).astype(np.uint8)
    hi = (q16[:, 1::2] & 0xF).astype(np.uint8)
    packed = (hi << 4) | lo
    return packed.tobytes()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--q4-bytes", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--manifest-part", required=True)
    ap.add_argument("--chunk-rows", type=int, default=128)
    args = ap.parse_args()

    q4 = open(args.q4_bytes, "ab", buffering=1024*1024)
    sc = open(args.scales, "ab", buffering=1024*1024)

    part = {}
    rows_total = 0
    bytes_q = 0
    bytes_s = 0

    sd = torch.load(args.shard, map_location="cpu", weights_only=True)
    for k, W in sd.items():
        if not (wanted(k) and isinstance(W, torch.Tensor) and W.ndim == 2 and W.is_floating_point()):
            continue
        W = W.float()
        r, c = W.shape
        part[k] = {
            "dtype": "int4","scheme": "weight_only","per": "row","bits": 4,"pack": "nibble_lohi",
            "scale_dtype": "fp16","zero_point": 0,"symmetric": True,"align": 256,
            "rows": r,"cols": c
        }
        step = max(1, args.chunk_rows)
        with torch.no_grad():
            for s in range(0, r, step):
                e = min(s+step, r)
                chunk = W[s:e, :]
                maxabs = torch.amax(torch.abs(chunk), dim=1).clamp_min(1e-12)
                scales = (maxabs / 7.0).cpu().numpy().astype(np.float16)
                sc.write(scales.tobytes()); bytes_s += scales.nbytes
                inv = (7.0 / maxabs).unsqueeze(1)
                q = torch.round(chunk * inv).clamp_(-8,7).to(torch.int8).cpu().numpy()
                q4.write(pack_int4_rows(q))
                rows_total += (e - s)
                bytes_q += ((q.shape[1] + (q.shape[1] & 1)) // 2) * q.shape[0]
        del W
    del sd
    gc.collect()

    q4.flush(); sc.flush()
    q4.close(); sc.close()

    Path(args.manifest_part).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_part, "w") as f:
        json.dump(part, f)
    print(f"[worker] {os.path.basename(args.shard)} -> rows={rows_total} q4={bytes_q/1024/1024:.2f}MiB scales={bytes_s/1024/1024:.2f}MiB")

if __name__ == "__main__":
    main()
