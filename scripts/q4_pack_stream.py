#!/usr/bin/env python3
import os, re, glob, json, gc
import torch, numpy as np
from pathlib import Path

HF_DIR  = "models/gpt-oss-20b/hf"
Q4_DIR  = "models/gpt-oss-20b/q4"
MAN_OUT = "quant/q4_manifest.expanded.json"

inc_w = re.compile(r'.*\.weight$')
exc_any = [
    re.compile(r'.*\.bias$'),
    re.compile(r'.*(embed|embedding).*'),
    re.compile(r'.*(layernorm|layer_norm|rms_norm|ln).*'),
    re.compile(r'.*(norm.*weight).*'),
    re.compile(r'.*(lm_head|final_linear)\.weight$'),
    re.compile(r'.*embed_out\.weight$'),
]

def wanted(k: str) -> bool:
    if not inc_w.fullmatch(k): return False
    for rx in exc_any:
        if rx.fullmatch(k): return False
    return True

def safe(key: str) -> str:
    return key.replace('.', '_')

def pack_row_int4_sym(row: torch.Tensor):
    a = row.abs().max().item()
    scale = 0.0 if a == 0.0 else a / 7.0
    if scale == 0.0:
        q = np.zeros(row.numel(), dtype=np.int8)
    else:
        q = torch.clamp(torch.round(row / scale), -8, 7).to(torch.int8).cpu().numpy()
    if q.size % 2 == 1:
        q = np.concatenate([q, np.zeros(1, dtype=np.int8)], axis=0)
    n = (q & 0x0F).astype(np.uint8)
    lo = n[0::2]; hi = n[1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed.tobytes(), np.float16(scale).tobytes()

def quantize_2d_per_row(W: torch.Tensor, qpath: Path, spath: Path):
    R, C = W.shape
    with open(qpath, 'wb') as fq, open(spath, 'wb') as fs:
        for r in range(R):
            b, s = pack_row_int4_sym(W[r].contiguous())
            fq.write(b); fs.write(s)

def main():
    os.makedirs(Q4_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MAN_OUT), exist_ok=True)
    manifest = {}

    shards = sorted(glob.glob(os.path.join(HF_DIR, "pytorch_model-*.bin")))
    for sh in shards:
        sd = torch.load(sh, map_location='cpu')  # carrega só este shard
        for k, v in sd.items():
            if not wanted(k): continue
            if not hasattr(v, "dim") or v.dim() != 2: continue
            R, C = v.shape
            stem = safe(k.replace('.weight',''))
            qfile = f"{stem}.q4.bin"
            sfile = f"{stem}.q4.scales.fp16.bin"
            qpath = Path(Q4_DIR) / qfile
            spath = Path(Q4_DIR) / sfile

            quantize_2d_per_row(v.float(), qpath, spath)
            del v; gc.collect()

            manifest[k] = {
                "dtype": "int4",
                "scheme": "weight_only",
                "per": "row",
                "bits": 4,
                "pack": "nibble_lohi",
                "scale_dtype": "fp16",
                "zero_point": 0,
                "symmetric": True,
                "align": 256,
                "rows": int(R), "cols": int(C),
                "int4_bin": str(qpath.as_posix()),
                "scale_bin": str(spath.as_posix()),
            }
        del sd; gc.collect()

    with open(MAN_OUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] wrote {MAN_OUT} with {len(manifest)} entries; q4 dir={Q4_DIR}")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    main()
