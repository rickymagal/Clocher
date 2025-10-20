import os, json, sys
from collections import defaultdict
from safetensors import safe_open
import numpy as np
import torch

RAW_DIR = "models/neox20b_raw"
OUT_DIR = "models/gpt-oss-20b"
os.makedirs(OUT_DIR, exist_ok=True)

index_path = os.path.join(RAW_DIR, "model.safetensors.index.json")
if not os.path.isfile(index_path):
    print(f"[erro] não achei {index_path}. Rode o download dos shards.", file=sys.stderr); sys.exit(2)

with open(index_path, "r") as f:
    index = json.load(f)
weight_map = index.get("weight_map", {})
if not weight_map:
    print("[erro] index sem weight_map", file=sys.stderr); sys.exit(2)

by_file = defaultdict(list)
for name, shard in weight_map.items():
    by_file[shard].append(name)
for shard in by_file:
    by_file[shard].sort()

meta = {"tensors": [], "dtype": "float32"}
bin_path = os.path.join(OUT_DIR, "model.ie.bin")
offset = 0

with open(bin_path, "wb") as fbin:
    for shard in sorted(by_file.keys()):
        shard_path = os.path.join(RAW_DIR, shard)
        if not os.path.isfile(shard_path):
            print(f"[warn] shard ausente: {shard_path}", file=sys.stderr)
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for name in by_file[shard]:
                if name not in sf.keys():
                    print(f"[warn] tensor {name} não está em {shard}", file=sys.stderr)
                    continue
                t = sf.get_tensor(name)  # torch.Tensor lazy em CPU
                if t.dtype != torch.float32:
                    t = t.to(dtype=torch.float32)
                if not t.is_contiguous():
                    t = t.contiguous()
                arr = t.numpy()
                buf = arr.tobytes(order="C")
                fbin.write(buf)
                meta["tensors"].append({
                    "name": name,
                    "shape": list(arr.shape),
                    "offset": int(offset),
                    "nbytes": int(len(buf)),
                    "dtype": "float32"
                })
                offset += len(buf)

with open(os.path.join(OUT_DIR, "model.ie.json"), "w") as f:
    json.dump(meta, f, indent=2)

with open(os.path.join(OUT_DIR, "vocab.json"), "w") as f:
    json.dump({"eos_token": "", "pad_token": "", "model_max_length": 2048}, f)

print("[ok] export concluído")
print("[info] bin:", bin_path, "tamanho=", os.path.getsize(bin_path))
