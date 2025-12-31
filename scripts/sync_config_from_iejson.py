#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from datetime import datetime

LAYER_RX = re.compile(r"^model\.layers\.(\d+)\.")

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)

def find_tensor(tensors, name: str):
    for t in tensors:
        if t.get("name") == name:
            return t
    return None

def infer_num_layers(tensors) -> int:
    mx = -1
    for t in tensors:
        n = t.get("name", "")
        m = LAYER_RX.match(n)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1 if mx >= 0 else 0

def main():
    ap = argparse.ArgumentParser(description="Sync config.json fields from model.ie.json tensor shapes.")
    ap.add_argument("--model-dir", required=True, help="Model directory containing config.json and model.ie.json")
    ap.add_argument("--dry-run", action="store_true", help="Print changes but do not write config.json")
    args = ap.parse_args()

    model_dir = args.model_dir
    ie_path = os.path.join(model_dir, "model.ie.json")
    cfg_path = os.path.join(model_dir, "config.json")

    if not os.path.isfile(ie_path):
        raise SystemExit(f"missing: {ie_path}")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"missing: {cfg_path}")

    ie = load_json(ie_path)
    cfg = load_json(cfg_path)

    tensors = ie.get("tensors", [])
    if not isinstance(tensors, list) or not tensors:
        raise SystemExit("model.ie.json has no tensors[]")

    changes = {}

    # vocab_size + hidden_size from embeddings (most reliable)
    emb = find_tensor(tensors, "model.embed_tokens.weight")
    if emb and isinstance(emb.get("shape"), list) and len(emb["shape"]) == 2:
        vocab_size = int(emb["shape"][0])
        hidden_size = int(emb["shape"][1])
        changes["vocab_size"] = vocab_size
        changes["hidden_size"] = hidden_size

    # num_hidden_layers from tensor name scan
    n_layers = infer_num_layers(tensors)
    if n_layers:
        changes["num_hidden_layers"] = int(n_layers)

    # intermediate_size (d_ff) from MoE gate/up bias: shape [n_experts, 2*d_ff]
    gu_bias = find_tensor(tensors, "model.layers.0.mlp.experts.gate_up_proj_bias")
    if gu_bias and isinstance(gu_bias.get("shape"), list) and len(gu_bias["shape"]) == 2:
        two_dff = int(gu_bias["shape"][1])
        if two_dff % 2 != 0:
            raise SystemExit(f"gate_up_proj_bias second dim is odd: {two_dff}")
        changes["intermediate_size"] = int(two_dff // 2)

    # attention heads from "sinks" length (this model stores per-head sinks)
    sinks = find_tensor(tensors, "model.layers.0.self_attn.sinks")
    q_bias = find_tensor(tensors, "model.layers.0.self_attn.q_proj.bias")
    k_bias = find_tensor(tensors, "model.layers.0.self_attn.k_proj.bias")
    if sinks and isinstance(sinks.get("shape"), list) and len(sinks["shape"]) == 1:
        n_heads = int(sinks["shape"][0])
        if q_bias and isinstance(q_bias.get("shape"), list) and len(q_bias["shape"]) == 1:
            qdim = int(q_bias["shape"][0])
            if qdim % n_heads == 0:
                head_dim = qdim // n_heads
                changes["num_attention_heads"] = n_heads
                if k_bias and isinstance(k_bias.get("shape"), list) and len(k_bias["shape"]) == 1:
                    kdim = int(k_bias["shape"][0])
                    if head_dim and (kdim % head_dim == 0):
                        changes["num_key_value_heads"] = int(kdim // head_dim)

    # Apply changes (only if they differ)
    to_write = {}
    for k, v in changes.items():
        old = cfg.get(k)
        if old != v:
            to_write[k] = (old, v)
            cfg[k] = v

    if not to_write:
        print("no changes needed (config.json already matches model.ie.json shapes)")
        return

    print("config.json updates:")
    for k in sorted(to_write):
        old, new = to_write[k]
        print(f"  {k}: {old} -> {new}")

    if args.dry_run:
        print("dry-run: not writing config.json")
        return

    # Backup then write
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = cfg_path + f".bak.{ts}"
    shutil.copy2(cfg_path, bak)
    save_json(cfg_path, cfg)
    print(f"wrote {cfg_path} (backup: {bak})")

if __name__ == "__main__":
    main()
