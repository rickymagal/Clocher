#!/usr/bin/env python3
import argparse, json, os, sys

def get_tensor(ie, name):
    for t in ie.get("tensors", []):
        if t.get("name") == name:
            return t
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    cfg_path = os.path.join(args.model_dir, "config.json")
    ie_path  = os.path.join(args.model_dir, "model.ie.json")

    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    ie  = json.load(open(ie_path,  "r", encoding="utf-8"))

    n_heads    = cfg.get("num_attention_heads") or cfg.get("n_heads")
    n_kv_heads = cfg.get("num_key_value_heads") or cfg.get("n_kv_heads")
    d_model    = cfg.get("hidden_size") or cfg.get("d_model")

    if not isinstance(n_heads, int) or n_heads <= 0:
        print("ERROR: could not read num_attention_heads from config.json", file=sys.stderr)
        return 2
    if not isinstance(n_kv_heads, int) or n_kv_heads <= 0:
        print("ERROR: could not read num_key_value_heads from config.json", file=sys.stderr)
        return 2
    if not isinstance(d_model, int) or d_model <= 0:
        print("ERROR: could not read hidden_size from config.json", file=sys.stderr)
        return 2

    q = get_tensor(ie, "model.layers.0.self_attn.q_proj.weight")
    k = get_tensor(ie, "model.layers.0.self_attn.k_proj.weight")
    if not q or not k:
        print("ERROR: missing q_proj/k_proj in model.ie.json", file=sys.stderr)
        return 3

    q_dim = int(q["shape"][0])
    k_dim = int(k["shape"][0])

    if q_dim % n_heads != 0:
        print(f"ERROR: q_dim={q_dim} not divisible by n_heads={n_heads}", file=sys.stderr)
        return 4

    head_dim = q_dim // n_heads
    ok_k = (k_dim == n_kv_heads * head_dim)

    print(f"d_model={d_model}")
    print(f"n_heads={n_heads}")
    print(f"n_kv_heads={n_kv_heads}")
    print(f"q_proj_out={q_dim}")
    print(f"k_proj_out={k_dim}")
    print(f"inferred head_dim={head_dim}")
    print(f"k_proj_matches (k_dim == n_kv_heads*head_dim) = {ok_k}")

    cfg_head_dim = d_model // n_heads if (d_model % n_heads == 0) else None
    print(f"config_derived head_dim (hidden_size/heads) = {cfg_head_dim}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
