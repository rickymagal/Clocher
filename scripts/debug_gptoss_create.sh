#!/usr/bin/env bash
set -u

MODEL_DIR="${1:-models/gpt-oss-20b}"
IE_JSON="$MODEL_DIR/model.ie.json"
IE_BIN="$MODEL_DIR/model.ie.bin"
TMAP_JSON="$MODEL_DIR/tensor_map.json"
HP_JSON="$MODEL_DIR/hparams.auto.json"
CFG1="$MODEL_DIR/config.json"
CFG2="$MODEL_DIR/hf/original/config.json"

echo "MODEL_DIR=$MODEL_DIR"
echo

echo "[1/6] File presence"
ls -la "$IE_JSON" 2>/dev/null || true
ls -la "$IE_BIN" 2>/dev/null || true
ls -la "$TMAP_JSON" 2>/dev/null || true
ls -la "$HP_JSON" 2>/dev/null || true
ls -la "$CFG1" 2>/dev/null || true
ls -la "$CFG2" 2>/dev/null || true
echo

echo "[2/6] Generate tensor_map.json + hparams.auto.json from model.ie.json"
python3 - <<'PY'
import json, os, re, sys
from math import prod

model_dir = os.environ.get("MODEL_DIR", "models/gpt-oss-20b")
ie_json   = os.path.join(model_dir, "model.ie.json")
tmap_json = os.path.join(model_dir, "tensor_map.json")
hp_json   = os.path.join(model_dir, "hparams.auto.json")

with open(ie_json, "r", encoding="utf-8") as f:
    j = json.load(f)

tensors = j.get("tensors", [])
if not isinstance(tensors, list) or not tensors:
    raise SystemExit("model.ie.json has no tensors[]")

by = {}
for t in tensors:
    if isinstance(t, dict) and isinstance(t.get("name"), str):
        by[t["name"]] = t

def get_int_field(t, *keys):
    for k in keys:
        v = t.get(k)
        if v is not None:
            return int(v)
    return None

def get_shape(name):
    t = by.get(name)
    if not t:
        return None
    sh = t.get("shape")
    if isinstance(sh, list) and all(isinstance(x, int) for x in sh) and len(sh) >= 1:
        return sh
    return None

def pick_existing(*cands):
    for c in cands:
        if c in by:
            return c
    return ""

# ---- tensor_map.json
out = []
for name, t in by.items():
    off = get_int_field(t, "offset")
    nb  = get_int_field(t, "size_bytes", "nbytes", "byte_size")
    if off is None or nb is None:
        continue
    out.append({"name": name, "offset": off, "size_bytes": nb})

if not out:
    raise SystemExit("No tensors had (name, offset, size_bytes).")

with open(tmap_json, "w", encoding="utf-8") as f:
    json.dump({"tensors": out}, f, indent=2)

print(f"Wrote {tmap_json} tensors={len(out)}")

# ---- core dims
embed = pick_existing("model.embed_tokens.weight", "transformer.wte.weight", "tok_embeddings.weight")
if not embed:
    raise SystemExit("Could not find embed_tokens weight.")

esh = get_shape(embed)
if not esh or len(esh) != 2:
    raise SystemExit(f"Embed tensor has no usable shape: {embed} shape={esh}")

vocab_size, d_model = esh[0], esh[1]

# layers count
layer_ids = []
pat = re.compile(r"^model\.layers\.(\d+)\.")
for name in by.keys():
    m = pat.match(name)
    if m:
        layer_ids.append(int(m.group(1)))
n_layers = (max(layer_ids) + 1) if layer_ids else 0
if n_layers <= 0:
    raise SystemExit("Could not detect n_layers from tensor names.")

# attention proj dims from shapes (preferred) else bytes fallback is unreliable across dtypes
q_name = pick_existing(
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.qweight",
    "model.layers.0.attn.q_proj.weight",
)
k_name = pick_existing(
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.k_proj.qweight",
    "model.layers.0.attn.k_proj.weight",
)
v_name = pick_existing(
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.v_proj.qweight",
    "model.layers.0.attn.v_proj.weight",
)

if not (q_name and k_name and v_name):
    # If naming differs, print helpful candidates and fail.
    cands = [n for n in by.keys() if n.startswith("model.layers.0") and "attn" in n and n.endswith("weight")]
    cands = sorted(cands)[:50]
    raise SystemExit("Could not find q/k/v proj weights. First 50 attn weights:\n" + "\n".join(cands))

def out_dim_from_proj(name):
    sh = get_shape(name)
    if sh and len(sh) == 2:
        a, b = sh
        # one dim should match d_model (input features); the other is output
        if a == d_model and b != d_model:
            return b
        if b == d_model and a != d_model:
            return a
        # if neither matches, still return first dim as a fallback
        return sh[0]
    # bytes fallback (very shaky). Fail instead of guessing wrong.
    raise SystemExit(f"Proj tensor missing shape; cannot infer dims safely: {name}")

q_dim  = out_dim_from_proj(q_name)
kv_dim = out_dim_from_proj(k_name)
kv_dim2= out_dim_from_proj(v_name)
if kv_dim2 != kv_dim:
    print(f"warn: k_proj out_dim={kv_dim} v_proj out_dim={kv_dim2} (using k_proj)")

# head geometry: choose a head_dim that divides both q_dim and kv_dim, prefer 64 then 128 then 32, else max <= 256.
prefs = [64, 128, 32, 80, 96, 48, 40, 24, 16]
head_dim = None
for hd in prefs:
    if q_dim % hd == 0 and kv_dim % hd == 0:
        head_dim = hd
        break
if head_dim is None:
    # fallback: largest common divisor <= 256
    for hd in range(256, 7, -1):
        if q_dim % hd == 0 and kv_dim % hd == 0:
            head_dim = hd
            break
if head_dim is None:
    raise SystemExit(f"Could not find a consistent head_dim for q_dim={q_dim} kv_dim={kv_dim}")

n_heads = q_dim // head_dim
n_kv_heads = kv_dim // head_dim

# d_ff from MLP shapes
mlp_candidates = [
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.fc1.weight",
    "model.layers.0.mlp.w1.weight",
    "model.layers.0.mlp.gate.weight",
]
mlp_found = ""
for nm in mlp_candidates:
    if nm in by:
        mlp_found = nm
        break
if not mlp_found:
    # try fuzzy
    for nm in by.keys():
        if nm.startswith("model.layers.0") and ".mlp." in nm and nm.endswith(".weight"):
            if any(k in nm for k in ("gate", "up", "fc1", "w1")):
                mlp_found = nm
                break
if not mlp_found:
    cands = [n for n in by.keys() if n.startswith("model.layers.0") and ".mlp." in n and n.endswith(".weight")]
    cands = sorted(cands)[:80]
    raise SystemExit("Could not find an MLP proj weight to infer d_ff. First 80 mlp weights:\n" + "\n".join(cands))

mlp_sh = get_shape(mlp_found)
if not mlp_sh or len(mlp_sh) != 2:
    raise SystemExit(f"MLP tensor has no usable shape: {mlp_found} shape={mlp_sh}")

a, b = mlp_sh
if a == d_model and b != d_model:
    d_ff = b
elif b == d_model and a != d_model:
    d_ff = a
else:
    d_ff = max(a, b)

# experts count (best-effort)
n_experts = None
for nm in by.keys():
    if nm.startswith("model.layers.0") and ("experts" in nm or "moe" in nm):
        # very rough: look for "experts.<id>."
        m = re.search(r"\.experts\.(\d+)\.", nm)
        if m:
            n_experts = max(n_experts or 0, int(m.group(1)) + 1)

hp = {
    "d_model": int(d_model),
    "vocab_size": int(vocab_size),
    "n_layers": int(n_layers),
    "n_heads": int(n_heads),
    "n_kv_heads": int(n_kv_heads),
    "d_head": int(head_dim),
    "d_ff": int(d_ff),
    "q_dim": int(q_dim),
    "kv_dim": int(kv_dim),
}
if n_experts is not None:
    hp["n_experts_layer0"] = int(n_experts)

with open(hp_json, "w", encoding="utf-8") as f:
    json.dump(hp, f, indent=2)

print(f"Wrote {hp_json}")
print(json.dumps(hp, indent=2))
PY
echo

echo "[3/6] Sanity: tensor_map end offset vs model.ie.bin size"
python3 - <<'PY'
import json, os, sys
model_dir = os.environ.get("MODEL_DIR", "models/gpt-oss-20b")
tmap = os.path.join(model_dir, "tensor_map.json")
binp = os.path.join(model_dir, "model.ie.bin")
if not os.path.exists(tmap):
    raise SystemExit("tensor_map.json missing")
if not os.path.exists(binp):
    raise SystemExit("model.ie.bin missing")
j = json.load(open(tmap, "r", encoding="utf-8"))
mx = 0
for t in j.get("tensors", []):
    off = int(t["offset"]); nb = int(t["size_bytes"])
    mx = max(mx, off + nb)
sz = os.path.getsize(binp)
print("model.ie.bin size_bytes =", sz)
print("max(offset+size_bytes) =", mx)
if mx > sz:
    print("ERROR: tensor map references beyond end of model.ie.bin")
    sys.exit(2)
print("OK")
PY
echo

echo "[4/6] IMPORTANT: config.json availability"
if [ -f "$CFG1" ]; then
  echo "Found $CFG1"
elif [ -f "$CFG2" ]; then
  echo "Found $CFG2"
else
  echo "MISSING config.json in both:"
  echo "  $CFG1"
  echo "  $CFG2"
  echo "If your engine still requires it, download it (example):"
  echo "  hf download openai/gpt-oss-20b config.json --local-dir $MODEL_DIR/hf/original --local-dir-use-symlinks False"
fi
echo

echo "[5/6] Rebuild"
make build
echo

echo "[6/6] Run with maximum logging"
export IE_LOG_LEVEL=debug
export IE_LOG=debug
export IE_DEBUG=1
export IE_VERIFY_TOUCH=1
export IE_BYTES_PER_TOKEN=67108864
export IE_STRIDE_BYTES=256

./build/inference-engine --model-dir "$MODEL_DIR" --prompt "Hello." --max-new 32
rc=$?
echo "exit=$rc"
exit $rc
