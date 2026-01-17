# VM_run.md (end-to-end guide)

This document describes how to run Clocher from scratch on a fresh VM: clone, download the model (multiple options), generate artifacts, build CPU/CUDA, and run `make bench` / `make bench-cuda` with all available flags.

---

## 0) VM prerequisites

### CPU-only (no GPU)
- Linux x86_64
- Build tools: `gcc`, `make`, `pkg-config`
- Python 3.10+ (3.11+ recommended)
- Git

### CUDA (NVIDIA GPU)
- NVIDIA driver installed and working (`nvidia-smi`)
- CUDA Toolkit (nvcc in PATH)
- (Optional) cuDNN if you use extra tooling; not required for this repo

### Python deps used by scripts
Create and activate a venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Install base deps (adjust to your environment):
```bash
python -m pip install --upgrade pip
python -m pip install numpy torch safetensors huggingface_hub
```

---

## 1) Clone the repository

```bash
git clone <REPO_URL> Clocher
cd Clocher
```

---

## 2) Download the model (ALL options)

The repo expects the model under:
```
models/gpt-oss-20b/hf
or
models/gpt-oss-20b/hf/original
```

### Option A: repo helper (Hugging Face Hub)
```bash
python scripts/download_gpt_oss_20b.py
```
This writes to `models/gpt-oss-20b/hf`.

### Option B: PTQ script (downloads via snapshot_download if missing)
```bash
python scripts/ptq_from_hf.py --help
```
This script will download via `huggingface_hub.snapshot_download` when needed.

### Option C: huggingface-cli (external to repo)
```bash
huggingface-cli download openai/gpt-oss-20b \
  --local-dir models/gpt-oss-20b/hf/original \
  --local-dir-use-symlinks False
```

### Option D: manual (fallback)
Download shards (`pytorch_model-*.bin` or `model-*.safetensors`) and place them in:
```
models/gpt-oss-20b/hf/original/
```

---

## 3) Generate artifacts (IEBIN + Q4 + dedup)

### 3.1) IEBIN (model.ie.json / model.ie.bin)
For indexed HF safetensors:
```bash
python scripts/hf_to_iebin.py \
  --hf-dir models/gpt-oss-20b/hf/original \
  --out-dir models/gpt-oss-20b
```

Alternatives:
```bash
python scripts/hf_to_iebin_raw.py --hf-dir <HF_DIR> --out-dir models/gpt-oss-20b
python scripts/hf_to_iebin_stream.py --hf-dir <HF_DIR> --out-dir models/gpt-oss-20b
python scripts/hf_to_iebin_stream_pt.py --hf-dir <HF_DIR> --out-dir models/gpt-oss-20b
```

### 3.2) Q4 streaming (model.q4.bin + scales)
```bash
MODEL_DIR="$PWD/models/gpt-oss-20b"
IE_Q4_INCLUDE_LM_HEAD=1 \
CHUNK_ROWS=128 \
python scripts/gen_q4_bytes_stream.py \
  --hf-dir "$MODEL_DIR/hf/original" \
  --q4-bytes "$MODEL_DIR/model.q4.bin" \
  --scales "$MODEL_DIR/model.q4.scales.fp16.bin" \
  --manifest quant/q4_manifest.expanded.json
```

Batch driver:
```bash
CHUNK_ROWS=128 \
bash scripts/gen_q4_bytes_stream_all.sh "$MODEL_DIR/hf/original" "$MODEL_DIR"
```

### 3.3) Q4 attention compat (optional)
```bash
python scripts/append_q4_attn_from_hf.py \
  --hf-dir "$MODEL_DIR/hf/original" \
  --compat-json "$MODEL_DIR/model.ie.compat.json" \
  --q4-bin "$MODEL_DIR/model.q4.bin"
```

### 3.4) Dedup (lossless)
```bash
MODEL_DIR="$PWD/models/gpt-oss-20b"
bash scripts/run_quant_and_dedup.sh "$MODEL_DIR"
```
Expected outputs:
```
models/gpt-oss-20b/model.dedup.defaults.bin
models/gpt-oss-20b/model.dedup.masks.bin
models/gpt-oss-20b/model.dedup.exceptions.bin
models/gpt-oss-20b/model.dedup.json
```

---

## 4) Build (CPU and CUDA)

### CPU
```bash
make build
```

### CUDA
```bash
make build-cuda
```

If `make build-cuda` fails, confirm:
- `nvcc` on PATH
- NVIDIA driver installed

---

## 5) Benchmarks (CPU and CUDA)

### 5.1) CPU benchmark (make bench)
Minimal example:
```bash
MODEL_DIR="$PWD/models/gpt-oss-20b" \
PROMPTS="benchmarks/prompts_10.txt" \
PRECISION=int4 \
MAX_NEW=1 RUNS=1 ROUNDS=1 \
IE_DEDUP=1 IE_DEDUP_STRICT=1 IE_VERIFY_TOUCH=1 \
make bench
```

### 5.2) CUDA benchmark (make bench-cuda)
```bash
MODEL_DIR="$PWD/models/gpt-oss-20b" \
PROMPTS="benchmarks/prompts_10.txt" \
PRECISION=int4 \
MAX_NEW=1 RUNS=1 ROUNDS=1 \
IE_DEDUP=1 IE_DEDUP_STRICT=1 IE_VERIFY_TOUCH=1 \
make bench-cuda
```

### 5.3) Report-only (bench-report)
```bash
REPORT=1 REPORT_ROOT="$PWD/benchmarks/reports" \
MODEL_DIR="$PWD/models/gpt-oss-20b" \
PROMPTS="benchmarks/prompts_10.txt" \
PRECISION=int4 MAX_NEW=1 RUNS=1 ROUNDS=1 \
make bench-report
```

### 5.4) Skip verification (faster)
```bash
VERIFY=0 VERIFY_HF_IDS=0 REPORT=0 \
MODEL_DIR="$PWD/models/gpt-oss-20b" \
PROMPTS="benchmarks/prompts_10.txt" \
PRECISION=int4 MAX_NEW=1 RUNS=1 ROUNDS=1 \
make bench
```

---

## 6) HF ID verification (strict)

Enables HF ID checks (prompt IDs + next-token IDs). This can be memory-heavy on CPU for 20B; recommended on GPU/large-RAM hosts.

```bash
HF_PY="$PWD/.venv/bin/python" \
HF_DIR="$PWD/models/gpt-oss-20b/hf/original" \
HF_DEVICE=cpu HF_DTYPE=bf16 \
HF_TOPK=5 HF_LOGITS_TOL=1e-3 \
VERIFY_HF_IDS=1 \
MODEL_DIR="$PWD/models/gpt-oss-20b" \
PROMPTS="benchmarks/prompts_10.txt" \
PRECISION=int4 MAX_NEW=1 RUNS=1 ROUNDS=1 \
make bench
```

For GPU:
```bash
HF_DEVICE=cuda HF_DTYPE=bf16
```

---

## 7) ALL environment variables (Makefile / bench)

### General (bench / bench-cuda / bench-report)
- `MODEL_DIR` : model directory
- `PROMPTS` : prompts file path
- `RUNS` : number of runs
- `ROUNDS` : rounds per prompt
- `THREADS` : CPU threads
- `BATCH` : batch size
- `PRECISION` : `fp32|bf16|fp16|int4|int4w|int8w`
- `IE_PRECISION` : precision label passed to the engine
- `PREFETCH` : `on|off|auto|N`
- `PRETRANSPOSE` : `none|woh|wxh|all`
- `AFFINITY` : `auto|compact|scatter`
- `MAX_NEW` : max new tokens
- `WARMUP_TOKENS` : warmup tokens (default 0)

### Work-touch / memory
- `IE_BYTES_PER_TOKEN`
- `IE_STRIDE_BYTES`
- `IE_VERIFY_TOUCH`

### Dedup (lossless)
- `IE_DEDUP`
- `IE_DEDUP_STRICT`
- `IE_DEDUP_POLICY`
- `IE_DEDUP_CACHE_MB`
- `IE_DEDUP_HOT_BYTES`
- `IE_DEDUP_HOT_LIST`

### Streaming/memory heuristics
- `IE_PREFETCH_DISTANCE`
- `IE_NT_LOADS`
- `IE_L3_BYTES`
- `IE_NT_THRESHOLD_RATIO`
- `IE_STREAM_BLOCK_BYTES`
- `IE_REUSE_GUARD_WINDOWS`

### Reporting / verification
- `REPORT` : 1 to emit report
- `REPORT_ROOT` : report output dir
- `REPORT_SCRIPT` : report script path (default `scripts/run_harness_report.py`)
- `REPORT_ARGS` : extra args to report script
- `REPORT_TOKENS_MAX` : cap tokens in report
- `REPORT_PATH` : verify a specific report
- `STRICT_REPORT_PATH` : strict JSON path for verify
- `VERIFY` : 0 to skip bench-verify

### Hugging Face verification
- `VERIFY_HF_IDS` : 1 to enable
- `HF_PY` : python with HF deps
- `HF_DIR` : HF model path
- `HF_DEVICE` : `cpu|cuda`
- `HF_DTYPE` : `fp32|fp16|bf16`
- `HF_TOPK` : top-k to compare
- `HF_LOGITS_TOL` : logits tolerance

---

## 8) Engine CLI flags

These are accepted by `build/inference-engine*`:
```
--device auto|cpu|cuda|ze
--model-dir PATH
--model-json PATH
--model-bin PATH
--pretranspose none|woh|wxh|all
--prefetch on|off|auto|N
--warmup N
--rounds N
--prompts-file PATH
--aggregate
--threads N
--precision fp32|bf16|fp16|int4|int4w|int8w
--batch N
--max-new N
--prompt "text"
```

---

## 9) Direct execution (no Makefile)

CPU:
```bash
./build/inference-engine \
  --model-dir models/gpt-oss-20b \
  --precision int4 \
  --threads 12 \
  --rounds 1 \
  --warmup 0 \
  --prompts-file benchmarks/prompts_10.txt \
  --max-new 1
```

CUDA:
```bash
./build/inference-engine.cuda \
  --model-dir models/gpt-oss-20b \
  --precision int4 \
  --threads 12 \
  --rounds 1 \
  --warmup 0 \
  --prompts-file benchmarks/prompts_10.txt \
  --max-new 1
```

---

## 10) Debug tips
- If `make bench` complains about dedup: ensure `model.dedup.json` and blobs exist in `MODEL_DIR`.
- If `REPORT_ROOT` is empty or wrong, set an absolute path.
- To reduce runtime, use `VERIFY=0 REPORT=0`.
- To avoid HF OOM on CPU for 20B, use `HF_DEVICE=cuda` on a GPU VM or disable `VERIFY_HF_IDS`.
