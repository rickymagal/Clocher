# Inference Engine (Clocher)

_A minimal C11 inference baseline with strict metrics & a Python harness._  
**Last updated:** 2025-10-24 21:00:48 UTC

---

## Quick start

```bash
# Build CPU and (optionally) CUDA binaries
make build
make build-cuda   # requires a CUDA toolchain; produces build/inference-engine.cuda

# Sanity: show CLI
./build/inference-engine --help
```

### Model format (IEBIN v1)

The engine consumes a pair of files in the current working directory:

```
model.ie.json   # metadata (tensor names, shapes, dtypes, scales)
model.ie.bin    # raw, mmap‑friendly tensor blob
```

You can pack from a Hugging Face checkpoint using the helper script below.

---

## NEW: INT4 **weight‑only** PTQ path (Q4)

This repository now includes a **4‑bit (INT4) weight‑only** path that trades a small accuracy budget for **4× smaller weights** and a substantial **memory‑bandwidth reduction**. It is designed to be drop‑in with the existing build and harness.

### Pipeline overview

1) **Prepare/locate a HF model** (already sharded is fine):

```
models/gpt-oss-20b/hf/
  ├── config.json
  ├── tokenizer.json / vocab.json / merges.txt
  └── pytorch_model-00001-of-00046.bin … pytorch_model-00046-of-00046.bin
```

2) **(Calibrate) produce the INT4 manifest**  
Use one of the PTQ helper scripts to create a manifest describing which tensors are quantized to 4‑bit and their per‑row/per‑tensor scales:

```bash
# Choose one depending on your source of truth and available samples.
# See: python3 scripts/ptq_from_hf.py --help
#      python3 scripts/ptq_from_bin.py --help
#      python3 scripts/ptq_from_source.py --help

python3 scripts/ptq_from_hf.py   --hf-dir models/gpt-oss-20b/hf   --out quant/q4_manifest.json   --bits 4 --weights-only 1
```

Notes:
- The manifest (JSON) maps tensor names to **INT4 packing + scale params**.
- Calib options (group size, symmetric/affine, clamp, percentile) are exposed by the script; pick values that satisfy your accuracy budget.

3) **Pack Hugging Face → IEBIN (with INT4)**

```bash
python3 scripts/hf_to_iebin.py   --hf-dir models/gpt-oss-20b/hf   --out-dir models/gpt-oss-20b   --q4-map quant/q4_manifest.json
```

This writes `model.ie.json` and `model.ie.bin` in `models/gpt-oss-20b/` (or as configured).

4) **Run the benchmark (CPU or CUDA) with INT4 weights**

The engine selects kernels by **precision**. For INT4 **weight‑only**, use `int4w`.

> **64 MB per token** = `IE_BYTES_PER_TOKEN=64000000` (decimal bytes).

**CPU:**

```bash
cd models/gpt-oss-20b
PROMPTS=../../benchmarks/prompts_10..txt IE_PRECISION=int4w IE_REQUIRE_MODEL=1 IE_BYTES_PER_TOKEN=64000000 IE_STRIDE_BYTES=256 RUNS=3 make -C ../.. bench
```

**CUDA:**

```bash
cd models/gpt-oss-20b
PROMPTS=../../benchmarks/prompts_10..txt IE_PRECISION=int4w IE_REQUIRE_MODEL=1 IE_BYTES_PER_TOKEN=64000000 IE_STRIDE_BYTES=256 RUNS=3 make -C ../.. bench-cuda
```

The CLI will emit a **single JSON line** per run, which the harness merges into `docs/PERFORMANCE.md`.

---

## Operational notes

- The `--device` flag is accepted by the CLI as a **no‑op hint** (selection is a build‑time concern). CPU: `build/inference-engine`, CUDA: `build/inference-engine.cuda`.
- **Strict mode**: set `IE_REQUIRE_MODEL=1` to require a valid `model.ie.json` + `model.ie.bin`. Otherwise the engine can operate in a deterministic **stub** mode for CI.
- **Work‑touch**: to emulate memory pressure, set `IE_BYTES_PER_TOKEN` (e.g. `64000000`) and `IE_STRIDE_BYTES` (e.g. `256`). The measured window includes generation **and** this per‑token touch over the mmap’d `model.ie.bin`.
- **RSS reporting**: the CLI samples peak RSS after the timed window (Linux: `/proc/self/status` `VmHWM`, fallback `getrusage`).

---

## Troubleshooting

- `ERROR: --q4-map not found`: pass a real path to your manifest, e.g. `quant/q4_manifest.json`.
- `error: unknown flag '--model-dir'` or `'--rounds'`: you are using an older binary; rebuild with `make clean && make build`.
- `RSS peak = 0`: ensure the process reads enough memory while resident (set `IE_BYTES_PER_TOKEN` to a non‑zero value) and you are not running inside a constrained container namespace that suppresses `VmHWM`.
- Prompts file not found: `PROMPTS=benchmarks/prompts_10..txt` (note the **double dot** in the demo file).

---

## What INT4 (weight‑only) means here

- **Activations** remain floating‑point; only **weights** are nibble‑packed (2 weights per byte) with per‑row scales.
- Dequantization is fused in the matmul kernels (see `engine/src/quant/int4_ptq.c` and GPU equivalents). Pretranspose/blocked‑K packing are applied before/after quantization as configured.
- Accuracy is safeguarded by a **calibration gate** (cosine similarity or task‑level eval), configurable in the PTQ scripts.

---

## See also

- Detailed design and optimization path: `docs/DESIGN.md`, `docs/adr-00060-optimization-path.md`
- Decision log: `docs/DECISIONS.md`


---
## INT4 Weight-Only — Addendum (2025-10-24 21:04:23 UTC)

This addendum **adds** a new optional path to the existing pipeline for **INT4 (weight-only)** packing and benchmarking. It does **not** replace the FP32/BF16/FP16 flows.

### Summary
- Quantization: post-training *weight-only* INT4 (aka `int4w`) via a manifest.
- Export: `scripts/hf_to_iebin.py` supports `--q4-map` to pack tensors into `model.ie.bin` while preserving `model.ie.json` meta.
- Runtime: select precision with `IE_PRECISION=int4w` (or `--precision int4w`), independent from host math precision.
- Benchmarks: `make bench` / `make bench-cuda` support strict runs with realistic memory pressure via `IE_BYTES_PER_TOKEN`.

### Prerequisites
- A working HF model directory (already used in your FP32 flow), e.g. `models/gpt-oss-20b/hf`.
- A quantization manifest (example at `quant/q4_manifest.json`). See **Manifest template** below.
- Python deps already used by `scripts/hf_to_iebin.py` (torch, numpy).

### Export to IEBIN with INT4
```bash
python3 scripts/hf_to_iebin.py   --hf-dir models/gpt-oss-20b/hf   --out-dir models/gpt-oss-20b   --q4-map quant/q4_manifest.json
```
This produces (or updates) `models/gpt-oss-20b/model.ie.json` and `models/gpt-oss-20b/model.ie.bin` with weight-only INT4 packing per manifest.

> Tip: if you see `ERROR: --q4-map not found`, double‑check the path to your manifest.

### Run the strict benchmark (CPU)
```bash
PROMPTS=benchmarks/prompts_10..txt IE_PRECISION=int4w IE_REQUIRE_MODEL=1 IE_BYTES_PER_TOKEN=64000000 IE_STRIDE_BYTES=256 RUNS=3 make bench
```
- `IE_REQUIRE_MODEL=1` makes the CLI fail if `model.ie.{json,bin}` are missing.
- `IE_BYTES_PER_TOKEN` enables the per-token *work-touch* loop over `model.ie.bin` to mimic bandwidth pressure.
- `IE_STRIDE_BYTES` controls the touch stride (256 is a good default).

### Run the strict benchmark (CUDA)
```bash
PROMPTS=benchmarks/prompts_10..txt IE_PRECISION=int4w IE_REQUIRE_MODEL=1 IE_BYTES_PER_TOKEN=64000000 IE_STRIDE_BYTES=256 RUNS=3 make bench-cuda
```

### Manifest template (example)
Save as `quant/q4_manifest.json`:
```json
{
  "version": 1,
  "rules": [
    {
      "pattern": ".*attn.*(q_proj|k_proj|v_proj|o_proj).*",
      "dtype":   "int4w",
      "group":    64,
      "zero":     "per-tensor",
      "scale":    "per-channel"
    },
    {
      "pattern": ".*mlp.*(gate_proj|up_proj|down_proj).*",
      "dtype":   "int4w",
      "group":    64,
      "zero":     "per-tensor",
      "scale":    "per-channel"
    }
  ]
}
```

### Troubleshooting
- **Unknown flag errors**: The CLI now accepts `--device`, `--model-dir`, `--model-json`, `--model-bin`, and `--rounds` as documented. If your harness still injects older flags, update it, or run the engine directly.
- **RSS peak shows 0 MB**: Ensure you are on Linux and that the engine is built with the updated `ie_metrics_sample_rss_peak()` (reads `/proc/self/status` → `VmHWM`, then falls back to `getrusage`). The JSON is captured *after* the measured window; the value will be `0` only if the OS reports `0` or if the sampling failed.

### Notes
- The **precision label** (`IE_PRECISION`) is a *storage / weight* hint, not necessarily the CPU math precision. You can still set `PRECISION=fp32` for host compute while using `IE_PRECISION=int4w` for storage.
- Deterministic stub mode remains available when `IE_REQUIRE_MODEL` is not set; strict runs require a valid IEBIN pair.


## Repository Layout

```text
.
├── benchmarks
│   ├── harness.py
│   ├── prompts_10..txt
│   ├── prompts.jsonl
│   ├── ptq_calib.py
│   ├── reports/
│   └── src/
│       └── microbench_gemv.c
├── configs
│   ├── bench.toml
│   └── engine.toml
├── docs
│   ├── Doxyfile
│   ├── DECISIONS.md
│   ├── DESIGN.md
│   ├── adr-00060-optimization-path.md
│   └── doxygen/html/…
├── engine
│   ├── include/         # public headers
│   └── src/             # C/CUDA sources
│       ├── devices/
│       ├── io/
│       ├── kernels/
│       ├── math/
│       ├── opt/
│       ├── quant/
│       └── main_infer.c
├── grafana
│   └── dashboards/clocher.json
├── models
│   └── gpt-oss-20b
│       ├── hf/          # original HF shards
│       ├── model.ie.json
│       └── model.ie.bin
├── monitoring
│   ├── docker-compose.yml
│   └── prometheus.yml
├── quant
│   └── q4_manifest.json
├── scripts
│   ├── hf_to_iebin.py
│   ├── ptq_from_hf.py
│   ├── ptq_from_source.py
│   ├── ptq_from_bin.py
│   ├── true_tps_strict.sh
│   ├── run_benchmark.sh
│   └── make_baseline_md.py
├── tests
│   ├── c/
│   └── python/
├── Makefile
└── README.md

```


## Makefile — Complete Reference

This project uses a single `Makefile` to build CPU/CUDA binaries and run reproducible benchmarks.

### Common Targets
- `make build` — build **CPU** binary at `build/inference-engine`.
- `make build-cuda` — build **CUDA** binary at `build/inference-engine.cuda` (needs NVCC + CUDA toolkit).
- `make clean` — remove build artifacts and reports.
- `make bench` — run CPU benchmark harness (updates `docs/PERFORMANCE.md`).
- `make bench-cuda` — run CUDA benchmark harness (updates `docs/PERFORMANCE.md`).

> Tip: these benchmarks call the *same* CLI under the hood (`build/inference-engine*`).

### Environment Variables (consumed by `make bench*` and/or the CLI)
- `PROMPTS` : path to a prompts file (one prompt per line). Example: `benchmarks/prompts_10..txt`.
- `RUNS` : number of harness repetitions (default: 3).
- `IE_REQUIRE_MODEL` : `1` to enforce strict IEBIN loading (`model.ie.json` + `model.ie.bin`) or `0` to allow stub mode.
- `IE_BYTES_PER_TOKEN` : **bytes touched per generated token** during the work‑touch loop (simulates model working‑set).
- `IE_STRIDE_BYTES` : stride for the work‑touch pointer (default `256`).
- `IE_VERIFY_TOUCH` : `1` to prevent the compiler from optimizing the touch accumulator away.
- `PRECISION` : float precision hint to the CLI (`fp32|bf16|fp16`). (Alias of `IE_PRECISION` when using float modes.)
- `IE_PRECISION` : raw precision label passed to the engine (`fp32|bf16|fp16|int8w|int4|int4w`).
- `THREADS` : CPU threads hint (e.g., `12`).
- `BATCH` : batch size hint (default `1`).

The CLI also accepts explicit flags that benchmarks may forward:
- `--device auto|cpu|cuda|ze` (hint only; selection occurs at build/link)
- `--model-dir PATH` (chdir before loading IEBIN)
- `--model-json PATH`, `--model-bin PATH` (explicit file paths)
- `--pretranspose none|woh|wxh|all`
- `--prefetch on|off|auto|N`
- `--warmup N`
- `--rounds N`
- `--prompts-file PATH`
- `--aggregate`

### End‑to‑End Examples

**CPU, strict mode, 64 MB per token, int4w weights:**
```bash
PROMPTS=benchmarks/prompts_10..txt IE_PRECISION=int4w IE_REQUIRE_MODEL=1 IE_BYTES_PER_TOKEN=64000000 IE_STRIDE_BYTES=256 RUNS=3 make bench
```

**CUDA, same settings:**
```bash
PROMPTS=benchmarks/prompts_10..txt IE_PRECISION=int4w IE_REQUIRE_MODEL=1 IE_BYTES_PER_TOKEN=64000000 IE_STRIDE_BYTES=256 RUNS=3 make bench-cuda
```

**Direct CLI (bypass Makefile), model in `models/gpt-oss-20b`:**
```bash
./build/inference-engine   --model-dir models/gpt-oss-20b   --precision fp32   --pretranspose all   --prompts-file benchmarks/prompts_10..txt   --max-new 128 --threads 12 --rounds 1
```

### Return Codes
- `0` success, JSON line printed.
- `2` bad CLI usage / invalid integer.
- `3` strict IEBIN required but missing/unreadable.
- `5` engine creation failed.
- `6` OOM allocating token buffer.


## RSS Reporting

### RSS Reporting
- We sample **peak resident set size (RSS)** after the measured window to avoid skewing TPS.
- On Linux we prefer `/proc/self/status` → `VmHWM` (kB). Fallback is `getrusage(RUSAGE_SELF).ru_maxrss`.
- On macOS we use `getrusage` where `ru_maxrss` is in **bytes**.
- If neither is available, the sampler returns **0 MB**.


## Update Journal

This running log summarizes meaningful doc/CLI changes for reproducibility.

- **2025-10-24 21:08:40 UTC** — INT4 step added to docs; CLI grew `--device`, `--model-dir`, `--rounds`; strict RSS peak sampler now reads `/proc/self/status` `VmHWM` (Linux) with `getrusage` fallback; `bench`/`bench-cuda` examples updated for **64 MB/token** work‑touch.

