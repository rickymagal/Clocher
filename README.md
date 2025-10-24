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

