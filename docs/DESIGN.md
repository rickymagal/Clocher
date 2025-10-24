# Design (CPU baseline + INT4 path)

This document describes the hot path, API boundaries, and precision modes.  
**Last updated:** 2025-10-24 21:00:48 UTC

## Process and boundaries
- Single binary `inference-engine`: create → generate → collect metrics → destroy.
- CLI prints exactly one JSON line per run so the Python harness can ingest results.
- No third‑party runtime dependencies; only `pthread` and `libm` (and CUDA for the GPU build).

## API surface (high level)
- `ie_engine_create(cfg)` → initializes state (weights, buffers, thread‑pool).
- `ie_engine_generate(prompt, max_new, params)` → produces tokens and updates metrics rings.
- `ie_engine_metrics(out)` → returns latency p50/p95, true TPS, **RSS peak**, KV hits/misses.
- `ie_engine_destroy()` → frees all resources.

## Hot path layout
- GEMV microkernel:
  - Generic scalar reference implementation.
  - AVX2/FMA path with light prefetch and blocked‑K packing.
- Activation:
  - `tanh` fast path with clamp (accuracy‑bounded), vector helper.
  - Optional fused bias + activation to reduce memory traffic.

## Precision modes

### Floating point
- FP32 baseline, optional BF16/FP16 round‑trip (accumulate FP32).

### INT8 PTQ (reference)
- Per‑tensor/per‑row scales (min‑max); (de)quant helpers; task gate in scripts.

### **NEW — INT4 PTQ (weight‑only)**
- **Format**: weights are **nibble‑packed** (2 values per byte). Each row (or group) carries a scale (and optional zero‑point if affine).
- **Packing**: INT4 packing integrates with the existing **pretranspose / blocked‑K** pipeline. Packing order: float → (optional) pretranspose → quantize to int4 → pack.
- **Dequantization**: fused in the matmul path; scale is broadcast per row (or group) to recover FP32 accumulators.
- **Manifests**: a `q4_manifest.json` enumerates which tensors use INT4 and their scale metadata. `scripts/hf_to_iebin.py` consumes this manifest via `--q4-map` to emit `model.ie.bin`/`.json`.
- **Selection**: at runtime choose `IE_PRECISION=int4w` (or `--precision int4w`), leaving activations in float. This is **bandwidth‑oriented** and preserves compute simplicity.

## Threading model
- Fixed thread‑pool over `pthread`, contiguous sharding, grainsize control.
- Affinity (Linux): `IE_TP_USE_AFFINITY=1` enables `auto|compact|scatter`.
- NUMA:
  - `scripts/set_numa.sh` can set OS policy (`interleave|node:X|strict`).
  - In‑repo probe reads `/sys/devices/system/node/online` to annotate reports.

## Layout and caching
- Blocked‑K packing with optional on‑disk caching (content‑addressed by shape + block size).
- CLI flag `--pretranspose` controls packing scope (`none|woh|wxh|all`). INT4 can be cached per layout variant.

## Metrics
- Per‑token latency ring (p50/p95).
- True TPS (`generated_tokens / wall_time_s`).
- **Peak RSS** (Linux `/proc/self/status:VmHWM` → MiB; fallback `getrusage`).
- KV hits/misses counter stubs aggregated per round.

## GPU integration (CUDA path)
- Device layer: `engine/src/devices/ie_device_cuda.cu`, kernels in `engine/src/kernels/ie_kernels_cuda.cu`.
- Build: `make build-cuda` → `build/inference-engine.cuda`.
- INT4 weight‑only support mirrors the CPU path; packing and scales are shared in `model.ie.json`.


---
## INT4 Weight-Only Path — Design Addendum (2025-10-24 21:04:23 UTC)

### Goals
- Introduce an **optional** path for *weight-only* INT4 (`int4w`) that:
  1) Packs HF weights into IEBIN using a **manifest-driven** policy.
  2) Leaves tokenization, shapes, and scheduling untouched.
  3) Preserves benchmark comparability via *work-touch* instrumentation.

### Design Choices
- **Manifest-driven packing**: `--q4-map` lets us target only GEMV-heavy matrices (attn/MLP projections) and keep sensitive tensors (embeddings, layernorms) in FP formats.
- **Separation of concerns**:
  - Storage precision (`IE_PRECISION`) is passed through `ie_engine_params_t::precision` untouched.
  - Host math precision (FP32/BF16/FP16) remains a separate selection.
- **Compat with harness**: The CLI accepts soft hints (`--device`, `--model-dir`, `--rounds`) so existing scripts don’t break.

### Data Flow (INT4 path)
HF shards → `hf_to_iebin.py --q4-map` → IEBIN (`model.ie.json` + `model.ie.bin` with INT4-packed tensors) → Runtime `IE_PRECISION=int4w` → GEMV kernels read packed weights (or dequant on load), semantics unchanged.

### Metrics Integrity
- The timed window is **only**: `ie_engine_generate(...)` + optional *work-touch* loop.
- RSS peak and KV counters are sampled **after** the window to avoid skewing TPS.

---

## Appendix — INT4 (Weight‑Only) Step (Summary)
- Convert HF shards → IEBIN with an INT4 manifest:
  ```bash
  python3 scripts/hf_to_iebin.py     --hf-dir models/gpt-oss-20b/hf     --out-dir models/gpt-oss-20b     --q4-map quant/q4_manifest.json
  ```
- Run benchmarks in strict mode with a **64 MB/token** work‑touch:
  ```bash
  PROMPTS=benchmarks/prompts_10..txt   IE_PRECISION=int4w IE_REQUIRE_MODEL=1   IE_BYTES_PER_TOKEN=64000000 IE_STRIDE_BYTES=256 RUNS=3   make bench           # or: make bench-cuda
  ```
- Precision hints: `PRECISION=fp32` (activates float path) and `IE_PRECISION=int4w` (weight‑only path).
