# Design (CPU baseline)

## Process and boundaries
- Single binary `inference-engine`: create → generate → collect metrics → destroy.
- CLI prints exactly one JSON line per run so the Python harness can ingest results.
- No third-party runtime dependencies; only `pthread` and `libm`.

## API surface (high level)
- `ie_engine_create(cfg)` → initializes state (weights, buffers, thread-pool).
- `ie_engine_generate(prompt, max_new, params)` → produces tokens and updates metrics rings.
- `ie_engine_get_metrics()` → returns p50/p95, TPS, RSS peak, KV hits/misses.
- `ie_engine_destroy()` → frees all resources.

## Hot path layout
- GEMV microkernel:
  - Generic scalar reference implementation
  - AVX2/FMA path with light prefetch and blocked-K packing
- Activation:
  - `tanh` fast path with clamp (accuracy-bounded), vector helper
  - Optional fused bias + activation to reduce memory traffic
- Precision:
  - FP32 baseline, optional BF16/FP16 round-trip (accumulate FP32)
  - INT8 PTQ with per-tensor/per-row scales (min-max); (de)quant helpers

## Threading model
- Fixed thread-pool over `pthread`, contiguous sharding, grainsize control.
- Affinity (Linux): `IE_TP_USE_AFFINITY=1` enables `auto|compact|scatter`.
- NUMA:
  - External helper `scripts/set_numa.sh` to set OS policy (`interleave|node:X|strict`).
  - In-repo probe reads `/sys/devices/system/node/online` to annotate reports.

## Layout and caching
- Blocked-K packing with optional on-disk caching (content-addressed by shape + block size).
- CLI flag `--pretranspose` controls packing scope (`none|woh|wxh|all`).

## Metrics
- Per-token latency ring (p50/p95).
- True TPS (`generated_tokens / wall_time_s`).
- Optional RSS peak, KV hits/misses counter stubs.

## Testing strategy
- C unit tests for tensor ops, kernels, math, thread-pool, tokenizer, weights, quant.
- Python tests for CLI, harness, metrics integrity, PTQ pipeline.
- Microbenchmarks for GEMV and math kernels.

## Tooling
- Makefile orchestrates build, test, bench, profile, docs.
- Doxygen for C API reference.
- `perf + FlameGraph` → `flamegraph.svg` and `PERFORMANCE.md`.
## Updates — 2025-10-17

### I/O Batching and Prefetch
- Introduced a multi-worker **prefetch + tokenization batcher** feeding a ring buffer.
- **Order preservation:** despite parallel tokenization, items are consumed in input order via indexed enqueue and contiguous micro-batch views.
- **Backpressure:** producer blocks when the ring is full; consumer signals on advance; micro-batch size is tunable.
- **Failure semantics:** tokenization status is carried per item; consumers free payloads on advance.

### CLI & Pipeline Integration
- New CLI knobs: `--prompts-file`, `--batch`, `--prefetch`, `--warmup`.
- Warmup performs a small generation before timed runs to stabilize caches.
- JSON output is a single stable line for automation with predictable field names and spacing.

---

## GPU integration & performance reporting — 2025-10-24 00:42:21 UTC

### CUDA-only GPU path
- Device layer: `engine/src/devices/ie_device_cuda.cu` with kernels in `engine/src/kernels/ie_kernels_cuda.cu`.
- Build: `make build-cuda` → `build/inference-engine.cuda`. CPU build remains `build/inference-engine`.
- Runtime selection: the benchmark harness sets `DEVICE=cuda` for GPU runs; CPU runs leave device unset (`cpu`).

### Reporting pipeline (last-3 policy)
- The CLI still emits **one JSON line per run** with stable fields.
- `scripts/update_performance_md.py` merges the **last 3 runs per device** (CPU and GPU) into one `docs/PERFORMANCE.md`:
  - Per-device **Summary**, **Latency**, and **Spatial Complexity**.
  - **Run Parameters & Conditions** reproduced from the environment (engine path, threads, precision, batch, prefetch, pretranspose, affinity, max-new, and the **IE_* knobs**).
  - A **Best true TPS** banner showing the overall best across devices.

### Doxygen coverage
- Treat `.cu` as C++ and `.cl` as C via `EXTENSION_MAPPING` so cross-references work.
- Keep function contracts short and specific; avoid duplicating host-side comments inside kernels.
