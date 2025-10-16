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
