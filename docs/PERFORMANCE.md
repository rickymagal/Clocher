# Performance Notes

_Last updated: **2025-10-17 23:57:44 UTC**_

## Summary (latest run)
- Reports directory: `/home/ricky/Desktop/inference-engine/benchmarks/reports/20251017_235743`
- TPS (true): **426666.667**
- Latency p50: **0.002**
- Latency p95: **0.005**

## Run parameters
- Threads: **4**
- Precision: **fp32**
- Pretranspose: **none**
- Affinity policy (CLI): **auto**

## Profiling Artifacts
- `flamegraph.svg`: **present**
- `perf.data`: **present**

## Hot Paths (annotated)
- GEMV (`ie_gemv_f32`): AVX2 microkernel if available; otherwise generic path.
- Activation (`tanh` fast path): clamped polynomial/table approximation.
- Thread pool scheduling: contiguous shard with grainsize control and optional pinning.

## Next optimization actions
- Validate NUMA policy impacts using `scripts/set_numa.sh` (`interleave|node:X|strict`).
- Explore epilogue fusion (bias + activation) in GEMV output.
- Extend blocked-K packing and prefetch distances based on flamegraph evidence.
