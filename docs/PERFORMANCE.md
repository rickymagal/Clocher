# Performance Notes

_Last updated: **2025-10-17 23:12:05 UTC**_

## Summary (latest run)
- Reports directory: `/home/ricky/Desktop/inference-engine/benchmarks/reports/20251017_231204`
- TPS (true): **15016.424**
- Latency p50: **0.067**
- Latency p95: **0.067**

## Run parameters
- Threads: **4**
- Precision: **fp32**
- Pretranspose: **none**
- Affinity policy (CLI): **auto**

## Profiling Artifacts
- `flamegraph.svg`: **absent**
- `perf.data`: **absent**

## Hot Paths (annotated)
- GEMV (`ie_gemv_f32`): AVX2 microkernel if available; otherwise generic path.
- Activation (`tanh` fast path): clamped polynomial/table approximation.
- Thread pool scheduling: contiguous shard with grainsize control and optional pinning.

## Next optimization actions
- Validate NUMA policy impacts using `scripts/set_numa.sh` (`interleave|node:X|strict`).
- Explore epilogue fusion (bias + activation) in GEMV output.
- Extend blocked-K packing and prefetch distances based on flamegraph evidence.
