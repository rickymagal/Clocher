# Performance Notes

_Last updated: **2025-10-17 18:27:45 UTC**_

## Summary (latest run)
- Reports directory: `benchmarks/reports/20251017_182252`
- TPS (true): **n/a**
- Latency p50: **n/a**
- Latency p95: **n/a**

## Run parameters
- Threads: **4**
- Precision: **fp32**
- Pretranspose: **none**
- Affinity policy (CLI): **n/a**
- Affinity env toggle (`IE_TP_USE_AFFINITY`): **n/a**
- Detected CPU features: **n/a**

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
