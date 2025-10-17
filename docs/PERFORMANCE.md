# Performance Notes

_Last updated: **2025-10-17 16:15:17 UTC**_

## Summary (latest run)
- Reports directory: `(no reports found)`
- TPS (true): **3111.501**
- Latency p50: **0.321 ms**
- Latency p95: **0.450 ms**

## Run parameters
- Threads: **12**
- Precision: **fp32**
- Pretranspose: **none**
- Device: **cpu**
- Batch: **1**
- Prefetch: **auto**
- Warmup: **0**
- Prompts file: **n/a**
- Affinity policy (CLI): **auto**
- Affinity env toggle (`IE_TP_USE_AFFINITY`): **disabled**
- Detected CPU features: **avx2, fma, sse4.2**

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
