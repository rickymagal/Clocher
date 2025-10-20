# Performance Notes

_Last updated: **2025-10-20 21:46:53 UTC**_

## Summary (latest benchmark)
- Source: `strict` (scripts/true_tps_strict.sh)
- Runs: **1**
- Tokens (total): **1280**
- Wall time (s): **31.604864**
- TPS (true): **40.500**

## Run parameters
- Threads: **12**
- Precision: **fp32**
- Pretranspose: **all**
- Batch: **1**
- Prefetch: **auto**
- Max new tokens: **128**
- Target seconds: **10**
- Prompts file: **benchmarks/prompts_10.txt**
- Affinity policy (CLI): **compact**
- IE_BYTES_PER_TOKEN: **44040192**
- IE_STRIDE_BYTES: **256**
- IE_VERIFY_TOUCH: **1**

## System & Model Info
- CPU: **12th Gen Intel(R) Core(TM) i5-12450H**
- Logical cores: **12**
- RAM (MemTotal): **7.5 GB**
- OS: **Fedora Linux 42 (KDE Plasma Desktop Edition)**
- Kernel: **6.16.9-200.fc42.x86_64**
- Model dir: `/home/ricky/Desktop/inference-engine/models/gpt-oss-20b`
- model.ie.json: `/home/ricky/Desktop/inference-engine/models/gpt-oss-20b/model.ie.json` (126.8 KB, mtime 2025-10-20T16:51:11.780555+00:00)
- model.ie.bin: `/home/ricky/Desktop/inference-engine/models/gpt-oss-20b/model.ie.bin` (77.3 GB, mtime 2025-10-20T16:51:08.307661+00:00)
- vocab.json: `/home/ricky/Desktop/inference-engine/models/gpt-oss-20b/vocab.json`
- Dtype: **float32**; Tensors: **664**

## Profiling Artifacts
- `flamegraph.svg`: **missing**
- `perf.data`: **missing**

## Hot Paths (annotated)
- GEMV (`ie_gemv_f32`): AVX2 microkernel if available; otherwise generic path.
- Activation (`tanh` fast path): clamped polynomial/table approximation.
- Thread pool scheduling: contiguous shard with grainsize control and optional pinning.

## Next optimization actions
- Validate NUMA policy using `scripts/set_numa.sh` (`interleave|node:X|strict`).
- Explore epilogue fusion (bias + activation) on GEMV output.
- Extend blocked-K packing and tune prefetch distances based on the flamegraph.
