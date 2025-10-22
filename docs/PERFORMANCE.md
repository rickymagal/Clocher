# Performance Notes

_Last updated: **2025-10-22 19:44:13 UTC**_

## Summary (latest benchmark)
- Runs: **5**
- Tokens gerados (Σ): **6400**
- Tempo de parede (Σ): **250.061 s**
- TPS verdadeiro (Σ tokens / Σ tempo): **25.594**

## Latency
- Latency p50 (mean across runs): **38.985 ms**
- Latency p95 (mean across runs): **77.970 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4388 MB**
- RSS peak (max): **4415 MB**
- KV cache: **75 hits / 6325 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB**/token
- Bytes touched (Σ): **400.0 GB**
- Working-set coverage (bytes_per_token / model.bin): **n/a**
- Effective bandwidth: **1.72 GB/s**

## Run Parameters & Conditions
- Engine bin: `/home/ricky/Desktop/inference-engine/build/inference-engine`
- Prompts file: `/home/ricky/Desktop/inference-engine/benchmarks/prompts_10..txt`
- Threads: **12**
- Precision: **fp32**
- Batch: **1**
- Prefetch: **auto**
- Pretranspose: **all**
- Affinity: **auto**
- Max new tokens: **128**
- IE_REQUIRE_MODEL: **1**
- IE_BYTES_PER_TOKEN: **67108864**
- IE_STRIDE_BYTES: **256**
- IE_VERIFY_TOUCH: **1**

## System & Model Info
- CPU: **12th Gen Intel(R) Core(TM) i5-12450H**
- Logical cores: **12**
- RAM (MemTotal): **7.5 GB**
- OS: **Fedora Linux 42 (KDE Plasma Desktop Edition)**
- Kernel: **6.16.9-200.fc42.x86_64-x86_64**
- Git commit: **d86fd2a DIRTY**
- Model dir: `unknown`
- model.ie.json: `unknown` (n/a)
- model.ie.bin: `unknown` (n/a)
- Dtype: **unknown**; Tensors: **0**
