# Performance Notes

_Last updated: **2025-10-22 18:07:34 UTC**_

## Summary (latest benchmark)
- Runs: **5**
- Tokens gerados (Σ): **6400**
- Tempo de parede (Σ): **257.886 s**
- TPS verdadeiro (Σ tokens / Σ tempo): **24.817**

## Latency
- Latency p50 (mean across runs): **40.207 ms**
- Latency p95 (mean across runs): **80.415 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4492 MB**
- RSS peak (max): **4517 MB**
- KV cache: **0 hits / 0 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB**/token
- Bytes touched (Σ): **400.0 GB**
- Working-set coverage (bytes_per_token / model.bin): **n/a**
- Effective bandwidth: **1.67 GB/s**

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
