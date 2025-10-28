# Performance Notes

_Last updated: **2025-10-28 14:12:59 UTC**_


**Best true TPS:** **GPU — 35.239**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **11.018 s**
- True TPS (Σ tokens / Σ time): **34.852**

## Latency
- p50 (mean across runs): **28.692 ms**
- p95 (mean across runs): **57.385 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4391.000 MB**
- RSS peak (max): **4391.000 MB**
- KV cache: **256 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **24.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.23 GB/s**

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.897 s**
- True TPS (Σ tokens / Σ time): **35.239**

## Latency
- p50 (mean across runs): **28.378 ms**
- p95 (mean across runs): **56.756 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4392.000 MB**
- RSS peak (max): **4392.000 MB**
- KV cache: **256 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **24.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.26 GB/s**

## Run Parameters & Conditions
- Engine bin: `/home/ricky/Desktop/Clocher/build/inference-engine`
- Prompts file: `/home/ricky/Desktop/Clocher/benchmarks/prompts_10..txt`
- Threads: **1**
- Precision: **fp32**
- Batch: **1**
- Prefetch: **auto**
- Pretranspose: **all**
- Affinity: **auto**
- Max new tokens: **128**
- IE_REQUIRE_MODEL: **1**
- IE_BYTES_PER_TOKEN: **64000000**
- IE_STRIDE_BYTES: **256**
- IE_VERIFY_TOUCH: **1**

## System & Model Info
- CPU: **12th Gen Intel(R) Core(TM) i5-12450H**
- Logical cores: **12**
- RAM (MemTotal): **8.1 GB**
- OS: **Fedora Linux 42 (KDE Plasma Desktop Edition)**
- Kernel: **6.16.9-200.fc42.x86_64-x86_64**
- Git commit: **6d2f57e DIRTY**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | KV hits | KV misses |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------:|----------:|
| CPU | 1 | 384 | 11.018 | 34.852 | 28.692 | 57.385 | 4391.000 | 256 | 128 |
| GPU | 1 | 384 | 10.897 | 35.239 | 28.378 | 56.756 | 4392.000 | 256 | 128 |
