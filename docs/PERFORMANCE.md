# Performance Notes

_Last updated: **2025-10-24 20:55:24 UTC**_


**Best true TPS:** **GPU — 34.260**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **11.424 s**
- True TPS (Σ tokens / Σ time): **33.612**

## Latency
- p50 (mean across runs): **29.751 ms**
- p95 (mean across runs): **59.502 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **3079.000 MB**
- RSS peak (max): **3079.000 MB**
- KV cache: **256 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **24.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.15 GB/s**

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **11.208 s**
- True TPS (Σ tokens / Σ time): **34.260**

## Latency
- p50 (mean across runs): **29.188 ms**
- p95 (mean across runs): **58.376 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **3077.000 MB**
- RSS peak (max): **3077.000 MB**
- KV cache: **256 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **24.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.19 GB/s**

## Run Parameters & Conditions
- Engine bin: `/home/ricky/Desktop/Clocher/build/inference-engine`
- Prompts file: `/home/ricky/Desktop/Clocher/benchmarks/prompts_10..txt`
- Threads: **12**
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
- Git commit: **7bc2c9b DIRTY**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | KV hits | KV misses |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------:|----------:|
| CPU | 1 | 384 | 11.424 | 33.612 | 29.751 | 59.502 | 3079.000 | 256 | 128 |
| GPU | 1 | 384 | 11.208 | 34.260 | 29.188 | 58.376 | 3077.000 | 256 | 128 |
