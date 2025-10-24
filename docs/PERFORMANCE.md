# Performance Notes

_Last updated: **2025-10-24 20:37:29 UTC**_


**Best true TPS:** **GPU — 33.935**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **11.700 s**
- True TPS (Σ tokens / Σ time): **32.821**

## Latency
- p50 (mean across runs): **30.469 ms**
- p95 (mean across runs): **60.937 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **0.000 MB**
- RSS peak (max): **0.000 MB**
- KV cache: **256 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **24.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.10 GB/s**

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **11.316 s**
- True TPS (Σ tokens / Σ time): **33.935**

## Latency
- p50 (mean across runs): **29.468 ms**
- p95 (mean across runs): **58.936 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **0.000 MB**
- RSS peak (max): **0.000 MB**
- KV cache: **256 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **24.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.17 GB/s**

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
- Git commit: **a37e750 DIRTY**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | KV hits | KV misses |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------:|----------:|
| CPU | 1 | 384 | 11.700 | 32.821 | 30.469 | 60.937 | 0.000 | 256 | 128 |
| GPU | 1 | 384 | 11.316 | 33.935 | 29.468 | 58.936 | 0.000 | 256 | 128 |
