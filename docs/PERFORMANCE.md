# Performance Notes

_Last updated: **2025-10-24 18:03:52 UTC**_


**Best true TPS:** **GPU — 36.789**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **3840**
- Wall time (Σ): **130.464 s**
- True TPS (Σ tokens / Σ time): **29.433**

## Latency
- p50 (mean across runs): **33.890 ms**
- p95 (mean across runs): **67.780 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4671.333 MB**
- RSS peak (max): **4681.000 MB**
- KV cache: **45 hits / 3795 misses**
- IE_BYTES_PER_TOKEN: **57.7 MB/token**
- Bytes touched (Σ): **221.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **1.70 GB/s**

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **3840**
- Wall time (Σ): **104.379 s**
- True TPS (Σ tokens / Σ time): **36.789**

## Latency
- p50 (mean across runs): **27.080 ms**
- p95 (mean across runs): **54.160 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4685.333 MB**
- RSS peak (max): **4694.000 MB**
- KV cache: **45 hits / 3795 misses**
- IE_BYTES_PER_TOKEN: **57.7 MB/token**
- Bytes touched (Σ): **221.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.12 GB/s**

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
- IE_BYTES_PER_TOKEN: **57720256**
- IE_STRIDE_BYTES: **256**
- IE_VERIFY_TOUCH: **1**

## System & Model Info
- CPU: **12th Gen Intel(R) Core(TM) i5-12450H**
- Logical cores: **12**
- RAM (MemTotal): **8.1 GB**
- OS: **Fedora Linux 42 (KDE Plasma Desktop Edition)**
- Kernel: **6.16.9-200.fc42.x86_64-x86_64**
- Git commit: **9f9e8fc DIRTY**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | KV hits | KV misses |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------:|----------:|
| CPU | 1 | 1280 | 42.858 | 29.866 | 33.397 | 66.795 | 4681.000 | 15 | 1265 |
| CPU | 2 | 1280 | 45.567 | 28.090 | 35.516 | 71.032 | 4653.000 | 15 | 1265 |
| CPU | 3 | 1280 | 42.038 | 30.448 | 32.757 | 65.513 | 4680.000 | 15 | 1265 |
| GPU | 1 | 1280 | 34.229 | 37.395 | 26.650 | 53.301 | 4692.000 | 15 | 1265 |
| GPU | 2 | 1280 | 35.250 | 36.312 | 27.449 | 54.898 | 4694.000 | 15 | 1265 |
| GPU | 3 | 1280 | 34.901 | 36.676 | 27.140 | 54.280 | 4670.000 | 15 | 1265 |
