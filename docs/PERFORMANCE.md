# Performance Notes

_Last updated: **2025-11-15 14:58:35 UTC**_


**Best true TPS:** **GPU — 36.264**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **11.360 s**
- True TPS (Σ tokens / Σ time): **33.802**

## Latency
- p50 (mean across runs): **29.584 ms**
- p95 (mean across runs): **59.168 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **5675.000 MB**
- RSS peak (max): **5675.000 MB**
- KV cache: **257 hits / 127 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.27 GB/s**

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.589 s**
- True TPS (Σ tokens / Σ time): **36.264**

## Latency
- p50 (mean across runs): **27.576 ms**
- p95 (mean across runs): **55.152 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **5688.000 MB**
- RSS peak (max): **5688.000 MB**
- KV cache: **257 hits / 127 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.43 GB/s**

### Memory Details
- No extended memory metrics were present in the logs.

## Run Parameters & Conditions
- Engine bin: `/home/ricky/Desktop/Clocher/build/inference-engine`
- Prompts file: `/home/ricky/Desktop/Clocher/benchmarks/prompts_10.txt`
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
- RAM (MemTotal): **8.1 GB**
- OS: **Fedora Linux 42 (KDE Plasma Desktop Edition)**
- Kernel: **6.16.9-200.fc42.x86_64-x86_64**
- Git commit: **7074e7f DIRTY**
- Model file: **models/gpt-oss-20b/model.ie.bin**
- Model size: **13.197 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 384 | 11.360 | 33.802 | 29.584 | 59.168 | 5675.000 | n/a | n/a | 0 | 0 |
| GPU | 1 | 384 | 10.589 | 36.264 | 27.576 | 55.152 | 5688.000 | n/a | n/a | 0 | 0 |
