# Performance Notes

_Last updated: **2025-12-22 20:04:06 UTC**_


**Best true TPS:** **CPU — 40.488**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **1152**
- Wall time (Σ): **28.453 s**
- True TPS (Σ tokens / Σ time): **40.488**

## Latency
- p50 (mean across runs): **24.699 ms**
- p95 (mean across runs): **49.398 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4846.667 MB**
- RSS peak (max): **4860.000 MB**
- KV cache: **768 hits / 384 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **73.7 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.59 GB/s**

### Memory Details
- PSS peak (mean / max): **4961.919 MB / 4981.277 MB**
- VMS peak (mean / max): **13127.379 MB / 13127.379 MB**
- RSS floor (mean / max): **19.917 MB / 20.504 MB**
- RSS delta vs baseline (mean / max): **4819.602 MB / 4824.156 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **6.81 % / 9.87 %**
- PSI memory 'full' (mean / max): **6.80 % / 9.85 %**
- System MemAvailable (mean): **4811.1 MB** — **62.2 % of MemTotal**

## Run Parameters & Conditions
- Engine bin: `/home/ricardomag/Desktop/Clocher/build/inference-engine`
- Prompts file: `/home/ricardomag/Desktop/Clocher/benchmarks/prompts_10.txt`
- Threads: **12**
- Precision: **int4w**
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
- OS: **KDE neon User Edition**
- Kernel: **6.14.0-37-generic-x86_64**
- Git commit: **5f5d27f DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 384 | 9.546 | 40.227 | 24.859 | 49.717 | 4860.000 | 4959.637 | 13127.379 | 0 | 0 |
| CPU | 2 | 384 | 9.553 | 40.199 | 24.876 | 49.753 | 4841.000 | 4981.277 | 13127.379 | 0 | 0 |
| CPU | 3 | 384 | 9.355 | 41.049 | 24.361 | 48.723 | 4839.000 | 4944.843 | 13127.379 | 0 | 0 |
