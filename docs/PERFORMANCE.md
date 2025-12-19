# Performance Notes

_Last updated: **2025-12-19 13:23:19 UTC**_


**Best true TPS:** **CPU — 34.003**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **11.293 s**
- True TPS (Σ tokens / Σ time): **34.003**

## Latency
- p50 (mean across runs): **29.409 ms**
- p95 (mean across runs): **58.819 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2639.333 MB**
- RSS peak (max): **2743.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.28 GB/s**

### Memory Details
- PSS peak (mean / max): **0.449 MB / 0.477 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.836 MB / 2.172 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **3.53 % / 6.12 %**
- PSI memory 'full' (mean / max): **3.53 % / 6.12 %**
- System MemAvailable (mean): **2738.2 MB** — **35.4 % of MemTotal**

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
- OS: **Linux Mint 21.3**
- Kernel: **5.15.0-163-generic-x86_64**
- Git commit: **5813b86 DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **47.827 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.677 | 34.806 | 28.730 | 57.461 | 2515.000 | 0.401 | 4.391 | 0 | 0 |
| CPU | 2 | 128 | 3.677 | 34.814 | 28.724 | 57.449 | 2660.000 | 0.468 | 4.520 | 0 | 0 |
| CPU | 3 | 128 | 3.939 | 32.496 | 30.773 | 61.546 | 2743.000 | 0.477 | 4.520 | 0 | 0 |
