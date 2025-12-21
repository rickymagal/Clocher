# Performance Notes

_Last updated: **2025-12-21 12:26:31 UTC**_


**Best true TPS:** **CPU — 33.717**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **128**
- Wall time (Σ): **3.796 s**
- True TPS (Σ tokens / Σ time): **33.717**

## Latency
- p50 (mean across runs): **29.659 ms**
- p95 (mean across runs): **59.318 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4629.000 MB**
- RSS peak (max): **4629.000 MB**
- KV cache: **1 hits / 127 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **8.2 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.16 GB/s**

### Memory Details
- PSS peak (mean / max): **0.472 MB / 0.472 MB**
- VMS peak (mean / max): **4.391 MB / 4.391 MB**
- RSS floor (mean / max): **1.922 MB / 1.922 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **1.91 % / 1.91 %**
- PSI memory 'full' (mean / max): **1.91 % / 1.91 %**
- System MemAvailable (mean): **4634.5 MB** — **60.0 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **128**
- Wall time (Σ): **3.986 s**
- True TPS (Σ tokens / Σ time): **32.110**

## Latency
- p50 (mean across runs): **31.143 ms**
- p95 (mean across runs): **62.286 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4636.000 MB**
- RSS peak (max): **4636.000 MB**
- KV cache: **1 hits / 127 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **8.2 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.06 GB/s**

### Memory Details
- PSS peak (mean / max): **0.472 MB / 0.472 MB**
- VMS peak (mean / max): **4.391 MB / 4.391 MB**
- RSS floor (mean / max): **1.984 MB / 1.984 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **1.74 % / 1.74 %**
- PSI memory 'full' (mean / max): **1.74 % / 1.74 %**
- System MemAvailable (mean): **4688.6 MB** — **60.7 % of MemTotal**

## Run Parameters & Conditions
- Engine bin: `/home/ricky/Desktop/Clocher/build/inference-engine`
- Prompts file: `/home/ricky/Desktop/Clocher/benchmarks/prompts_10.txt`
- Threads: **16**
- Precision: **int4**
- Batch: **1**
- Prefetch: **auto**
- Pretranspose: **all**
- Affinity: **scatter**
- Max new tokens: **128**
- IE_REQUIRE_MODEL: **1**
- IE_BYTES_PER_TOKEN: **64000000**
- IE_STRIDE_BYTES: **256**
- IE_VERIFY_TOUCH: **1**

## System & Model Info
- CPU: **12th Gen Intel(R) Core(TM) i5-12450H**
- Logical cores: **12**
- RAM (MemTotal): **8.1 GB**
- OS: **Linux Mint 21.3**
- Kernel: **5.15.0-163-generic-x86_64**
- Git commit: **8dd5437 DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.796 | 33.717 | 29.659 | 59.318 | 4629.000 | 0.472 | 4.391 | 0 | 0 |
| GPU | 1 | 128 | 3.986 | 32.110 | 31.143 | 62.286 | 4636.000 | 0.472 | 4.391 | 0 | 0 |
