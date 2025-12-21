# Performance Notes

_Last updated: **2025-12-21 15:12:50 UTC**_


**Best true TPS:** **CPU — 34.205**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **256**
- Wall time (Σ): **7.484 s**
- True TPS (Σ tokens / Σ time): **34.205**

## Latency
- p50 (mean across runs): **29.235 ms**
- p95 (mean across runs): **58.471 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4160.000 MB**
- RSS peak (max): **4160.000 MB**
- KV cache: **129 hits / 127 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **16.4 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.19 GB/s**

### Memory Details
- PSS peak (mean / max): **4245.127 MB / 4245.127 MB**
- VMS peak (mean / max): **13127.461 MB / 13127.461 MB**
- RSS floor (mean / max): **27.254 MB / 27.254 MB**
- RSS delta vs baseline (mean / max): **4127.891 MB / 4127.891 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **5.56 % / 5.56 %**
- PSI memory 'full' (mean / max): **5.56 % / 5.56 %**
- System MemAvailable (mean): **4187.2 MB** — **54.2 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **256**
- Wall time (Σ): **8.273 s**
- True TPS (Σ tokens / Σ time): **30.944**

## Latency
- p50 (mean across runs): **32.316 ms**
- p95 (mean across runs): **64.632 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4101.000 MB**
- RSS peak (max): **4101.000 MB**
- KV cache: **129 hits / 127 misses**
- IE_BYTES_PER_TOKEN: **64.0 MB/token**
- Bytes touched (Σ): **16.4 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **1.98 GB/s**

### Memory Details
- PSS peak (mean / max): **4196.646 MB / 4196.646 MB**
- VMS peak (mean / max): **13130.234 MB / 13130.234 MB**
- RSS floor (mean / max): **29.520 MB / 29.520 MB**
- RSS delta vs baseline (mean / max): **4048.340 MB / 4048.340 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **4.04 % / 4.04 %**
- PSI memory 'full' (mean / max): **4.04 % / 4.04 %**
- System MemAvailable (mean): **4182.3 MB** — **54.1 % of MemTotal**

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
- Git commit: **7de6f91 DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 256 | 7.484 | 34.205 | 29.235 | 58.471 | 4160.000 | 4245.127 | 13127.461 | 0 | 0 |
| GPU | 1 | 256 | 8.273 | 30.944 | 32.316 | 64.632 | 4101.000 | 4196.646 | 13130.234 | 0 | 0 |
