# Performance Notes

_Last updated: **2025-12-21 10:45:20 UTC**_


**Best true TPS:** **CPU — 33.808**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **128**
- Wall time (Σ): **3.786 s**
- True TPS (Σ tokens / Σ time): **33.808**

## Latency
- p50 (mean across runs): **29.579 ms**
- p95 (mean across runs): **59.158 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **3836.000 MB**
- RSS peak (max): **3836.000 MB**
- KV cache: **0 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **8.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.27 GB/s**

### Memory Details
- PSS peak (mean / max): **0.475 MB / 0.475 MB**
- VMS peak (mean / max): **4.391 MB / 4.391 MB**
- RSS floor (mean / max): **1.996 MB / 1.996 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **3.42 % / 3.42 %**
- PSI memory 'full' (mean / max): **3.42 % / 3.42 %**
- System MemAvailable (mean): **3897.9 MB** — **50.4 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **128**
- Wall time (Σ): **3.983 s**
- True TPS (Σ tokens / Σ time): **32.133**

## Latency
- p50 (mean across runs): **31.121 ms**
- p95 (mean across runs): **62.241 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4055.000 MB**
- RSS peak (max): **4055.000 MB**
- KV cache: **0 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **8.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.16 GB/s**

### Memory Details
- PSS peak (mean / max): **0.478 MB / 0.478 MB**
- VMS peak (mean / max): **4.391 MB / 4.391 MB**
- RSS floor (mean / max): **2.000 MB / 2.000 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **4.67 % / 4.67 %**
- PSI memory 'full' (mean / max): **4.61 % / 4.61 %**
- System MemAvailable (mean): **4103.6 MB** — **53.1 % of MemTotal**

## Run Parameters & Conditions
- Engine bin: `/home/ricky/Desktop/Clocher/build/inference-engine`
- Prompts file: `/home/ricky/Desktop/Clocher/benchmarks/prompts_10.txt`
- Threads: **12**
- Precision: **int4**
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
- Git commit: **6add236 DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.786 | 33.808 | 29.579 | 59.158 | 3836.000 | 0.475 | 4.391 | 0 | 0 |
| GPU | 1 | 128 | 3.983 | 32.133 | 31.121 | 62.241 | 4055.000 | 0.478 | 4.391 | 0 | 0 |
