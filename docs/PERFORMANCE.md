# Performance Notes

_Last updated: **2025-12-20 18:24:17 UTC**_


**Best true TPS:** **CPU — 38.120**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.073 s**
- True TPS (Σ tokens / Σ time): **38.120**

## Latency
- p50 (mean across runs): **26.233 ms**
- p95 (mean across runs): **52.466 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4407.000 MB**
- RSS peak (max): **4436.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.56 GB/s**

### Memory Details
- PSS peak (mean / max): **0.512 MB / 0.537 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.751 MB / 2.066 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **6.12 % / 6.79 %**
- PSI memory 'full' (mean / max): **6.07 % / 6.76 %**
- System MemAvailable (mean): **4445.3 MB** — **57.5 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.400 s**
- True TPS (Σ tokens / Σ time): **36.924**

## Latency
- p50 (mean across runs): **27.082 ms**
- p95 (mean across runs): **54.165 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **3815.667 MB**
- RSS peak (max): **3853.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.48 GB/s**

### Memory Details
- PSS peak (mean / max): **0.528 MB / 0.556 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.836 MB / 2.164 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **6.23 % / 7.70 %**
- PSI memory 'full' (mean / max): **6.23 % / 7.70 %**
- System MemAvailable (mean): **3924.4 MB** — **50.8 % of MemTotal**

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
- Git commit: **0edbb92 DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.470 | 36.891 | 27.107 | 54.214 | 4369.000 | 0.468 | 4.391 | 0 | 0 |
| CPU | 2 | 128 | 3.310 | 38.666 | 25.863 | 51.726 | 4416.000 | 0.531 | 4.520 | 0 | 0 |
| CPU | 3 | 128 | 3.293 | 38.866 | 25.730 | 51.459 | 4436.000 | 0.537 | 4.520 | 0 | 0 |
| GPU | 1 | 128 | 3.461 | 36.981 | 27.041 | 54.081 | 3785.000 | 0.481 | 4.391 | 0 | 0 |
| GPU | 2 | 128 | 3.466 | 36.928 | 27.079 | 54.159 | 3809.000 | 0.546 | 4.520 | 0 | 0 |
| GPU | 3 | 128 | 3.472 | 36.864 | 27.127 | 54.254 | 3853.000 | 0.556 | 4.520 | 0 | 0 |
