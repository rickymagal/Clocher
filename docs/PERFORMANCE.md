# Performance Notes

_Last updated: **2025-12-15 23:57:51 UTC**_


**Best true TPS:** **GPU — 37.959**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.527 s**
- True TPS (Σ tokens / Σ time): **36.477**

## Latency
- p50 (mean across runs): **27.415 ms**
- p95 (mean across runs): **54.829 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4051.000 MB**
- RSS peak (max): **4250.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.45 GB/s**

### Memory Details
- PSS peak (mean / max): **0.508 MB / 0.532 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.803 MB / 2.141 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **2.71 % / 4.46 %**
- PSI memory 'full' (mean / max): **2.71 % / 4.46 %**
- System MemAvailable (mean): **4562.4 MB** — **59.0 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.116 s**
- True TPS (Σ tokens / Σ time): **37.959**

## Latency
- p50 (mean across runs): **26.344 ms**
- p95 (mean across runs): **52.688 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4403.000 MB**
- RSS peak (max): **4481.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.55 GB/s**

### Memory Details
- PSS peak (mean / max): **0.501 MB / 0.526 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.539 MB / 1.930 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **4.02 % / 5.16 %**
- PSI memory 'full' (mean / max): **4.02 % / 5.16 %**
- System MemAvailable (mean): **4735.5 MB** — **61.3 % of MemTotal**

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
- Git commit: **1a16d09 DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **47.827 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.819 | 33.515 | 29.837 | 59.674 | 3699.000 | 0.462 | 4.391 | 0 | 0 |
| CPU | 2 | 128 | 3.694 | 34.650 | 28.860 | 57.720 | 4204.000 | 0.529 | 4.520 | 0 | 0 |
| CPU | 3 | 128 | 3.014 | 42.469 | 23.547 | 47.093 | 4250.000 | 0.532 | 4.520 | 0 | 0 |
| GPU | 1 | 128 | 3.275 | 39.090 | 25.582 | 51.165 | 4317.000 | 0.452 | 4.391 | 0 | 0 |
| GPU | 2 | 128 | 3.221 | 39.744 | 25.161 | 50.322 | 4411.000 | 0.523 | 4.520 | 0 | 0 |
| GPU | 3 | 128 | 3.621 | 35.350 | 28.289 | 56.577 | 4481.000 | 0.526 | 4.520 | 0 | 0 |
