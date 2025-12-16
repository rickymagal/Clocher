# Performance Notes

_Last updated: **2025-12-16 16:17:06 UTC**_


**Best true TPS:** **CPU — 38.953**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **9.858 s**
- True TPS (Σ tokens / Σ time): **38.953**

## Latency
- p50 (mean across runs): **25.672 ms**
- p95 (mean across runs): **51.344 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2572.333 MB**
- RSS peak (max): **2612.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.61 GB/s**

### Memory Details
- PSS peak (mean / max): **0.516 MB / 0.546 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.840 MB / 2.168 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **2.99 % / 5.16 %**
- PSI memory 'full' (mean / max): **2.99 % / 5.16 %**
- System MemAvailable (mean): **3096.0 MB** — **40.1 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.539 s**
- True TPS (Σ tokens / Σ time): **36.437**

## Latency
- p50 (mean across runs): **27.444 ms**
- p95 (mean across runs): **54.889 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2869.333 MB**
- RSS peak (max): **3023.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.45 GB/s**

### Memory Details
- PSS peak (mean / max): **0.500 MB / 0.525 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.798 MB / 2.105 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **2.69 % / 4.00 %**
- PSI memory 'full' (mean / max): **2.69 % / 4.00 %**
- System MemAvailable (mean): **3074.3 MB** — **39.8 % of MemTotal**

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
- Git commit: **6473c9e DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **47.827 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.166 | 40.430 | 24.734 | 49.468 | 2513.000 | 0.471 | 4.391 | 0 | 0 |
| CPU | 2 | 128 | 3.338 | 38.346 | 26.078 | 52.156 | 2592.000 | 0.531 | 4.520 | 0 | 0 |
| CPU | 3 | 128 | 3.354 | 38.163 | 26.203 | 52.407 | 2612.000 | 0.546 | 4.520 | 0 | 0 |
| GPU | 1 | 128 | 3.613 | 35.424 | 28.229 | 56.458 | 2636.000 | 0.454 | 4.391 | 0 | 0 |
| GPU | 2 | 128 | 3.521 | 36.349 | 27.511 | 55.023 | 2949.000 | 0.521 | 4.520 | 0 | 0 |
| GPU | 3 | 128 | 3.404 | 37.604 | 26.593 | 53.185 | 3023.000 | 0.525 | 4.520 | 0 | 0 |
