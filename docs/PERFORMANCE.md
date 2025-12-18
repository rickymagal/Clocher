# Performance Notes

_Last updated: **2025-12-18 20:02:38 UTC**_


**Best true TPS:** **CPU — 40.448**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **9.494 s**
- True TPS (Σ tokens / Σ time): **40.448**

## Latency
- p50 (mean across runs): **24.723 ms**
- p95 (mean across runs): **49.446 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2468.667 MB**
- RSS peak (max): **2577.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.71 GB/s**

### Memory Details
- PSS peak (mean / max): **0.521 MB / 0.545 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.818 MB / 2.137 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **3.84 % / 6.45 %**
- PSI memory 'full' (mean / max): **3.84 % / 6.45 %**
- System MemAvailable (mean): **2820.0 MB** — **36.5 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **9.756 s**
- True TPS (Σ tokens / Σ time): **39.359**

## Latency
- p50 (mean across runs): **25.407 ms**
- p95 (mean across runs): **50.814 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2720.333 MB**
- RSS peak (max): **2744.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.64 GB/s**

### Memory Details
- PSS peak (mean / max): **0.520 MB / 0.543 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.857 MB / 2.195 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **3.35 % / 5.93 %**
- PSI memory 'full' (mean / max): **3.35 % / 5.93 %**
- System MemAvailable (mean): **2907.5 MB** — **37.6 % of MemTotal**

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
- Git commit: **8330101 DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **47.827 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.095 | 41.357 | 24.180 | 48.359 | 2274.000 | 0.477 | 4.391 | 0 | 0 |
| CPU | 2 | 128 | 3.245 | 39.445 | 25.352 | 50.703 | 2555.000 | 0.540 | 4.520 | 0 | 0 |
| CPU | 3 | 128 | 3.154 | 40.587 | 24.638 | 49.276 | 2577.000 | 0.545 | 4.520 | 0 | 0 |
| GPU | 1 | 128 | 3.352 | 38.187 | 26.187 | 52.374 | 2679.000 | 0.475 | 4.391 | 0 | 0 |
| GPU | 2 | 128 | 3.242 | 39.486 | 25.326 | 50.651 | 2738.000 | 0.542 | 4.520 | 0 | 0 |
| GPU | 3 | 128 | 3.163 | 40.471 | 24.709 | 49.418 | 2744.000 | 0.543 | 4.520 | 0 | 0 |
