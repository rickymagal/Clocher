# Performance Notes

_Last updated: **2025-12-21 10:12:06 UTC**_


**Best true TPS:** **GPU — 35.337**.

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.867 s**
- True TPS (Σ tokens / Σ time): **35.337**

## Latency
- p50 (mean across runs): **28.299 ms**
- p95 (mean across runs): **56.597 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2847.333 MB**
- RSS peak (max): **3028.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.37 GB/s**

### Memory Details
- PSS peak (mean / max): **0.519 MB / 0.545 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.589 MB / 1.949 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **4.15 % / 6.69 %**
- PSI memory 'full' (mean / max): **4.08 % / 6.52 %**
- System MemAvailable (mean): **3022.4 MB** — **39.1 % of MemTotal**

## Run Parameters & Conditions
- Engine bin: `/home/ricky/Desktop/Clocher/build/inference-engine.cuda`
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
- Git commit: **b13da1a DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| GPU | 1 | 128 | 3.703 | 34.566 | 28.930 | 57.861 | 2517.000 | 0.474 | 4.391 | 0 | 0 |
| GPU | 2 | 128 | 3.917 | 32.675 | 30.605 | 61.209 | 2997.000 | 0.537 | 4.520 | 0 | 0 |
| GPU | 3 | 128 | 3.246 | 39.431 | 25.361 | 50.722 | 3028.000 | 0.545 | 4.520 | 0 | 0 |
