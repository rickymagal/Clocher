# Performance Notes

_Last updated: **2025-12-15 12:31:55 UTC**_


**Best true TPS:** **CPU — 41.245**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **9.310 s**
- True TPS (Σ tokens / Σ time): **41.245**

## Latency
- p50 (mean across runs): **24.246 ms**
- p95 (mean across runs): **48.491 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **3534.333 MB**
- RSS peak (max): **3592.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **2.77 GB/s**

### Memory Details
- PSS peak (mean / max): **3585.538 MB / 3656.344 MB**
- VMS peak (mean / max): **43481.641 MB / 43481.641 MB**
- RSS floor (mean / max): **23.031 MB / 32.910 MB**
- RSS delta vs baseline (mean / max): **3495.884 MB / 3581.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **3.08 % / 5.02 %**
- PSI memory 'full' (mean / max): **2.97 % / 4.91 %**
- System MemAvailable (mean): **4436.0 MB** — **57.4 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **15.303 s**
- True TPS (Σ tokens / Σ time): **25.093**

## Latency
- p50 (mean across runs): **39.852 ms**
- p95 (mean across runs): **79.704 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **5291.667 MB**
- RSS peak (max): **5336.000 MB**
- KV cache: **0 hits / 384 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.001**
- Effective bandwidth: **1.68 GB/s**

### Memory Details
- PSS peak (mean / max): **5439.606 MB / 5482.250 MB**
- VMS peak (mean / max): **43481.641 MB / 43481.641 MB**
- RSS floor (mean / max): **25.854 MB / 27.172 MB**
- RSS delta vs baseline (mean / max): **5258.043 MB / 5300.480 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **2.06 % / 3.15 %**
- PSI memory 'full' (mean / max): **2.06 % / 3.15 %**
- System MemAvailable (mean): **5387.3 MB** — **69.7 % of MemTotal**

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
- Git commit: **e1426ba DIRTY**
- Model file: **models/gpt-oss-20b/model.ie.bin**
- Model size: **45.590 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.078 | 41.591 | 24.044 | 48.087 | 3436.000 | 3460.727 | 43481.641 | 0 | 0 |
| CPU | 2 | 128 | 3.118 | 41.049 | 24.361 | 48.722 | 3575.000 | 3639.543 | 43481.641 | 0 | 0 |
| CPU | 3 | 128 | 3.114 | 41.098 | 24.332 | 48.664 | 3592.000 | 3656.344 | 43481.641 | 0 | 0 |
| GPU | 1 | 128 | 4.853 | 26.376 | 37.914 | 75.828 | 5228.000 | 5369.276 | 43481.641 | 0 | 0 |
| GPU | 2 | 128 | 5.318 | 24.068 | 41.548 | 83.097 | 5311.000 | 5467.293 | 43481.641 | 0 | 0 |
| GPU | 3 | 128 | 5.132 | 24.942 | 40.094 | 80.187 | 5336.000 | 5482.250 | 43481.641 | 0 | 0 |
