# Performance Notes

_Last updated: **2025-12-20 17:52:16 UTC**_


**Best true TPS:** **CPU — 39.314**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **9.767 s**
- True TPS (Σ tokens / Σ time): **39.314**

## Latency
- p50 (mean across runs): **25.436 ms**
- p95 (mean across runs): **50.872 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **3414.333 MB**
- RSS peak (max): **3629.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.64 GB/s**

### Memory Details
- PSS peak (mean / max): **0.533 MB / 0.557 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.816 MB / 2.164 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **2.51 % / 4.43 %**
- PSI memory 'full' (mean / max): **2.52 % / 4.43 %**
- System MemAvailable (mean): **3755.7 MB** — **48.6 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **10.459 s**
- True TPS (Σ tokens / Σ time): **36.716**

## Latency
- p50 (mean across runs): **27.236 ms**
- p95 (mean across runs): **54.472 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **3764.333 MB**
- RSS peak (max): **3778.000 MB**
- KV cache: **3 hits / 381 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.46 GB/s**

### Memory Details
- PSS peak (mean / max): **0.520 MB / 0.543 MB**
- VMS peak (mean / max): **4.477 MB / 4.520 MB**
- RSS floor (mean / max): **1.546 MB / 1.938 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **3.80 % / 5.21 %**
- PSI memory 'full' (mean / max): **3.80 % / 5.21 %**
- System MemAvailable (mean): **3970.0 MB** — **51.4 % of MemTotal**

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
- Git commit: **0c5c214 DIRTY**
- Model file: **/home/ricky/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.189 | 40.143 | 24.911 | 49.822 | 3190.000 | 0.487 | 4.391 | 0 | 0 |
| CPU | 2 | 128 | 3.347 | 38.246 | 26.147 | 52.294 | 3424.000 | 0.556 | 4.520 | 0 | 0 |
| CPU | 3 | 128 | 3.232 | 39.604 | 25.250 | 50.500 | 3629.000 | 0.557 | 4.520 | 0 | 0 |
| GPU | 1 | 128 | 3.433 | 37.288 | 26.818 | 53.636 | 3753.000 | 0.477 | 4.391 | 0 | 0 |
| GPU | 2 | 128 | 3.557 | 35.983 | 27.791 | 55.581 | 3762.000 | 0.539 | 4.520 | 0 | 0 |
| GPU | 3 | 128 | 3.469 | 36.902 | 27.099 | 54.198 | 3778.000 | 0.543 | 4.520 | 0 | 0 |
