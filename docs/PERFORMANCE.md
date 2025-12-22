# Performance Notes

_Last updated: **2025-12-22 23:38:26 UTC**_


**Best true TPS:** **CPU — 42.747**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **128**
- Wall time (Σ): **2.994 s**
- True TPS (Σ tokens / Σ time): **42.747**

## Latency
- p50 (mean across runs): **23.393 ms**
- p95 (mean across runs): **46.787 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2830.000 MB**
- RSS peak (max): **2830.000 MB**
- KV cache: **0 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **8.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.87 GB/s**

### Memory Details
- PSS peak (mean / max): **2876.062 MB / 2876.062 MB**
- VMS peak (mean / max): **26857.961 MB / 26857.961 MB**
- RSS floor (mean / max): **102.832 MB / 102.832 MB**
- RSS delta vs baseline (mean / max): **2716.250 MB / 2716.250 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **0.79 % / 0.79 %**
- PSI memory 'full' (mean / max): **0.79 % / 0.79 %**
- System MemAvailable (mean): **2921.8 MB** — **37.8 % of MemTotal**

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **128**
- Wall time (Σ): **3.101 s**
- True TPS (Σ tokens / Σ time): **41.272**

## Latency
- p50 (mean across runs): **24.230 ms**
- p95 (mean across runs): **48.459 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2891.000 MB**
- RSS peak (max): **2891.000 MB**
- KV cache: **0 hits / 128 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **8.6 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.77 GB/s**

### Memory Details
- PSS peak (mean / max): **2951.759 MB / 2951.759 MB**
- VMS peak (mean / max): **26860.746 MB / 26860.746 MB**
- RSS floor (mean / max): **14.355 MB / 14.355 MB**
- RSS delta vs baseline (mean / max): **2868.906 MB / 2868.906 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **0.94 % / 0.94 %**
- PSI memory 'full' (mean / max): **0.94 % / 0.94 %**
- System MemAvailable (mean): **2950.6 MB** — **38.2 % of MemTotal**

## Run Parameters & Conditions
- Engine bin: `/home/ricardomag/Desktop/Clocher/build/inference-engine`
- Prompts file: `/home/ricardomag/Desktop/Clocher/benchmarks/prompts_10.txt`
- Threads: **12**
- Precision: **int4w**
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
- OS: **KDE neon User Edition**
- Kernel: **6.14.0-37-generic-x86_64**
- Git commit: **12f6632 DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 2.994 | 42.747 | 23.393 | 46.787 | 2830.000 | 2876.062 | 26857.961 | 0 | 0 |
| GPU | 1 | 128 | 3.101 | 41.272 | 24.230 | 48.459 | 2891.000 | 2951.759 | 26860.746 | 0 | 0 |
