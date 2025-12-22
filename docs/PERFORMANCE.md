# Performance Notes

_Last updated: **2025-12-22 23:50:51 UTC**_


**Best true TPS:** **GPU — 40.578**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **9.713 s**
- True TPS (Σ tokens / Σ time): **39.535**

## Latency
- p50 (mean across runs): **25.294 ms**
- p95 (mean across runs): **50.588 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2866.333 MB**
- RSS peak (max): **2875.000 MB**
- KV cache: **0 hits / 384 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **2.65 GB/s**

### Memory Details
- PSS peak (mean / max): **2918.350 MB / 2922.819 MB**
- VMS peak (mean / max): **26857.961 MB / 26857.961 MB**
- RSS floor (mean / max): **15.052 MB / 24.488 MB**
- RSS delta vs baseline (mean / max): **2829.772 MB / 2841.023 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **4.90 % / 6.46 %**
- PSI memory 'full' (mean / max): **4.90 % / 6.46 %**
- System MemAvailable (mean): **2902.3 MB** — **37.5 % of MemTotal**

### Deduplication
- IE_DEDUP: **1**
- IE_DEDUP_STRICT: **0**
- IE_DEDUP_POLICY: **lossless**
- IE_DEDUP_CACHE_MB: **512**
- Artifacts (bytes / MB):
  - model.defaults.bin: **28** (0.00 MB)
  - model.masks.bin: **25** (0.00 MB)
  - model.exceptions.bin: **30** (0.00 MB)
  - Total dedup blobs: **83** (0.00 MB)
- Dedup blobs / model.ie.bin: **0.0000**
- model.ie.bin size: **13_761_264_768** (13761.26 MB)
- Artifact paths (best effort):
  - defaults: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.defaults.bin`
  - masks: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.masks.bin`
  - exceptions: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.exceptions.bin`

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **9.463 s**
- True TPS (Σ tokens / Σ time): **40.578**

## Latency
- p50 (mean across runs): **24.644 ms**
- p95 (mean across runs): **49.288 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2875.000 MB**
- RSS peak (max): **2897.000 MB**
- KV cache: **0 hits / 384 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **2.72 GB/s**

### Memory Details
- PSS peak (mean / max): **2924.443 MB / 2957.741 MB**
- VMS peak (mean / max): **26860.746 MB / 26860.746 MB**
- RSS floor (mean / max): **7.776 MB / 11.645 MB**
- RSS delta vs baseline (mean / max): **2850.382 MB / 2879.684 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **3.30 % / 5.60 %**
- PSI memory 'full' (mean / max): **3.31 % / 5.60 %**
- System MemAvailable (mean): **2898.8 MB** — **37.5 % of MemTotal**

### Deduplication
- IE_DEDUP: **1**
- IE_DEDUP_STRICT: **0**
- IE_DEDUP_POLICY: **lossless**
- IE_DEDUP_CACHE_MB: **512**
- Artifacts (bytes / MB):
  - model.defaults.bin: **28** (0.00 MB)
  - model.masks.bin: **25** (0.00 MB)
  - model.exceptions.bin: **30** (0.00 MB)
  - Total dedup blobs: **83** (0.00 MB)
- Dedup blobs / model.ie.bin: **0.0000**
- model.ie.bin size: **13_761_264_768** (13761.26 MB)
- Artifact paths (best effort):
  - defaults: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.defaults.bin`
  - masks: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.masks.bin`
  - exceptions: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.exceptions.bin`

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
- IE_DEDUP: **1**
- IE_DEDUP_STRICT: **0**
- IE_DEDUP_POLICY: **lossless**
- IE_DEDUP_CACHE_MB: **512**

## System & Model Info
- CPU: **12th Gen Intel(R) Core(TM) i5-12450H**
- Logical cores: **12**
- RAM (MemTotal): **8.1 GB**
- OS: **KDE neon User Edition**
- Kernel: **6.14.0-37-generic-x86_64**
- Git commit: **f946d57 DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 3.316 | 38.603 | 25.905 | 51.810 | 2858.000 | 2922.819 | 26857.961 | 0 | 0 |
| CPU | 2 | 128 | 3.180 | 40.249 | 24.845 | 49.691 | 2866.000 | 2912.414 | 26857.961 | 0 | 0 |
| CPU | 3 | 128 | 3.217 | 39.790 | 25.132 | 50.264 | 2875.000 | 2919.816 | 26857.961 | 0 | 0 |
| GPU | 1 | 128 | 3.100 | 41.289 | 24.220 | 48.439 | 2832.000 | 2878.316 | 26860.746 | 0 | 0 |
| GPU | 2 | 128 | 3.127 | 40.931 | 24.432 | 48.863 | 2897.000 | 2937.271 | 26860.746 | 0 | 0 |
| GPU | 3 | 128 | 3.236 | 39.555 | 25.281 | 50.563 | 2896.000 | 2957.741 | 26860.746 | 0 | 0 |
