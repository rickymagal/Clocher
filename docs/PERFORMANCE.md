# Performance Notes

_Last updated: **2025-12-27 15:32:41 UTC**_


**Best true TPS:** **GPU — 941.971**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **13.778 s**
- True TPS (Σ tokens / Σ time): **27.871**

## Latency
- p50 (mean across runs): **35.880 ms**
- p95 (mean across runs): **71.759 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **1841.552 MB**
- RSS peak (max): **1878.328 MB**
- KV cache: **0 hits / 384 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **1.87 GB/s**

### Memory Details
- PSS peak (mean / max): **0.000 MB / 0.000 MB**
- VMS peak (mean / max): **0.000 MB / 0.000 MB**
- RSS floor (mean / max): **0.000 MB / 0.000 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **508690 / 78**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**

### Deduplication
- IE_DEDUP: **1**
- IE_DEDUP_STRICT: **0**
- IE_DEDUP_POLICY: **lossless**
- IE_DEDUP_CACHE_MB: **512**
- Artifacts (bytes / MB):
  - model.defaults.bin: **368340480** (368.34 MB)
  - model.masks.bin: **45977760** (45.98 MB)
  - model.exceptions.bin: **221578093** (221.58 MB)
  - Total dedup blobs: **635896333** (635.90 MB)
- Dedup blobs / model.ie.bin: **0.046209**
- model.ie.bin size: **13761264768** (13761.26 MB)
- Artifact paths (best effort):
  - defaults: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.defaults.bin` → `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/dedup_out/model.defaults.bin`
  - masks: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.masks.bin` → `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/dedup_out/model.masks.bin`
  - exceptions: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.exceptions.bin` → `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/dedup_out/model.exceptions.bin`

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **1152**
- Wall time (Σ): **1.223 s**
- True TPS (Σ tokens / Σ time): **941.971**

## Latency
- p50 (mean across runs): **1.062 ms**
- p95 (mean across runs): **2.123 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **94.710 MB**
- RSS peak (max): **94.793 MB**
- KV cache: **0 hits / 1152 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **77.3 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **63.21 GB/s**

### Memory Details
- PSS peak (mean / max): **0.000 MB / 0.000 MB**
- VMS peak (mean / max): **0.000 MB / 0.000 MB**
- RSS floor (mean / max): **0.000 MB / 0.000 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **17624 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**

### Deduplication
- IE_DEDUP: **1**
- IE_DEDUP_STRICT: **0**
- IE_DEDUP_POLICY: **lossless**
- IE_DEDUP_CACHE_MB: **512**
- Artifacts (bytes / MB):
  - model.defaults.bin: **368340480** (368.34 MB)
  - model.masks.bin: **45977760** (45.98 MB)
  - model.exceptions.bin: **221578093** (221.58 MB)
  - Total dedup blobs: **635896333** (635.90 MB)
- Dedup blobs / model.ie.bin: **0.046209**
- model.ie.bin size: **13761264768** (13761.26 MB)
- Artifact paths (best effort):
  - defaults: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.defaults.bin` → `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/dedup_out/model.defaults.bin`
  - masks: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.masks.bin` → `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/dedup_out/model.masks.bin`
  - exceptions: `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.exceptions.bin` → `/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/dedup_out/model.exceptions.bin`

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
- Git commit: **52f0d229 DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 128 | 4.510 | 28.380 | 35.236 | 70.473 | 1824.875 | 0.000 | 0.000 | 157642 | 26 |
| CPU | 2 | 128 | 4.017 | 31.868 | 31.380 | 62.759 | 1878.328 | 0.000 | 0.000 | 175868 | 23 |
| CPU | 3 | 128 | 5.251 | 24.377 | 41.023 | 82.046 | 1821.453 | 0.000 | 0.000 | 175180 | 29 |
| GPU | 1 | 384 | 0.413 | 930.049 | 1.075 | 2.150 | 94.793 | 0.000 | 0.000 | 5876 | 0 |
| GPU | 2 | 384 | 0.401 | 957.046 | 1.045 | 2.090 | 94.750 | 0.000 | 0.000 | 5874 | 0 |
| GPU | 3 | 384 | 0.409 | 939.218 | 1.065 | 2.129 | 94.586 | 0.000 | 0.000 | 5874 | 0 |
