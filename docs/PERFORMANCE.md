# Performance Notes

_Last updated: **2025-12-27 15:21:13 UTC**_


**Best true TPS:** **GPU — 947.207**.

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **1152**
- Wall time (Σ): **1.216 s**
- True TPS (Σ tokens / Σ time): **947.207**

## Latency
- p50 (mean across runs): **1.056 ms**
- p95 (mean across runs): **2.111 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **94.465 MB**
- RSS peak (max): **94.746 MB**
- KV cache: **0 hits / 1152 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **77.3 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **63.57 GB/s**

### Memory Details
- PSS peak (mean / max): **0.000 MB / 0.000 MB**
- VMS peak (mean / max): **0.000 MB / 0.000 MB**
- RSS floor (mean / max): **0.000 MB / 0.000 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **17628 / 2**
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
- Git commit: **62f85757 DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| GPU | 1 | 384 | 0.408 | 942.106 | 1.061 | 2.123 | 94.262 | 0.000 | 0.000 | 5882 | 2 |
| GPU | 2 | 384 | 0.410 | 935.700 | 1.069 | 2.137 | 94.387 | 0.000 | 0.000 | 5872 | 0 |
| GPU | 3 | 384 | 0.398 | 964.285 | 1.037 | 2.074 | 94.746 | 0.000 | 0.000 | 5874 | 0 |
