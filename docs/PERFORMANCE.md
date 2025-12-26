# Performance Notes

_Last updated: **2025-12-26 22:10:34 UTC**_


**Best true TPS:** **GPU — 624.162**.

## GPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **384**
- Wall time (Σ): **0.615 s**
- True TPS (Σ tokens / Σ time): **624.162**

## Latency
- p50 (mean across runs): **1.602 ms**
- p95 (mean across runs): **3.204 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **95.000 MB**
- RSS peak (max): **95.000 MB**
- KV cache: **0 hits / 384 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **41.89 GB/s**

### Memory Details
- PSS peak (mean / max): **92.907 MB / 92.919 MB**
- VMS peak (mean / max): **4760.309 MB / 4760.309 MB**
- RSS floor (mean / max): **8.531 MB / 8.656 MB**
- RSS delta vs baseline (mean / max): **86.219 MB / 86.449 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **0.00 % / 0.00 %**
- PSI memory 'full' (mean / max): **0.00 % / 0.00 %**
- System MemAvailable (mean): **1198.5 MB** — **15.5 % of MemTotal**

### Deduplication
- IE_DEDUP: **0**
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
- Engine bin: `/home/ricardomag/Desktop/Clocher/build/inference-engine.cuda`
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
- IE_DEDUP: **0**
- IE_DEDUP_STRICT: **0**
- IE_DEDUP_POLICY: **lossless**
- IE_DEDUP_CACHE_MB: **512**

## System & Model Info
- CPU: **12th Gen Intel(R) Core(TM) i5-12450H**
- Logical cores: **12**
- RAM (MemTotal): **8.1 GB**
- OS: **KDE neon User Edition**
- Kernel: **6.14.0-37-generic-x86_64**
- Git commit: **49e279e DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| GPU | 1 | 128 | 0.210 | 609.708 | 1.640 | 3.280 | 95.000 | 92.895 | 4760.309 | 0 | 0 |
| GPU | 2 | 128 | 0.202 | 634.828 | 1.575 | 3.150 | 95.000 | 92.919 | 4760.309 | 0 | 0 |
| GPU | 3 | 128 | 0.204 | 628.505 | 1.591 | 3.182 | 95.000 | 92.907 | 4760.309 | 0 | 0 |
