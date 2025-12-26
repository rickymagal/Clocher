# Performance Notes

_Last updated: **2025-12-26 01:48:27 UTC**_


**Best true TPS:** **GPU — 40.399**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **1152**
- Wall time (Σ): **30.665 s**
- True TPS (Σ tokens / Σ time): **37.567**

## Latency
- p50 (mean across runs): **26.619 ms**
- p95 (mean across runs): **53.238 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4652.000 MB**
- RSS peak (max): **4680.000 MB**
- KV cache: **768 hits / 384 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **77.3 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **2.52 GB/s**

### Memory Details
- PSS peak (mean / max): **4704.302 MB / 4738.703 MB**
- VMS peak (mean / max): **26857.961 MB / 26857.961 MB**
- RSS floor (mean / max): **9.811 MB / 12.398 MB**
- RSS delta vs baseline (mean / max): **4636.013 MB / 4661.355 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **6.65 % / 9.17 %**
- PSI memory 'full' (mean / max): **6.65 % / 9.17 %**
- System MemAvailable (mean): **4595.3 MB** — **59.4 % of MemTotal**

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
- Wall time (Σ): **28.516 s**
- True TPS (Σ tokens / Σ time): **40.399**

## Latency
- p50 (mean across runs): **24.753 ms**
- p95 (mean across runs): **49.506 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4664.333 MB**
- RSS peak (max): **4676.000 MB**
- KV cache: **768 hits / 384 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **77.3 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **2.71 GB/s**

### Memory Details
- PSS peak (mean / max): **4743.626 MB / 4779.243 MB**
- VMS peak (mean / max): **26860.746 MB / 26860.746 MB**
- RSS floor (mean / max): **9.195 MB / 11.637 MB**
- RSS delta vs baseline (mean / max): **4646.392 MB / 4660.062 MB**
- Page faults (minor Σ / major Σ): **0 / 0**
- Swap I/O (in Σ / out Σ): **0.0 MB / 0.0 MB**
- PSI memory 'some' (mean / max): **7.59 % / 10.50 %**
- PSI memory 'full' (mean / max): **7.59 % / 10.50 %**
- System MemAvailable (mean): **4611.2 MB** — **59.7 % of MemTotal**

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
- Git commit: **2305d6f DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 384 | 10.810 | 35.521 | 28.152 | 56.305 | 4680.000 | 4738.703 | 26857.961 | 0 | 0 |
| CPU | 2 | 384 | 10.447 | 36.757 | 27.205 | 54.411 | 4663.000 | 4700.147 | 26857.961 | 0 | 0 |
| CPU | 3 | 384 | 9.408 | 40.817 | 24.499 | 48.999 | 4613.000 | 4674.057 | 26857.961 | 0 | 0 |
| GPU | 1 | 384 | 9.237 | 41.573 | 24.054 | 48.108 | 4676.000 | 4779.243 | 26860.746 | 0 | 0 |
| GPU | 2 | 384 | 9.534 | 40.275 | 24.829 | 49.659 | 4674.000 | 4732.109 | 26860.746 | 0 | 0 |
| GPU | 3 | 384 | 9.744 | 39.407 | 25.376 | 50.752 | 4643.000 | 4719.525 | 26860.746 | 0 | 0 |
