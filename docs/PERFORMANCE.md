# Performance Notes

_Last updated: **2025-12-27 16:14:16 UTC**_


**Best true TPS:** **GPU — 998.266**.

## CPU — Summary (latest benchmark)
- Runs: **3**
- Tokens generated (Σ): **1152**
- Wall time (Σ): **28.915 s**
- True TPS (Σ tokens / Σ time): **39.841**

## Latency
- p50 (mean across runs): **25.100 ms**
- p95 (mean across runs): **50.200 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **2567.223 MB**
- RSS peak (max): **2586.699 MB**
- KV cache: **0 hits / 1152 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **77.3 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **2.67 GB/s**

### Memory Details
- PSS peak (mean / max): **0.000 MB / 0.000 MB**
- VMS peak (mean / max): **0.000 MB / 0.000 MB**
- RSS floor (mean / max): **0.000 MB / 0.000 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **1180255 / 31**
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
- Wall time (Σ): **1.154 s**
- True TPS (Σ tokens / Σ time): **998.266**

## Latency
- p50 (mean across runs): **1.002 ms**
- p95 (mean across runs): **2.003 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **94.828 MB**
- RSS peak (max): **94.910 MB**
- KV cache: **0 hits / 1152 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **77.3 GB**
- Working-set coverage (bytes_per_token / model.ie.bin): **0.004877**
- Effective bandwidth: **66.99 GB/s**

### Memory Details
- PSS peak (mean / max): **0.000 MB / 0.000 MB**
- VMS peak (mean / max): **0.000 MB / 0.000 MB**
- RSS floor (mean / max): **0.000 MB / 0.000 MB**
- RSS delta vs baseline (mean / max): **0.000 MB / 0.000 MB**
- Page faults (minor Σ / major Σ): **17623 / 0**
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
- Git commit: **4781cdaa DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Report Verification
- The bench report uses `benchmarks/prompts_10.txt` and `benchmarks/expected_tokens.txt` by default.
- `make bench REPORT=1` auto-generates `benchmarks/expected_tokens.txt` via `tools/generate_expected_tokens.py` when missing.
- Each prompt is hashed to a `prompt_id` using FNV-1a over the prompt bytes.
- During the run, the engine compares generated token IDs against the expected list for that `prompt_id`.
- The per-prompt report records `expected_present` and `expected_ok` plus the generated token IDs.
- `make bench REPORT=1 VERIFY=1` runs the report and `tools/verify_report_tokens.py --require-expected` to fail if any prompt is missing expected tokens or mismatches.

Expected tokens file format:
- One prompt per line: `<prompt_id><space><token0,token1,token2,...>`
- `prompt_id` can be decimal or 0x-prefixed hex (FNV-1a 64-bit).

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 384 | 9.490 | 40.464 | 24.713 | 49.427 | 2539.055 | 0.000 | 0.000 | 393408 | 10 |
| CPU | 2 | 384 | 9.718 | 39.515 | 25.307 | 50.614 | 2586.699 | 0.000 | 0.000 | 393425 | 14 |
| CPU | 3 | 384 | 9.707 | 39.558 | 25.279 | 50.559 | 2575.914 | 0.000 | 0.000 | 393422 | 7 |
| GPU | 1 | 384 | 0.392 | 978.894 | 1.022 | 2.043 | 94.910 | 0.000 | 0.000 | 5874 | 0 |
| GPU | 2 | 384 | 0.383 | 1003.586 | 0.996 | 1.993 | 94.824 | 0.000 | 0.000 | 5872 | 0 |
| GPU | 3 | 384 | 0.379 | 1012.942 | 0.987 | 1.974 | 94.750 | 0.000 | 0.000 | 5877 | 0 |
