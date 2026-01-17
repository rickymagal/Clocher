# Performance Notes

_Last updated: **2026-01-17 00:15:14 UTC**_


**Best true TPS:** **n/a — n/a**.

## Run Parameters & Conditions
- Engine bin: `/home/ricardomag/Desktop/Clocher/build/inference-engine`
- Prompts file: `/home/ricardomag/Desktop/Clocher/benchmarks/prompts_10.txt`
- Threads: **12**
- Precision: **int4**
- Batch: **1**
- Prefetch: **auto**
- Pretranspose: **all**
- Affinity: **auto**
- Max new tokens: **1**
- IE_REQUIRE_MODEL: **1**
- IE_BYTES_PER_TOKEN: **67108864**
- IE_STRIDE_BYTES: **256**
- IE_VERIFY_TOUCH: **1**
- IE_DEDUP: **1**
- IE_DEDUP_STRICT: **0**
- IE_DEDUP_POLICY: **lossless**
- IE_DEDUP_CACHE_MB: **0**

## System & Model Info
- CPU: **12th Gen Intel(R) Core(TM) i5-12450H**
- Logical cores: **12**
- RAM (MemTotal): **8.1 GB**
- OS: **KDE neon User Edition**
- Kernel: **6.14.0-37-generic-x86_64**
- Git commit: **2941beee DIRTY**
- Model file: **/home/ricardomag/Desktop/Clocher/models/gpt-oss-20b/model.ie.bin**
- Model size: **13.761 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
