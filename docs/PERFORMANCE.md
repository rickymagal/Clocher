# Performance Notes

_Last updated: **2025-11-14 02:16:51 UTC**_


**Best true TPS:** **GPU — 32.848**.

## CPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **13.494 s**
- True TPS (Σ tokens / Σ time): **28.457**

## Latency
- p50 (mean across runs): **35.141 ms**
- p95 (mean across runs): **70.282 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4908.000 MB**
- RSS peak (max): **4908.000 MB**
- KV cache: **257 hits / 127 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **1.91 GB/s**

### Memory Details
- No extended memory metrics were present in the logs.

## GPU — Summary (latest benchmark)
- Runs: **1**
- Tokens generated (Σ): **384**
- Wall time (Σ): **11.690 s**
- True TPS (Σ tokens / Σ time): **32.848**

## Latency
- p50 (mean across runs): **30.443 ms**
- p95 (mean across runs): **60.886 ms**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **4976.000 MB**
- RSS peak (max): **4976.000 MB**
- KV cache: **257 hits / 127 misses**
- IE_BYTES_PER_TOKEN: **67.1 MB/token**
- Bytes touched (Σ): **25.8 GB**
- Working-set coverage (bytes_per_token / model.bin): **0.005**
- Effective bandwidth: **2.20 GB/s**

### Memory Details
- No extended memory metrics were present in the logs.

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
- OS: **Fedora Linux 42 (KDE Plasma Desktop Edition)**
- Kernel: **6.16.9-200.fc42.x86_64-x86_64**
- Git commit: **d699e99 DIRTY**
- Model file: **models/gpt-oss-20b/model.ie.bin**
- Model size: **13.197 GB**

## Comparative Runs

| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |
|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|
| CPU | 1 | 384 | 13.494 | 28.457 | 35.141 | 70.282 | 4908.000 | n/a | n/a | 0 | 0 |
| GPU | 1 | 384 | 11.690 | 32.848 | 30.443 | 60.886 | 4976.000 | n/a | n/a | 0 | 0 |
