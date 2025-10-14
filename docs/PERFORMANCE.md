# PERFORMANCE

This document records policies and reproduction guidance for CPU performance.
All content is generated in English by this script.

## Current Profile (baseline FP32)
- Experiment timestamp: **2025-10-14 17:14 UTC**
- Prompt/runner: `benchmarks/prompts.jsonl` via `benchmarks/harness.py`

### Metrics (latest report)
- **Avg True TPS (harness)**: **6534.23** tok/s
- **p50**: **0.173** ms | **p95**: **0.231** ms
- **Total tokens**: **160** | **samples**: **10**

### Hot Paths (flamegraph)
- `ie_gemv_f32_generic` — **85.6%**
- `ie_engine_create` — **10.4%**
- `[unknown]` — **4.0%**

> Flamegraph: `../flamegraph.svg`

## Next Optimizations
- **GEMV**: AVX2/AVX-512 micro-kernels, blocking and FMA epilogues.
- **Threading/NUMA**: contiguous row sharding + CPU pinning (compact/scatter).
- **tanh**: polynomial/LUT approximation for faster nonlinearities.
- **Embedding**: avoid `sinf` or precompute token-dependent patterns.

## Reproduction & Policies
- **Threads**: `--threads N` (default 1 for stable CI).
- **CPU Affinity (Linux)**: enable per run with `IE_TP_USE_AFFINITY=1` and choose `--affinity {auto,compact,scatter}`.
- **NUMA (Linux)**: use `scripts/set_numa.sh {compact|interleave|node:X} -- <CMD>` (external helper; engine remains runtime-only).

## Evolution Table
| Build/tag | Precision | Threads | Avg True TPS | p50 (ms) | p95 (ms) | Notes |
|-----------|:---------:|:-------:|-------------:|---------:|---------:|:------|
| baseline  | fp32      | auto    | 6534.23 | 0.173 | 0.231 | Initial baseline |
