# Architectural Decision Records

**Last updated:** 2025-10-24 21:00:48 UTC

- **ADR-0001 (2025-10-13)**: Core in C11, zero third-party runtime; Python stdlib harness.
- **ADR-0002 (2025-10-13)**: IEBIN v1 weights: `model.ie.json` + `model.ie.bin` + `vocab.json` (mmap-friendly).
- **ADR-0003 (2025-10-13)**: Baseline = FP32 naive path; TPS = generated_tokens / wall_time.
- **ADR-0006 (2025-10-14)**: Optimization path selection (CPU baseline) — AVX2 microkernels, thread-pool with affinity, fast math, layout packing. See `adr-00060-optimization-path.md`.
- **ADR-0010 (2025-10-16)**: INT8 PTQ min-max (per-tensor/per-row) with calibration script and accuracy budget gate (cosine ≥ threshold).
- **ADR-0011 (2025-10-24)**: Drop Level Zero from default benchmarking; GPU bench is CUDA-only; CPU/GPU share a unified reporting pipeline.

---

## **ADR-0013 (2025-10-24): Adopt INT4 weight-only PTQ & manifest-guided packing**

**Context.** The engine is primarily **memory-bandwidth bound** at inference time. Even with blocked‑K packing and prefetch, large models remain limited by weight fetch. INT8 provided a 2× compression; we want a **denser format** while preserving a simple compute path.

**Decision.**
- Introduce **INT4 weight‑only** quantization with:
  - **Nibble packing** (2 weights/byte) and **per‑row (or group) scales**.
  - A repository‑wide **manifest** (`q4_manifest.json`) describing which tensors are INT4 and how to dequantize them.
  - CLI/runtime selection via `IE_PRECISION=int4w` (or `--precision int4w`).
- Keep activations in floating point; accumulation stays FP32 for numerical stability.

**Consequences.**
- **4× smaller weights** → lower bandwidth → higher effective TPS under work‑touch or real inference.
- Slight accuracy cost, controlled by calibration. PTQ scripts expose gates (cosine/task‑level) to accept/reject a manifest.
- Kernel changes are incremental: dequant scale application in the matmul inner loop; packing integrates with existing pretranspose caches.

**Status.** Accepted. Implemented across CPU and CUDA builds; scripts support end‑to‑end HF→IEBIN packing with `--q4-map`.


---
## Decision: Add INT4 (Weight-Only) Optional Pipeline (2025-10-24 21:04:23 UTC)

**Context.** We already ship FP32/BF16/FP16 flows. We add a *weight-only* INT4 path to reduce bandwidth and model footprint while keeping API stability.

**Decision.**
- Introduce a manifest-driven INT4 packing step in export (`--q4-map`).
- Expose runtime selection via `IE_PRECISION=int4w` (or `--precision int4w`).
- Keep timing discipline: measure generation + work-touch only; sample metrics after.

**Consequences.**
- Lower I/O pressure when `IE_BYTES_PER_TOKEN` simulates large working sets.
- Slight decode overhead for dequantization (amortized in GEMV paths).
- No change to tokenization or batching semantics.

**Status.** Accepted and implemented. Backward-compatible (FP paths unaffected).

---

## Appendix — INT4 (Weight‑Only) Step (Summary)
- Convert HF shards → IEBIN with an INT4 manifest:
  ```bash
  python3 scripts/hf_to_iebin.py     --hf-dir models/gpt-oss-20b/hf     --out-dir models/gpt-oss-20b     --q4-map quant/q4_manifest.json
  ```
- Run benchmarks in strict mode with a **64 MB/token** work‑touch:
  ```bash
  PROMPTS=benchmarks/prompts_10..txt   IE_PRECISION=int4w IE_REQUIRE_MODEL=1   IE_BYTES_PER_TOKEN=64000000 IE_STRIDE_BYTES=256 RUNS=3   make bench           # or: make bench-cuda
  ```
- Precision hints: `PRECISION=fp32` (activates float path) and `IE_PRECISION=int4w` (weight‑only path).

---
**Last updated:** 2025-11-10 21:46:36 UTC

- **ADR-0014 (2025-11-10)**: NUMA‑aware topology and thread binding
  - **Context:** We need consistent socket awareness to reduce cross‑socket traffic and stabilize TPS.
  - **Decision:** Introduce `ie_topology` (sysfs‑backed) to detect sockets and map threads→CPUs with `AFFINITY` hints.
  - **Consequences:** Lower LLC misses and cross‑node memory, improved repeatability; graceful single‑socket fallback.
  - **Status:** Accepted. Implemented in `engine/include/ie_topology.h` and `engine/src/opt/topology.c`.

- **ADR-0015 (2025-11-10)**: Replicate “hot” weights per socket and bind workers
  - **Context:** For bandwidth‑bound models, remote‑node page faults throttle TPS. Replicating hot tensors near compute reduces latency.
  - **Decision:** Add `replicate_hot.c` to create per‑socket replicas (mmap‑based), then bind per‑socket workers to CPUs on that socket.
  - **Consequences:** Higher memory footprint when enabled, but improved locality and throughput on multi‑socket hosts.
  - **Status:** Accepted. Guarded by `IE_HOT_REPLICATE=1`; optional `MADV_WILLNEED` prefetch.

- **ADR-0016 (2025-11-10)**: Activation precision soft hint (INT8/FP8)
  - **Context:** Allow experimenting with lower‑precision activations without entangling storage precision for weights.
  - **Decision:** Add `IE_ACT_PREC` (values: `int8|fp8|fp16|bf16|fp32`). Host accumulators remain FP32; backends may ignore unsupported hints.
  - **Consequences:** Decouples storage (e.g., `int4w`) from activation math, enabling orthogonal tuning.
  - **Status:** Accepted. Backward‑compatible; default remains FP32 if unset.
---
## ADR‑0017 (Memory Streaming Heuristics): Prefetch & Non‑Temporal Loads — 2025-11-12 18:01:19 UTC

**Context.** GEMV paths are bandwidth‑bound. Sequential blocked‑K layouts help, but cache pollution and late prefetch still cap TPS.

**Decision.**
- Adopt tunables `IE_PREFETCH_DISTANCE`, `IE_NT_LOADS`, `IE_NT_RATIO`, and `IE_L3_BYTES` to steer prefetch distance, streaming loads,
  and an approximate L3 budget for “hot” slices.
- Default is **conservative** (`auto`) with architecture checks; explicit values override.

**Consequences.**
- Wins on large models/long K with single‑touch patterns.
- Small rows or re‑use‑heavy layers may regress if NT is forced → keep `auto` as default and expose per‑run sweeps in the harness.

**Status.** Accepted. Implemented in CPU AVX2 and CUDA pack/GEMV call sites with guarded paths.

---
## ADR‑0018 (Metrics & Reporting): Spatial Metrics in PERFORMANCE.md — 2025-11-12 18:01:19 UTC

**Context.** Throughput alone hides memory behavior. We need bytes/token, coverage vs model, RSS peak, and effective bandwidth.

**Decision.**
- Extend the JSON summary and the Markdown generator to compute spatial fields from `IE_BYTES_PER_TOKEN`, run totals, and model size.
- Update Prometheus exporter with a `ie_build_info` gauge (labels) and RSS gauges.

**Consequences.**
- Comparable runs across machines now capture memory pressure explicitly.
- Grafana dashboards can plot GB/s alongside TPS for stability analysis.

**Status.** Accepted. Shipped with `monitoring/metrics_memory.toml`, updated `scripts/metrics_exporter.py`,
and harness sweep labels for memory knobs.
