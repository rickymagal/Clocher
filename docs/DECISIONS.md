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

## ADR‑0019 (Sparsity): Block‑sparse weights, CPU‑only prototype — 2025‑11‑14 23:00:00 UTC

**Context.**
- Phase 1 of the memory work focused on *dense* optimizations: INT4/INT8 activations, NUMA‑aware topology, hot‑weights replication, and streaming loads.
- The next natural axis is **algorithmic sparsity**: skip zero blocks instead of streaming everything.
- We want a conservative, inspectable prototype that:
  - lives entirely on the CPU path;
  - does **not** disturb the existing dense loading/CLI;
  - is testable in isolation (C unit tests + microbench).

**Decision.**
- Introduce a small, self‑contained **block‑sparse format** and CPU kernel:
  - New descriptor type `ie_block_sparse_matrix_t` in `engine/include/sparse_format.h`.
  - Loader in `engine/src/sparse_io.c` that reads a compact binary header and BSR payload.
  - Reference GEMV in `engine/src/gemm_block_sparse.c` operating on FP32 weights.
- Extend the device abstraction:
  - Add `gemv_block_sparse_f32` to the `ie_device` vtable.
  - Provide a CPU implementation wired to `ie_gemv_block_sparse_f32`.
  - Keep CUDA/Level‑Zero entries as stubs that return “unimplemented”; the public helpers fall back to CPU.
- Add offline and benchmarking tools:
  - `tools/convert_to_block_sparse.c` to convert dense row‑major weights into the on‑disk BSR format.
  - `tests/c/test_block_sparse.c` with small hand‑built matrices to validate loader + GEMV.
  - `benchmarks/src/microbench_gemv_block_sparse.c` to compare dense vs block‑sparse GEMV on CPU.
- Restrict scope explicitly:
  - **CPU only** for this ADR; no GPU kernels are required to pass tests.
  - **FP32 weights only**; quantized/specialized paths remain dense.

**Consequences.**
- We have a **concrete, repeatable** sparsity experiment:
  - Same repository, same Makefile; add a dense `.bin`, run the converter, microbench the result.
  - Unit tests ensure correctness on small matrices.
- The device layer remains forward‑compatible:
  - Future GPU support can fill in the existing vtable slot without touching the public API.
  - Callers can ask for block‑sparse GEMV and still get a correct CPU fallback today.
- No risk to existing users:
  - CLI behavior and `model.ie.bin` format are unchanged.
  - All new code is opt‑in and exercised only by tests/microbench/scripts introduced in this phase.

**Status.**
- Accepted. Implemented as a **CPU‑only prototype**; further ADRs will cover wiring full‑model weights and any GPU/quantized variants.


---

## ADR-0020 (2025-12-22): Adopt lossless dedup artifacts and schema2 runtime loader

**Context.** The engine is primarily memory-bandwidth bound at inference time. We already attack the problem
via INT4 weight-only storage and streaming heuristics, but dense IEBIN still requires reading large swaths of
the weight blob repeatedly. We need a **lossless** dedup scheme that can reduce DRAM traffic without changing
model outputs.

**Decision.**
- Adopt a three-blob dedup artifact set alongside IEBIN:
  - `model.defaults.bin`
  - `model.masks.bin`
  - `model.exceptions.bin`
- Enable the loader behind runtime flags:
  - `IE_DEDUP=1`, `IE_DEDUP_POLICY=lossless`
  - `IE_DEDUP_STRICT=1` for “fail fast” correctness runs
  - `IE_DEDUP_CACHE_MB` to cap reconstruction cache memory
  - `IE_DEDUP_DEBUG=1` for verbose diagnostics
- Keep schema2 metadata parsing strict, but allow compatibility with HF-derived metadata by:
  - requiring lowercase `dtype`
  - supporting `file`/`file_data_offset` (and generating aliases from `shard`/`shard_data_offset` in the metadata pipeline when needed)

**Consequences.**
- Offline extraction time increases (diffing defaults vs targets), but runtime is simplified to patch-apply.
- Production runs must ship three additional binary files next to `model.ie.json` / `model.ie.bin` (or symlink them).
- Benchmarks can now report meaningful “dedup on/off” TPS deltas under strict work-touch conditions.

**Status.** Accepted. Implemented (CPU path) and integrated in strict harness runs. CUDA path uses the same
artifact layout and flags where supported.
