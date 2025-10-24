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

