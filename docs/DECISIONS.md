# Architectural Decision Records

- **ADR-0001 (2025-10-13)**: Core in C11, zero third-party runtime; Python stdlib harness.
- **ADR-0002 (2025-10-13)**: IEBIN v1 weights: `model.ie.json` + `model.ie.bin` + `vocab.json` (mmap-friendly).
- **ADR-0003 (2025-10-13)**: Baseline = FP32 naive path; TPS = generated_tokens / wall_time.
- **ADR-0006 (2025-10-14)**: Optimization path selection (CPU baseline) — AVX2 microkernels, thread-pool with affinity, fast math, layout packing. See `adr-00060-optimization-path.md`.
- **ADR-0010 (2025-10-16)**: INT8 PTQ min-max (per-tensor/per-row) with calibration script and accuracy budget gate (cosine ≥ threshold).
