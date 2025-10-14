# ADR-0006 — Optimization path selection (CPU baseline)

## Status
Accepted — 2025-10-14

## Context
We need deterministic, portable, zero-dependency CPU performance. Upstream runtimes
(ORT/TVM/OpenVINO/TensorRT-CPU) bring heavy deps and reduce control over hot inner loops.

## Decision
1. **Hand-written AVX2 microkernels** for GEMV/GEMM with block tiling + FMA.
2. **Threading via pthreads pool** with pinning modes (`compact|scatter`) and optional NUMA hints.
3. **Math fast paths**:
   - `tanhf`: polynomial or table approximation gated by accuracy budget.
   - `embed_token`: reduce `sinf` usage; precompute small patterns or use cheaper blend.
4. **Layout** toggles (SoA-friendly) and prefetch hints in hot loops.
5. Keep an **optional oneDNN fallback** behind a build flag (non-default) for validation.

## Consequences
- Maximum control and transparency of performance changes.
- Minimal external dependencies → easier reproducibility.
- Larger code surface maintained in-tree; unit tests and microbenches mandatory.

## Next
- Implement AVX2 GEMV microkernel; integrate behind a runtime flag.
- Add thread pool + sharding; measure scaling and NUMA effects.
- Validate accuracy impact of fast `tanh` vs baseline libm.
