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
   - `tanhf`: polynomial/table approximation gated by accuracy budget (with clamp).
   - Minimize expensive transcendentals in token feature paths.
4. **Layout** toggles (blocked-K packing) and prefetch hints in hot loops.
5. Keep an **optional oneDNN fallback** behind a build flag (non-default) for validation.

## Consequences
- Maximum control and transparency of performance changes.
- Minimal external dependencies → easier reproducibility.
- Larger code surface maintained in-tree; unit tests and microbenches mandatory.

## Next
- Complete AVX2 GEMV microkernel integration behind runtime gating.
- Validate scaling of thread-pool and NUMA effects with flamegraph evidence.
- Compare accuracy/latency impact of fast `tanh` vs libm baseline.
## Updates — 2025-10-17

- **Batching impact:** amortizes tokenizer and scheduling overhead; reduces per-prompt latency variance for short prompts.
- **Prefetch policy:** `auto` maps to enabled when `--prompts-file` is provided; otherwise conservative for single-prompt runs.
- **Profiling workflow:** `make perf-report` records with `perf` and renders with FlameGraph; supports `ROUNDS`, `MAX_NEW`, `FREQ`, and `CALLGRAPH` envs.
- **Limits observed:** if `perf.data` is empty, increase `ROUNDS/MAX_NEW/FREQ` or adjust kernel `perf_event_paranoid`.
