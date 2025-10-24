# ADR-00060 — Optimization Path Selection (updated for INT4)

**Status:** Accepted • **Last updated:** 2025-10-24 21:00:48 UTC

## Problem
We need a portable CPU baseline with predictable performance and a clear path to exploit memory locality and bandwidth while keeping the code simple and testable.

## Decision
- Keep a single binary with **FP32 baseline** and optional precision switches: BF16/FP16 (accumulate FP32), INT8 PTQ, and **INT4 weight‑only (INT4W)**.
- Maintain **blocked‑K packing** and optional **pretranspose** for cache locality.
- Threading via a fixed `pthread` pool with optional affinity and NUMA hints.

## INT4W specifics (new)
- **When to use**: memory‑bound regimes (large models, small batches) where weight bandwidth dominates. Select with `IE_PRECISION=int4w` or `--precision int4w`.
- **How it works**:
  - PTQ scripts derive a `q4_manifest.json` (per‑tensor entry with packing metadata and scales).
  - `scripts/hf_to_iebin.py --q4-map …` consumes the manifest and packs `model.ie.bin` accordingly.
  - Kernels fuse **dequant(scale × int4)** into the matmul path; accumulation remains FP32.
- **Interplay with pretranspose**: quantization occurs after any offline layout transform intended for locality; manifests capture the final layout to keep IO deterministic.
- **Quality gate**: manifests are accepted only if a calibration threshold (cosine or task metric) is met; the threshold is repo‑configurable in the PTQ helpers.

## Rationale
INT4W yields **4× compression** over FP32 and **2× over INT8**, unlocking higher tokens/s on bandwidth‑limited hardware with minimal additional complexity.

## Consequences
- Slight accuracy loss bounded by the calibration gate.
- Simplified runtime: no activation quantization; scales live with the packed weights and are broadcast during compute.
- Uniform harness & docs: packing is explicit and reproducible via scripts and environment flags.

