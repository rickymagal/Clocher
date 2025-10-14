# Clocher

C11, zero-dependency **CPU LLM inference baseline** with a minimal CLI and a stdlib-only Python harness for **reproducible TPS**.  
Designed for strict portability, deterministic baselines, and clean automation (Makefile + CI).

---

## Features
- **C11 core** (no external deps) with real FP32 compute (mat-vec, `tanhf`, deterministic embedding)
- **Per-token latency ring** → p50/p95 in the CLI JSON; harness computes true TPS
- **Stdlib-only harness**: generates CSV/JSON reports under `benchmarks/reports/<UTC_TS>/`
- **Microbench** for GEMV/tanh/embed hotspots
- **Profiling** via `perf → flamegraph` (with Callgrind fallback)
- **Doxygen-ready**: every C function documented; `docs/Doxyfile` included
- **Unit tests** (C + Python) and **Makefile** automation

---

## Quickstart
```bash
make build
make test
make bench
