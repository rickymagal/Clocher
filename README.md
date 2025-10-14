# Clocher — CPU LLM Inference Baseline (C11 core + Python stdlib harness)

C11, zero-dependency **CPU inference baseline** with a minimal CLI and a stdlib-only Python harness for **reproducible throughput (TPS)** and **per-token latency**. The project emphasizes portability, deterministic runs, rigorous tests, and turnkey automation (Makefile + CI + Doxygen).

---

## Features

- **C11 core**, no external runtime deps (`-lpthread -lm` only)
- **Deterministic baseline** dummy generator (repeatable tokens for equal prompts)
- **Per-token latency ring** with p50/p95 in CLI JSON
- **Vectorization**: AVX2/FMA GEMV path (runtime-gated), light prefetch
- **Fast activations**: clamped `tanh` approx + vector helpers
- **Precision plumbing**: optional BF16/FP16 round-trip (FP32 accumulation)
- **Threading**: fixed thread pool, contiguous partitioning, grainsize control
- **Affinity (Linux)**: opt-in CPU pinning via `IE_TP_USE_AFFINITY=1`
- **Layout**: blocked-K packing + optional on-disk cache
- **Microbenchmarks** and **perf → flamegraph** pipeline
- **Unit tests** (C + Python), **Doxygen** comments everywhere, **Makefile** automation

---

## Repository layout

```
engine/
  include/        # Public headers (Doxygen)
  src/            # C sources: core, kernels, math, io, opt
benchmarks/
  harness.py      # Python stdlib harness (CSV/JSON reports)
  reports/        # Generated results (timestamped)
  src/            # Microbench sources
docs/
  Doxyfile        # Doxygen config (outputs to docs/doxygen/)
scripts/
  run_benchmark.sh
  profile_flamegraph.sh
  set_numa.sh     # NUMA policy wrapper (no libnuma link required)
  make_baseline_md.py
  update_performance_md.py
tests/
  c/              # C unit tests
  python/         # Python tests (unittest)
Makefile
```

---

## Requirements

- **Build toolchain**: GCC/Clang with C11
- **Linux/macOS** (Windows via MSYS/WSL)
- **perf** (Linux) for flamegraphs (optional)
- **Python 3.8+** (stdlib only) for the harness
- **Doxygen** for HTML docs (optional)

---

## Quickstart

```bash
make build     # build the CLI binary
make test      # run C + Python tests
make bench     # run harness → CSV/JSON reports under benchmarks/reports/<UTC_TS>/
```

---

## CLI usage

```bash
./build/inference-engine [--prompt TEXT] [--max-new N]
                         [--threads N]
                         [--precision fp32|bf16|fp16]
                         [--affinity auto|compact|scatter]
                         [--pretranspose none|woh|wxh|all]
                         [--grainsize K]
                         [--help]
```

### Example

```bash
./build/inference-engine --prompt "hello" --max-new 8 --threads 4 --precision fp32
```

**Output (JSON):**
```json
{
  "tokens_generated": 8,
  "tokens": [1705,1657,1537,1183,1704,1315,1917,1162],
  "wall_time_s": 0.001209,
  "tps_true": 7639.897996,
  "latency_p50_ms": 0.131,
  "latency_p95_ms": 0.143,
  "rss_peak_mb": 0,
  "kv_hits": 0,
  "kv_misses": 8
}
```

---

## Notable flags

- `--threads N` — number of worker threads.
- `--affinity {auto,compact,scatter}` — policy name; **effective on Linux only** and only if `IE_TP_USE_AFFINITY=1`.
- `--grainsize K` — chunk size for contiguous partitioning (used by parallel-for).
- `--precision {fp32,bf16,fp16}` — optional round-trip to BF16/FP16 with **FP32 accumulation**.
- `--pretranspose {none,woh,wxh,all}` — pretranspose/pack weights and optionally cache to disk; `all` applies to both projections if present.

**Linux CPU pinning toggle**
```bash
IE_TP_USE_AFFINITY=1 ./build/inference-engine --prompt "x" --max-new 8 --threads 4 --affinity scatter
```

---

## Benchmarks & baseline

Run the harness:

```bash
make bench
# Reports: benchmarks/reports/<UTC_TS>/{samples.csv,summary.json}
```

Create a baseline markdown from the latest report:

```bash
make baseline-report
# Generates: benchmarks/reports/<UTC_TS>/BASELINE.md
```

---

## Profiling & flamegraphs (Linux)

```bash
make profile
# → perf.data, flamegraph.svg, and PERFORMANCE.md updated
```

If `perf` is missing: `sudo apt-get install linux-tools-common linux-tools-generic` (or distro equivalent).

Update the performance notes (English) again at any time:

```bash
python3 scripts/update_performance_md.py
```

---

## NUMA policies (Linux, no libnuma link)

Use the helper to set process-wide policy and then launch the CLI:

```bash
scripts/set_numa.sh interleave -- ./build/inference-engine --prompt "x" --max-new 8
scripts/set_numa.sh node:0     -- ./build/inference-engine --prompt "x" --max-new 8
scripts/set_numa.sh strict     -- ./build/inference-engine --prompt "x" --max-new 8
```

> The project also detects available nodes via sysfs to annotate reports; true in-process policy calls are intentionally out of scope to keep the binary dependency-free.

---

## Testing

```bash
make test
```

- **C tests**: tensors, kernels (generic + AVX2), thread-pool, math, tokenizer, weights.
- **Python tests**: CLI flags and error paths, harness report generation, determinism, metrics ring (p50/p95).

---

## Documentation (Doxygen)

```bash
make docs-doxygen
# HTML at: docs/doxygen/html/index.html
```

All C functions (public and internal TUs) are documented with Doxygen-style comments.

---

## Coding standards

- **C**: C11, warnings as errors (`-Wall -Wextra -Werror -pedantic`)
- **Formatting**: `make fmt` (uses `clang-format` if available)
- **Linting**: `make lint` (uses `clang-tidy` if available)
- **Commits**: semantic commit messages
- **Makefile**: single entry point for build, test, bench, profile, docs

---

## Reproducibility

- Deterministic token generator for equal prompts.
- Harness records parameters (threads, affinity toggle, precision, pretranspose) with timestamps.
- Optional on-disk caches for packed weights (files are content-addressed by shape/blk_k).

---

## Roadmap (next steps)

- **Step 5**: INT8 PTQ with calibration set and accuracy budget checks; validation harness.
- **Step 6**: I/O & batching — async prefetch, pinned workers, microbatch sizing, warmup policy.
- Optional: LFU cache for frequent embeddings; advisory `--numa` CLI; Docker + CI matrix polish.

---

## License

TBD by client/employer. (A permissive license such as MIT/BSD is recommended for baselines.)
