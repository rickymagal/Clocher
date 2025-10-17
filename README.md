# Clocher — CPU LLM Inference Baseline (C11 core + Python stdlib harness)

C11, zero-dependency **CPU inference baseline** with a minimal CLI and a stdlib-only Python harness for **reproducible throughput (TPS)** and **per-token latency**. The project emphasizes portability, determinism, rigorous tests, and turnkey automation (Makefile + CI + Doxygen).

---

## Features

- **C11 core**, no external runtime deps (`-lpthread -lm` only)
- **Deterministic generator** (repeatable tokens for equal prompts)
- **Per-token latency ring** with p50/p95 in CLI JSON
- **Vectorization**: AVX2/FMA GEMV path (runtime-gated), light prefetch
- **Fast activations**: clamped `tanh` approx + vector helpers
- **Precision plumbing**: optional BF16/FP16 round-trip (FP32 accumulation)
- **INT8 PTQ pipeline**: min-max scales (per-tensor/per-row), calibration script with accuracy checks
- **Threading**: fixed thread pool, contiguous sharding, grainsize control
- **Affinity (Linux)**: opt-in CPU pinning via `IE_TP_USE_AFFINITY=1`
- **Layout**: blocked-K packing + optional on-disk cache
- **Microbenchmarks** and **perf → flamegraph** pipeline
- **Unit tests** (C + Python), **Doxygen** comments everywhere, **Makefile** automation

---

## Repository layout

```
engine/
  include/        # Public headers (Doxygen)
  src/            # C sources: core, kernels, math, io, opt, quant
benchmarks/
  harness.py      # Python stdlib harness (CSV/JSON reports)
  ptq_calib.py    # INT8 PTQ calibration/validation
  reports/        # Generated results (timestamped)
  src/            # Microbench sources
docs/
  Doxyfile        # Doxygen config (outputs to docs/doxygen/)
scripts/
  run_benchmark.sh
  profile_flamegraph.sh
  set_numa.sh     # NUMA policy helper (no libnuma link required)
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
- **Python 3.8+** (stdlib only) for harness & scripts
- **Doxygen** for HTML docs (optional)

---

## Quickstart

```bash
make build     # build the CLI binary
make test      # run C + Python tests
make bench     # run harness → CSV/JSON under benchmarks/reports/<UTC_TS>/
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
- `--affinity {auto,compact,scatter}` — policy name; **effective on Linux** only and only if `IE_TP_USE_AFFINITY=1`.
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
# → perf.data, flamegraph.svg, and PERFORMANCE.md updated (English)
```

Re-emit/update performance notes:

```bash
python3 scripts/update_performance_md.py
```

---

## NUMA policies (Linux, no libnuma link)

Set a process-wide policy, then launch the engine:

```bash
scripts/set_numa.sh interleave -- ./build/inference-engine --prompt "x" --max-new 8
scripts/set_numa.sh node:0     -- ./build/inference-engine --prompt "x" --max-new 8
scripts/set_numa.sh strict     -- ./build/inference-engine --prompt "x" --max-new 8
```

---

## INT8 PTQ pipeline

Per-tensor or per-row min-max scaling with accuracy checks.

**From an existing FP32 weight matrix (.bin, row-major):**
```bash
python3 benchmarks/ptq_calib.py   --weights bin/W.bin --rows 768 --cols 768   --mode per_row --out-prefix out/W_int8 --accuracy-threshold 0.995
```

**From Hugging Face (state_dict key):**
```bash
make ptq-from-hf HF_MODEL=facebook/opt-125m   KEY=model.decoder.layers.0.self_attn.q_proj.weight   OUT_PREFIX=out/qproj_int8
```

Artifacts: `<prefix>.int8.bin`, `<prefix>.scales.bin`, `<prefix>.report.json`.

---

## Testing

```bash
make test
```

- **C tests**: tensors, kernels (generic + AVX2), thread-pool, math, tokenizer, weights, **int8_ptq**.
- **Python tests**: CLI, harness, metrics ring (p50/p95), determinism, **PTQ calibration pipeline**.

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

## Roadmap

- **Step 6**: I/O & batching — async prefetch, pinned workers, microbatch sizing, warmup policy.
- Extras: GPU/iGPU variant (CUDA/oneAPI), INT4 exploration, Prometheus/Grafana dashboard.

---
## Updates — 2025-10-17

- **CLI flags added:** `--prompts-file PATH`, `--batch N`, `--prefetch on|off|auto`, `--warmup N`.
- **Examples:**
  - `inference-engine --prompts-file benchmarks/prompts_10.txt --batch 32 --max-new 8 --prefetch on --warmup 8`
  - `inference-engine --prompt "hello" --max-new 16 --threads 4 --precision fp32`
- **Stable JSON output:** ensure exact field formatting used by the test harness (e.g., `"tokens_generated": 0` with a space after the colon).
- **Batcher integration:** asynchronous prefetch + tokenization with order guarantees; micro-batch views from a ring buffer.
- **Bench & perf:** `make bench` uses prompts file and batching; `make perf-report` autodetects FlameGraph tools or accepts `STACKCOLLAPSE`/`FLAMEGRAPH` env overrides.
