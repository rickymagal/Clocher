# Design (CPU baseline + INT4 path)

This document describes the hot path, API boundaries, and precision modes.  
**Last updated:** 2025-10-24 21:00:48 UTC

## Process and boundaries
- Single binary `inference-engine`: create → generate → collect metrics → destroy.
- CLI prints exactly one JSON line per run so the Python harness can ingest results.
- No third‑party runtime dependencies; only `pthread` and `libm` (and CUDA for the GPU build).

## API surface (high level)
- `ie_engine_create(cfg)` → initializes state (weights, buffers, thread‑pool).
- `ie_engine_generate(prompt, max_new, params)` → produces tokens and updates metrics rings.
- `ie_engine_metrics(out)` → returns latency p50/p95, true TPS, **RSS peak**, KV hits/misses.
- `ie_engine_destroy()` → frees all resources.

## Hot path layout
- GEMV microkernel:
  - Generic scalar reference implementation.
  - AVX2/FMA path with light prefetch and blocked‑K packing.
- Activation:
  - `tanh` fast path with clamp (accuracy‑bounded), vector helper.
  - Optional fused bias + activation to reduce memory traffic.

## Precision modes

### Floating point
- FP32 baseline, optional BF16/FP16 round‑trip (accumulate FP32).

### INT8 PTQ (reference)
- Per‑tensor/per‑row scales (min‑max); (de)quant helpers; task gate in scripts.

### **NEW — INT4 PTQ (weight‑only)**
- **Format**: weights are **nibble‑packed** (2 values per byte). Each row (or group) carries a scale (and optional zero‑point if affine).
- **Packing**: INT4 packing integrates with the existing **pretranspose / blocked‑K** pipeline. Packing order: float → (optional) pretranspose → quantize to int4 → pack.
- **Dequantization**: fused in the matmul path; scale is broadcast per row (or group) to recover FP32 accumulators.
- **Manifests**: a `q4_manifest.json` enumerates which tensors use INT4 and their scale metadata. `scripts/hf_to_iebin.py` consumes this manifest via `--q4-map` to emit `model.ie.bin`/`.json`.
- **Selection**: at runtime choose `IE_PRECISION=int4w` (or `--precision int4w`), leaving activations in float. This is **bandwidth‑oriented** and preserves compute simplicity.

## Threading model
- Fixed thread‑pool over `pthread`, contiguous sharding, grainsize control.
- Affinity (Linux): `IE_TP_USE_AFFINITY=1` enables `auto|compact|scatter`.
- NUMA:
  - `scripts/set_numa.sh` can set OS policy (`interleave|node:X|strict`).
  - In‑repo probe reads `/sys/devices/system/node/online` to annotate reports.

## Layout and caching
- Blocked‑K packing with optional on‑disk caching (content‑addressed by shape + block size).
- CLI flag `--pretranspose` controls packing scope (`none|woh|wxh|all`). INT4 can be cached per layout variant.

## Metrics
- Per‑token latency ring (p50/p95).
- True TPS (`generated_tokens / wall_time_s`).
- **Peak RSS** (Linux `/proc/self/status:VmHWM` → MiB; fallback `getrusage`).
- KV hits/misses counter stubs aggregated per round.

## GPU integration (CUDA path)
- Device layer: `engine/src/devices/ie_device_cuda.cu`, kernels in `engine/src/kernels/ie_kernels_cuda.cu`.
- Build: `make build-cuda` → `build/inference-engine.cuda`.
- INT4 weight‑only support mirrors the CPU path; packing and scales are shared in `model.ie.json`.


---
## INT4 Weight-Only Path — Design Addendum (2025-10-24 21:04:23 UTC)

### Goals
- Introduce an **optional** path for *weight-only* INT4 (`int4w`) that:
  1) Packs HF weights into IEBIN using a **manifest-driven** policy.
  2) Leaves tokenization, shapes, and scheduling untouched.
  3) Preserves benchmark comparability via *work-touch* instrumentation.

### Design Choices
- **Manifest-driven packing**: `--q4-map` lets us target only GEMV-heavy matrices (attn/MLP projections) and keep sensitive tensors (embeddings, layernorms) in FP formats.
- **Separation of concerns**:
  - Storage precision (`IE_PRECISION`) is passed through `ie_engine_params_t::precision` untouched.
  - Host math precision (FP32/BF16/FP16) remains a separate selection.
- **Compat with harness**: The CLI accepts soft hints (`--device`, `--model-dir`, `--rounds`) so existing scripts don’t break.

### Data Flow (INT4 path)
HF shards → `hf_to_iebin.py --q4-map` → IEBIN (`model.ie.json` + `model.ie.bin` with INT4-packed tensors) → Runtime `IE_PRECISION=int4w` → GEMV kernels read packed weights (or dequant on load), semantics unchanged.

### Metrics Integrity
- The timed window is **only**: `ie_engine_generate(...)` + optional *work-touch* loop.
- RSS peak and KV counters are sampled **after** the window to avoid skewing TPS.

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
## Updates — 2025-11-10

### NUMA‑aware topology (`ie_topology`)
- Discovers sockets and CPUs from Linux sysfs and exposes a compact API:
  - `ie_topology_init/destroy` — lifetime.
  - `ie_topology_sockets()` — number of sockets.
  - `ie_topology_first_cpu_on_socket(s)` — first CPU index on socket `s` (for pinning).
- Integration:
  - Thread‑pool honors `IE_TP_USE_AFFINITY=1` with `AFFINITY=auto|compact|scatter`.
  - `ie_hot_replicate_by_socket(...)` uses topology for binding.

### Hot weights replication
- For “hot” tensors (frequently touched), we can replicate pages **per socket** to reduce remote memory hits.
- Implementation sketch:
  - `mmap` replica per socket → optional `madvise(..., MADV_WILLNEED)` → worker binding → socket‑local access.
  - Controlled by `IE_HOT_REPLICATE=1` and an optional cap `IE_HOT_REPL_LIMIT_MB`.
- Trade‑offs: additional memory; on single‑socket machines, the feature is a no‑op with negligible overhead.

### Activation precision hint
- Runtime hint `IE_ACT_PREC=int8|fp8|fp16|bf16|fp32` allows experimenting with lower‑precision activations while **keeping FP32 accumulators**.
- Weight precision remains independent (e.g., `IE_PRECISION=int4w` for nibble‑packed weights).

### Timing discipline (unchanged semantics)
- The benchmark **measured window** contains only `ie_engine_generate(...)` + optional work‑touch loop.
- All metrics collection (RSS, KV, JSON print) happens **after** the window.

### Example configurations
- INT4 weights, FP8 activations, NUMA‑aware with hot replication (CPU):
  ```bash
  export IE_REQUIRE_MODEL=1 IE_PRECISION=int4w IE_ACT_PREC=fp8
  export IE_BYTES_PER_TOKEN=$((64*1024*1024)) IE_STRIDE_BYTES=256 IE_VERIFY_TOUCH=1
  export IE_TP_USE_AFFINITY=1 AFFINITY=compact IE_HOT_REPLICATE=1
  PROMPTS=benchmarks/prompts_10..txt RUNS=3 make bench
  ```
- Same with CUDA:
  ```bash
  PROMPTS=benchmarks/prompts_10..txt RUNS=3 make bench-cuda
  ```
---
# Memory Phase Design Addendum (updated 2025-11-12 18:01:19 UTC)

## Goals
1. Reduce **DRAM traffic** on weight fetch via layout (`blocked‑K`), NUMA locality, and selective non‑temporal loads.
2. Enable **activation down‑precision** (INT8/FP8) orthogonally to weight storage (e.g., INT4 weight‑only).
3. Measure and visualize **spatial metrics** alongside TPS: MB/token, bytes touched, model coverage, effective bandwidth.

## Components
- **Topology & Binding (`ie_topology`)**: discovers sockets/CPUs from Linux sysfs. Exposes helpers for pinning (compact/scatter).
- **Hot replication (`replicate_hot.c`)**: optional per‑socket replicas for frequently‑touched weights; uses `mmap` + `madvise`.
- **Blocked‑K Pretranspose**: builds and caches a row‑major, **K‑blocked** layout; improves sequentiality and prefetch efficacy.
- **Streaming heuristics**: `IE_PREFETCH_DISTANCE`, `IE_NT_LOADS`, and `IE_NT_RATIO` drive prefetch and non‑temporal load decisions.
- **Activation precision**: runtime hint `IE_ACT_PREC` selects decode path (INT8 per‑tensor/per‑group, FP8 E4M3/E5M2). Accumulation is FP32.

## Measurement
The benchmark window includes **generation** and the **work‑touch** loop (controlled by `IE_BYTES_PER_TOKEN`, `IE_STRIDE_BYTES`).
After the window, the engine samples **peak RSS (VmHWM)**. The docs generator now derives:
- **MB/token** from `IE_BYTES_PER_TOKEN`
- **Total bytes touched** = `tokens_sum * bytes_per_token`
- **Coverage** = `bytes_per_token / size(model.ie.bin)`
- **Effective bandwidth** = `bytes_touched / wall_time` (GB/s)

## Backward Compatibility & Fallbacks
- Single‑socket hosts: topology collapses to one socket; binding is a no‑op.
- Unsupported backends ignore `IE_ACT_PREC`, `IE_NT_LOADS`, etc., without failing.
- When `IE_REQUIRE_MODEL` is unset, CI stub mode continues to work (no mmap or spatial metrics).

## Risks & Mitigations
- **Over‑eager NT loads** can hurt on small working sets → guard via `auto` heuristics and `IE_NT_RATIO` throttle.
- **Replica memory cost** on multi‑socket servers → gate with `IE_HOT_REPLICATE` and `IE_HOT_REPL_LIMIT_MB`.

# Block‑sparse weights (Phase 2, CPU only)

This chapter describes the **block‑sparse weights prototype** implemented in the
second phase of the memory work. The goal is to make *algorithmic sparsity*
concrete and measurable while keeping the dense engine and IEBIN format intact.

At this stage the implementation is **CPU‑only**, FP32‑only, and is exercised
through C unit tests and a dedicated microbenchmark. It is deliberately small
and self‑contained so that we can iterate on formats and policies without
destabilizing the main inference path.

## Goals

- Provide a **well‑defined in‑memory descriptor** for block‑sparse matrices
  (`ie_block_sparse_matrix_t`).
- Define a **compact on‑disk format** that can be produced by a simple C tool
  (`tools/convert_to_block_sparse.c`) and loaded by the engine.
- Implement a **reference GEMV kernel** for block‑sparse matrices on CPU:
  numerically equivalent to dense GEMV (modulo FP roundoff).
- Wire this into the device abstraction so that:
  - the CPU backend can execute block‑sparse GEMV; and
  - other backends can safely report “unimplemented” and trigger a CPU
    fallback.
- Keep the feature fully **opt‑in**:
  - No changes to `model.ie.bin` or the CLI.
  - No new runtime flags required for existing workflows.

## In‑memory layout: `ie_block_sparse_matrix_t`

The new public descriptor lives in `engine/include/sparse_format.h`:

- Global dimensions:
  - `rows`, `cols` — dense matrix shape.
  - `block_rows`, `block_cols` — tile shape (same for all blocks).
- Block‑row CSR (BSR) structure:
  - `n_block_rows` — number of block rows (typically
    `ceil(rows / block_rows)`).
  - `row_ptr` — length `n_block_rows + 1`; `row_ptr[br]..row_ptr[br+1]-1`
    indexes the non‑zero blocks (`nnzb` total).
  - `col_idx` — length `nnzb`; column index in block coordinates
    (`0..ceil(cols / block_cols)-1`).
- Values:
  - `values` — contiguous FP32 array of length
    `nnzb * block_rows * block_cols`, stored in row‑major order *within*
    each block.

Semantics:

- Conceptually the matrix is partitioned into `block_rows x block_cols`
  tiles. For each block row `br` we list all non‑zero tiles in ascending
  block column order.
- The actual dense dimensions are always taken from `rows` / `cols`. Tail
  blocks near the bottom/right edges are automatically clipped in the
  GEMV kernel.

The helpers in `sparse_format.h` cover:

- sanity checks for header fields;
- allocation/free helpers; and
- small utilities to compute block counts and strides.

## On‑disk format and loader (`engine/src/sparse_io.c`)

To keep experiments reproducible without touching the IEBIN format, we use a
separate, compact binary format for block‑sparse matrices:

- A fixed‑size header that records:
  - magic / version;
  - `rows`, `cols`, `block_rows`, `block_cols`;
  - `n_block_rows`, `nnzb`;
  - sizes of the three payload arrays (`row_ptr`, `col_idx`, `values`).
- Payload sections, tightly packed:
  1. `row_ptr` (`uint32_t` × `n_block_rows + 1`);
  2. `col_idx` (`uint32_t` × `nnzb`);
  3. `values` (`float` × `nnzb * block_rows * block_cols`).

The loader `ie_block_sparse_load(const char *path, ie_block_sparse_matrix_t *out)`
performs:

1. open + read header;
2. validate fields (non‑zero dimensions, monotonically increasing
   `row_ptr`, etc.);
3. allocate arrays for `row_ptr`, `col_idx`, `values`;
4. read the three payload sections; and
5. on success, fill `out` with owned pointers and return `IE_SPARSE_OK`.

Error paths:

- Any structural or I/O error returns a specific `ie_sparse_status_t`
  (e.g. `IE_SPARSE_ERR_IO`, `IE_SPARSE_ERR_FORMAT`).
- On error, partially allocated buffers are freed and `out` is zeroed.

## CPU kernel (`engine/src/gemm_block_sparse.c`)

The reference GEMV implementation:

```c
void ie_gemv_block_sparse_f32(const ie_block_sparse_matrix_t *m,
                              const float *x,
                              float *y,
                              const float *bias);
```

Design:

- Single‑threaded, straightforward loop structure:
  - iterate over block rows (`br`);
  - for each local row in the block (`local_r`), compute the dense row
    index `row = br * block_rows + local_r`;
  - iterate over non‑zero blocks in that block row using
    `row_ptr[br]..row_ptr[br+1]`;
  - for each block, compute the starting column and take an inner
    product between the block row and the corresponding slice of `x`.
- Tail safety:
  - rows with `row >= rows` are skipped;
  - columns with `col >= cols` are skipped inside the innermost loop.
- Bias:
  - `bias == NULL` is allowed; in that case the accumulator starts at
    `0.0f`;
  - otherwise we seed `acc` with `bias[row]`.

This function is small and easy to inspect, prioritizing correctness and
debuggability over clever micro‑optimizations. Higher‑level code can decide
whether and how to shard block rows across threads.

## Device abstraction (`engine/src/devices/ie_device_common.c`)

The `ie_device` vtable gains a new entry:

```c
int  (*gemv_block_sparse_f32)(void *self,
                              const ie_block_sparse_matrix_t *m,
                              const float *x,
                              float *y,
                              const float *bias);
```

Backend implementations:

- **CPU:**
  - `cpu_gemv_block_sparse_f32` simply forwards to
    `ie_gemv_block_sparse_f32`.
  - `ie_device_gemv_block_sparse_f32` (public helper) routes through the
    vtable and, on failure, can fall back to a CPU device if the current
    device is not already CPU.
- **CUDA / Level Zero:**
  - for this phase they return an “unimplemented” error code;
  - callers that use `ie_device_gemv_block_sparse_f32` will see the error
    and can choose to fall back to CPU.
  - No GPU kernels are required for tests to pass.

This layout keeps the public API stable while allowing future ADRs to add
true GPU implementations under the same method.

## Tools and tests

### Offline converter (`tools/convert_to_block_sparse.c`)

The converter takes a dense, row‑major FP32 matrix and produces the
block‑sparse binary format:

- Inputs:
  - path to a dense `.bin` (raw `float` array of shape `rows × cols`);
  - `rows`, `cols`, `block_rows`, `block_cols`;
  - optional threshold / sparsification policy (for now, the prototype
    typically uses *exact* sparsity patterns produced upstream).
- Steps:
  1. read dense matrix into memory;
  2. scan it by blocks, classify non‑zero blocks;
  3. build `row_ptr` / `col_idx`;
  4. emit the header + payload to an output path.

This keeps sparsification an explicit, offline step: the engine never
modifies weights at load time.

### C unit tests (`tests/c/test_block_sparse.c`)

The tests validate both the loader and the GEMV kernel:

- Small 4×4 and 8×8 matrices with hand‑written dense values and known
  products.
- Construction of the corresponding block‑sparse structures in memory,
  followed by calls to `ie_gemv_block_sparse_f32`.
- Round‑trip tests for the on‑disk format: write block‑sparse payloads,
  load via `ie_block_sparse_load`, compare against the in‑memory
  structures, and re‑run GEMV.

The test target is wired into `make test` alongside the existing C unit
tests so regressions are caught early.

### Microbenchmark (`benchmarks/src/microbench_gemv_block_sparse.c`)

A dedicated microbenchmark compares dense vs block‑sparse GEMV on CPU:

- Synthesizes a dense matrix and a block‑sparse version with a chosen
  sparsity pattern.
- Runs timed loops for both kernels and prints:
  - runtime, ns/element, and effective GB/s;
  - any basic correctness statistics (e.g. max absolute difference).
- Built and run via `make microbench-block-sparse`.

This is a **local** measurement tool; it does not depend on real model
weights or the CLI.

## Integration strategy and future work

This phase intentionally stops short of wiring block‑sparse weights into
`model.ie.bin` and the inference CLI:

- existing users see no change in behavior;
- sparsity experiments use their own binaries and scripts; and
- the code paths remain small enough to refactor without churn.

Follow‑up work (future ADRs) may cover:

- extending the IEBIN format (or adding a sidecar) to carry block‑sparse
  weights for specific layers;
- heuristics for which layers to sparsify (e.g. MLP vs attention);
- GPU kernels for popular architectures; and
- interaction with quantization and activation formats.

For now, the block‑sparse prototype gives us a solid, CPU‑only baseline to
reason about algorithmic sparsity independently of lower‑level memory and
topology tricks introduced in the first phase.


---

## Lossless Deduplication (Schema2): defaults + masks + exceptions

This project includes a **lossless** dedup path that reduces DRAM reads by reusing repeated weight content.
The runtime reconstructs bytes exactly (bit-for-bit) by applying a sparse patch stream over shared defaults.

### Artifacts

The runtime loader expects these files in the model directory (next to `model.ie.json` / `model.ie.bin`):

- `model.defaults.bin` — concatenated default payload bytes
- `model.masks.bin` — exception mask bits/bytes (identifies where defaults differ)
- `model.exceptions.bin` — exception bytes (dense list of the differing bytes)

In the current workflow, these are typically generated into `models/<MODEL>/dedup_out/` and then symlinked.

### Offline generation: extract defaults/masks/exceptions

The extractor takes:
- `tensor_map.json` (maps tensor names to on-disk byte ranges / sources)
- `groups.indices.json` (dedup grouping and indices)
and produces the three binary blobs:

```bash
python3 tools/dedup_extract_int4.py \
  --model-dir "models/gpt-oss-20b" \
  --tensor-map "models/gpt-oss-20b/dedup_out/tensor_map.json" \
  --groups "models/gpt-oss-20b/dedup_out/groups.indices.json" \
  --out-prefix "models/gpt-oss-20b/dedup_out/model"
```

Then link them into the model root:

```bash
cd models/gpt-oss-20b
ln -sf dedup_out/model.defaults.bin   model.defaults.bin
ln -sf dedup_out/model.masks.bin      model.masks.bin
ln -sf dedup_out/model.exceptions.bin model.exceptions.bin
cd ../..
```

### Runtime controls

- `IE_DEDUP=1` enables the dedup loader.
- `IE_DEDUP_STRICT=1` fails fast if dedup artifacts are missing or malformed.
- `IE_DEDUP_POLICY=lossless` selects the exact reconstruction policy (no approximations).
- `IE_DEDUP_CACHE_MB=<N>` configures the in-memory cache used by the dedup loader.
- `IE_DEDUP_DEBUG=1` enables verbose parsing/debug logs.

Strict runs should also enforce real IEBIN loading and anti-optimization work-touch:

- `IE_REQUIRE_MODEL=1`
- `IE_VERIFY_TOUCH=1`
- `IE_BYTES_PER_TOKEN=<nonzero>`
- `IE_STRIDE_BYTES=256` (default baseline)

### Schema2 JSON compatibility notes (model.ie.json)

The schema2 parser is strict about per-tensor metadata. When ingesting IEBIN metadata derived from HF shards,
ensure the following are true for each tensor entry:

- `dtype` is lowercase (`bf16`, `f16`, `f32`, `u8`, ...)
- file metadata uses the schema2 keys:
  - `file` (the backing shard filename)
  - `file_data_offset` (byte offset inside that shard)

If your `model.ie.json` uses `shard` / `shard_data_offset` instead, you can add aliases:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("models/gpt-oss-20b/model.ie.json")
j = json.loads(p.read_text(encoding="utf-8"))

t = j.get("tensors")
if not isinstance(t, list):
    raise SystemExit("ERROR: tensors is not a list")

for e in t:
    if not isinstance(e, dict):
        continue
    d = e.get("dtype")
    if isinstance(d, str):
        e["dtype"] = d.lower()
    if "file" not in e and "shard" in e and isinstance(e["shard"], str):
        e["file"] = e["shard"]
    if "file_data_offset" not in e and "shard_data_offset" in e and isinstance(e["shard_data_offset"], int):
        e["file_data_offset"] = e["shard_data_offset"]

p.write_text(json.dumps(j, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
print("OK: tensors =", len(t))
PY
```

You may see warnings like `unknown dtype=u8 ... (continuing)` for quantized auxiliary tensors (blocks/scales);
these are expected as long as the loader is configured to ignore/skip unknown storage dtypes where safe.

### Example strict run (CPU)

```bash
env -i \
  IE_REQUIRE_MODEL=1 \
  IE_VERIFY_TOUCH=1 \
  IE_BYTES_PER_TOKEN=67108864 \
  IE_STRIDE_BYTES=256 \
  IE_DEDUP=1 \
  IE_DEDUP_STRICT=1 \
  IE_DEDUP_POLICY=lossless \
  IE_DEDUP_CACHE_MB=512 \
  IE_DEDUP_DEBUG=1 \
  PRECISION=int4w \
  ./build/inference-engine \
    --device cpu \
    --model-dir "$(pwd)/models/gpt-oss-20b" \
    --prompts-file "$(pwd)/benchmarks/prompts_10.txt" \
    --max-new 8 --rounds 1
```
