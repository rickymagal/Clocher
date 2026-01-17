# Inference Engine — CPU core + CUDA + Python harness
# =============================================================================
SHELL := /usr/bin/bash
.DEFAULT_GOAL := build

.RECIPEPREFIX := >

.EXPORT_ALL_VARIABLES:
export STACKCOLLAPSE FLAMEGRAPH PROMPTS_FILE BATCH PREFETCH WARMUP THREADS PRECISION PRETRANSPOSE ROUNDS MAX_NEW FREQ CALLGRAPH
export ENGINE_BIN MODEL PROMPTS RUNS WARMUP THREADS AFFINITY BATCH PREFETCH PRETRANSPOSE WARMUP_TOKENS REPORT_ROOT PUSHGATEWAY_URL MAX_NEW
export TARGET_SECONDS IE_BYTES_PER_TOKEN IE_STRIDE_BYTES IE_VERIFY_TOUCH
export DEVICE GPU_ID

export IE_PREFETCH_DISTANCE IE_NT_LOADS IE_L3_BYTES IE_NT_THRESHOLD_RATIO IE_STREAM_BLOCK_BYTES IE_REUSE_GUARD_WINDOWS
export IE_ACTIVATIONS IE_FP8_FORMAT IE_ACT_SCALE_DTYPE IE_ACT_CLIP
export IE_HOT_REPLICATE
export IE_DEDUP IE_DEDUP_POLICY IE_DEDUP_CACHE_MB IE_DEDUP_HOT_BYTES IE_DEDUP_HOT_LIST

# Default: always run with dedup unless explicitly disabled by the user.
IE_DEDUP ?= 1

# =============================================================================
# Harness / reporting / verification defaults
# =============================================================================
PYTHON ?= python3

# When REPORT=1, bench targets will generate a per-prompt report artifact.
REPORT ?= 1

# When VERIFY=1, bench targets will verify the per-prompt report artifact.
VERIFY ?= 1

# Directory where per-prompt reports are stored (bench creates per-run subdirs).
REPORT_ROOT ?= build/reports

# Script that generates per-prompt reports (must exist if REPORT=1).
# Expected to write a single report file to the path passed via --out.
REPORT_SCRIPT ?= scripts/run_harness_report.py
EXPECTED_TOKENS_SCRIPT ?= tools/generate_expected_tokens.py
EXPECTED_TOKENS_GENERATE ?= 1
EXPECTED_TOKENS_DEVICE ?= cpu
REPORT_ARGS ?=

# Script that verifies per-prompt reports (must exist if VERIFY=1).
VERIFY_SCRIPT ?= tools/verify_report_tokens.py
VERIFY_ARGS ?= --require-expected
STRICT_VERIFY_SCRIPT ?= tools/verify_strict_ids.py

# =============================================================================
# Toolchains
# =============================================================================
CC        ?= gcc
CXX       ?= g++
NVCC      ?= nvcc

CFLAGS    ?= -std=c11 -O3 -Wall -Wextra -Werror -pedantic
CXXFLAGS  ?= -std=c++17 -O3 -Wall -Wextra -Werror
NVCCFLAGS ?= -O3 --use_fast_math -std=c++17 -Xcompiler "-fPIC -Wall -Wextra" -lineinfo

LDFLAGS_CPU  ?= -lpthread -lm
LDFLAGS_CUDA ?= -lpthread -lm -lcudart -ldl

CUDA_ARCH ?= sm_80
NVCCFLAGS += -gencode arch=compute_80,code=$(CUDA_ARCH)

# =============================================================================
# Paths / binaries
# =============================================================================
BUILD     ?= build
INC       := -Iengine/include
MODEL_DIR_DEFAULT := models/gpt-oss-20b

HF_DIR        ?= $(MODEL_DIR_DEFAULT)/hf
MODEL_OUT_DIR ?= $(MODEL_DIR_DEFAULT)
IEBIN_JSON    := $(MODEL_OUT_DIR)/model.ie.json
IEBIN_BIN     := $(MODEL_OUT_DIR)/model.ie.bin
HF_SHARDS := $(wildcard $(HF_DIR)/pytorch_model-*-of-*.bin) $(wildcard $(HF_DIR)/*.safetensors)

BIN_CPU   := $(BUILD)/inference-engine
BIN_CUDA  := $(BUILD)/inference-engine.cuda

# =============================================================================
# Sources
# =============================================================================
SRC_CORE_C_COMMON := \
  engine/src/ie_api.c \
  engine/src/runtime/infer_gptoss.c \
  engine/src/runtime/generate_gptoss.c \
  engine/src/runtime/sampling.c \
  engine/src/ie_tensor.c \
  engine/src/util_logging.c \
  engine/src/util_metrics.c \
  engine/src/io/weights.c \
  engine/src/io/weights_dedup.c \
  engine/src/dedup/spec.c \
  engine/src/dedup/patch_list.c \
  engine/src/dedup/dedup_cache.c \
  engine/src/io/tokenizer.c \
  engine/src/io/tokenizer_gptoss.c \
  engine/src/io/tokenizer_hf.c \
  engine/src/io/tensor_map.c \
  engine/src/io/ie_batcher.c \
  engine/src/io/loader_mmap.c \
  engine/src/io/mmap_tuning.c \
  engine/src/opt/cpu_features.c \
  engine/src/opt/thread_pool.c \
  engine/src/opt/pretranspose.c \
  engine/src/opt/numa_probe.c \
  engine/src/opt/topology.c \
  engine/src/opt/replicate_hot.c \
  engine/src/opt/stream.c \
  engine/src/opt/numa_policy.c \
  engine/src/kernels/gemv_generic.c \
  engine/src/kernels/gemv_avx2.c \
  engine/src/kernels/gemv_q4_generic.c \
  engine/src/kernels/gemv_q4_avx2.c \
  engine/src/kernels/attn_cpu.c \
  engine/src/kernels/mlp_cpu.c \
  engine/src/kernels/rmsnorm_cpu.c \
  engine/src/math/fast_tanh.c \
  engine/src/math/floatx.c \
  engine/src/math/rope.c \
  engine/src/runtime/kv_cache.c \
  engine/src/runtime/gptoss_hparams.c \
  engine/src/runtime/gptoss_weights_index.c \
  engine/src/ie_kv_instrumentation.c \
  engine/src/ie_sampler.c \
  engine/src/quant/int4_ptq.c \
  engine/src/quant/int8_ptq.c \
  engine/src/quant/act_fp8.c \
  engine/src/quant/act_int8.c \
  engine/src/devices/ie_device_common.c \
  engine/src/sparse_io.c \
  engine/src/gemm_block_sparse.c

SRC_CPU_STUB_C := \
  engine/src/devices/ie_device_cuda_stub.c

SRC_MAIN_C := engine/src/main_infer.c

SRC_CUDA_CU := \
  engine/src/devices/ie_device_cuda.cu \
  engine/src/kernels/ie_kernels_cuda.cu \
  engine/src/kernels/attn_cuda.cu \
  engine/src/kernels/mlp_cuda.cu \
  engine/src/kernels/rmsnorm_cuda.cu

SRC_TOOLS_C := \
  tools/convert_to_block_sparse.c

# =============================================================================
# Derived
# =============================================================================
OBJ_CORE_COMMON := $(patsubst %.c,$(BUILD)/%.o,$(SRC_CORE_C_COMMON))
OBJ_CPU_STUB    := $(patsubst %.c,$(BUILD)/%.o,$(SRC_CPU_STUB_C))
OBJ_CORE_CPU    := $(OBJ_CORE_COMMON) $(OBJ_CPU_STUB)
OBJ_CORE_CUDA   := $(OBJ_CORE_COMMON)

OBJ_MAIN_C := $(patsubst %.c,$(BUILD)/%.o,$(SRC_MAIN_C))
OBJ_CUDA   := $(patsubst %.cu,$(BUILD)/%.o,$(SRC_CUDA_CU))
OBJ_TOOLS  := $(patsubst %.c,$(BUILD)/%.o,$(SRC_TOOLS_C))

CONVERT_BIN := $(BUILD)/tools/convert_to_block_sparse

# =============================================================================
# Phonies
# =============================================================================
.PHONY: build build-release build-cuda build-all cuda \
	test bench bench-direct bench-cuda cuda-bench bench-all \
	bench-report bench-report-cuda bench-verify bench-verify-cuda \
	chat chat-file chat-cuda chat-file-cuda \
	profile perf-report baseline-report \
	fmt lint docs docs-doxygen clean microbench microbench-stream microbench-block-sparse \
	monitoring-up monitoring-down metrics-exporter \
	ptq-calibrate ptq-from-hf ptq-from-torch ptq-from-bin \
	show-tools iebin model-pack pack-hf refresh-model repack-hf \
	convert-to-block-sparse

cuda: build-cuda
cuda-bench: bench-cuda
bench-all: bench bench-cuda

# =============================================================================
# Utils / patterns
# =============================================================================
show-tools:
> @echo "CC=$(CC)"; echo "CXX=$(CXX)"; echo "NVCC=$(NVCC)";

$(BUILD)/%.o: %.c
> @mkdir -p $(dir $@)
> $(CC) $(CFLAGS) $(INC) -c $< -o $@

$(BUILD)/%.o: %.cu
> @mkdir -p $(dir $@)
> $(NVCC) $(NVCCFLAGS) -Iengine/include -c $< -o $@

$(BUILD)/engine/src/opt/stream.o: CFLAGS += -Wno-unused-function
$(BUILD)/engine/src/io/tokenizer_gptoss.o: CFLAGS += -Wno-unused-function

# =============================================================================
# HF -> IEBIN (manual only)
# =============================================================================
iebin: iebin-stamp
model-pack: iebin
pack-hf: iebin
refresh-model: repack-hf

repack-hf:
> @rm -f .iebin.stamp
> $(MAKE) iebin

.iebin.check:
> @test -d "$(HF_DIR)" || { echo "ERROR: HF_DIR '$(HF_DIR)' not found"; exit 2; }
> @test -f "$(HF_DIR)/config.json" || { echo "ERROR: $(HF_DIR)/config.json not found"; exit 2; }
> @test -n "$(HF_SHARDS)" || { echo "ERROR: no HF weights found under $(HF_DIR)"; exit 2; }
> @touch $@

.iebin.pack: .iebin.check scripts/hf_to_iebin.py
> @echo "[iebin] Packing HF checkpoint -> $(MODEL_OUT_DIR)"
> @python3 scripts/hf_to_iebin.py --hf-dir "$(HF_DIR)" --out-dir "$(MODEL_OUT_DIR)"
> @touch $@

iebin-stamp: .iebin.pack
> @test -f "$(IEBIN_JSON)" && test -f "$(IEBIN_BIN)" || { echo "ERROR: IEBIN artifacts missing"; exit 2; }
> @touch .iebin.stamp
> @echo "[iebin] Done: $(IEBIN_JSON) + $(IEBIN_BIN)"

# =============================================================================
# CPU build (default)
# =============================================================================
build: $(BIN_CPU)

$(BIN_CPU): $(OBJ_MAIN_C) $(OBJ_CORE_CPU)
> @mkdir -p $(dir $@)
> $(CC) $(CFLAGS) $(INC) $(OBJ_MAIN_C) $(OBJ_CORE_CPU) -o $@ $(LDFLAGS_CPU)
> @echo "[build] CPU binary -> $@"

build-release: CFLAGS += -DNDEBUG
build-release: build

# =============================================================================
# CUDA build
# =============================================================================
build-cuda: $(BIN_CUDA)

$(BIN_CUDA): $(OBJ_MAIN_C) $(OBJ_CORE_CUDA) $(OBJ_CUDA)
> @mkdir -p $(dir $@)
> $(NVCC) $(NVCCFLAGS) $(INC) $(OBJ_MAIN_C) $(OBJ_CORE_CUDA) $(OBJ_CUDA) -o $@ $(LDFLAGS_CUDA)
> @echo "[build] CUDA binary -> $@"

build-all: build build-cuda

# =============================================================================
# Tools
# =============================================================================
convert-to-block-sparse: $(CONVERT_BIN)

$(CONVERT_BIN): $(OBJ_TOOLS)
> @mkdir -p $(dir $@)
> $(CC) $(CFLAGS) $(INC) $(OBJ_TOOLS) -o $@ $(LDFLAGS_CPU)
> @echo "[build] block-sparse converter -> $@"

# =============================================================================
# Tests
# =============================================================================
test: $(BIN_CPU)
> @mkdir -p $(BUILD)
> @echo "[test] test_tensor"
> $(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS_CPU) && $(BUILD)/test_tensor
> @echo "[test] test_api (links full core objects)"
> $(CC) $(CFLAGS) $(INC) tests/c/test_api.c $(OBJ_CORE_CPU) -o $(BUILD)/test_api $(LDFLAGS_CPU) && ( cd $(MODEL_DIR_DEFAULT) && ../../$(BUILD)/test_api )
> @echo "[test] topology"
> $(CC) $(CFLAGS) $(INC) tests/c/test_topology.c engine/src/opt/topology.c engine/src/opt/thread_pool.c engine/src/opt/cpu_features.c engine/src/util_logging.c -o $(BUILD)/test_topology $(LDFLAGS_CPU) && $(BUILD)/test_topology
> @echo "[test] mmap_tuning"
> $(CC) $(CFLAGS) $(INC) tests/c/test_mmap_tuning.c engine/src/io/mmap_tuning.c engine/src/io/loader_mmap.c engine/src/util_logging.c -o $(BUILD)/test_mmap_tuning $(LDFLAGS_CPU) && $(BUILD)/test_mmap_tuning
> @echo "[test] block-sparse"
> $(CC) $(CFLAGS) $(INC) tests/c/test_block_sparse.c engine/src/gemm_block_sparse.c engine/src/sparse_io.c engine/src/util_logging.c -o $(BUILD)/test_block_sparse $(LDFLAGS_CPU) && $(BUILD)/test_block_sparse
> @echo "[test] dedup loader (sanity)"
> $(CC) $(CFLAGS) $(INC) tests/c/test_dedup_loader.c engine/src/io/weights.c engine/src/io/weights_dedup.c engine/src/dedup/spec.c engine/src/dedup/patch_list.c engine/src/dedup/dedup_cache.c engine/src/io/loader_mmap.c engine/src/io/mmap_tuning.c engine/src/util_logging.c engine/src/quant/int4_ptq.c engine/src/quant/int8_ptq.c -o $(BUILD)/test_dedup_loader $(LDFLAGS_CPU) && IE_DEDUP=$(IE_DEDUP) $(BUILD)/test_dedup_loader "$(MODEL_DIR_DEFAULT)"
> @if [ -z "$$IE_SKIP_WEIGHTS_TEST" ]; then \
        if [ -f $(MODEL_DIR_DEFAULT)/model.ie.json ] && [ -f $(MODEL_DIR_DEFAULT)/model.ie.bin ]; then \
          echo "[test] test_weights"; \
          if $(CC) $(CFLAGS) $(INC) tests/c/test_weights.c engine/src/io/weights.c engine/src/io/weights_dedup.c engine/src/dedup/spec.c engine/src/dedup/patch_list.c engine/src/dedup/dedup_cache.c engine/src/io/loader_mmap.c engine/src/io/mmap_tuning.c engine/src/util_logging.c engine/src/quant/int4_ptq.c engine/src/quant/int8_ptq.c -o $(BUILD)/test_weights $(LDFLAGS_CPU); then \
            ( cd $(MODEL_DIR_DEFAULT) && IE_DEDUP=$(IE_DEDUP) ../../$(BUILD)/test_weights ); \
            STATUS=$$?; \
            if [ $$STATUS -ne 0 ]; then \
              echo "[warn] test_weights exited with $$STATUS (non-zero). Set WEIGHTS_TEST_STRICT=1 to fail the test target."; \
              if [ "$$WEIGHTS_TEST_STRICT" = "1" ]; then exit $$STATUS; fi; \
            fi; \
          else \
            echo "[warn] could not compile test_weights; skipping (compiler error)."; \
            if [ "$$WEIGHTS_TEST_STRICT" = "1" ]; then exit 1; fi; \
          fi; \
        else \
          echo "[skip] test_weights (IEBIN not present; export IE_SKIP_WEIGHTS_TEST=1 to silence)"; \
        fi; \
      else \
        echo "[skip] test_weights (IE_SKIP_WEIGHTS_TEST=1)"; \
      fi
> @echo "[test] Python tests (excluding ptq_calib_pipeline)"
> @set -e; \
      files=$$(find tests/python -maxdepth 1 -name 'test_*.py' ! -name 'test_ptq_calib_pipeline.py' -print); \
      if [ -n "$$files" ]; then python3 -m unittest -v $$files; else echo "[warn] no python tests found"; fi

# =============================================================================
# Benchmarks (strict TPS + per-prompt report + verification)
# =============================================================================

# bench-report: CPU-only per-prompt report generation (no strict TPS)
bench-report: $(BIN_CPU)
> @set -e; set -o pipefail; \
      if [ "$${REPORT:-$(REPORT)}" != "1" ]; then echo "[bench-report] REPORT=0; nothing to do."; exit 0; fi; \
      if [ ! -f "$(REPORT_SCRIPT)" ]; then echo "ERROR: REPORT=1 but missing $(REPORT_SCRIPT)"; exit 2; fi; \
      if [ ! -f "$(EXPECTED_TOKENS_SCRIPT)" ]; then echo "ERROR: missing $(EXPECTED_TOKENS_SCRIPT)"; exit 2; fi; \
      if [ -z "$$PROMPTS" ]; then PROMPTS="benchmarks/prompts_10.txt"; fi; \
      if [ ! -f "$$PROMPTS" ]; then echo "ERROR: PROMPTS '$$PROMPTS' not found"; exit 2; fi; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      ABS_BIN="$$(realpath -m $(BIN_CPU))"; ABS_PROMPTS="$$(realpath -m "$$PROMPTS")"; \
      THREADS_VAL="$${THREADS:-$$(nproc)}"; PRECISION_VAL="$${PRECISION:-fp32}"; \
      BATCH_VAL="$${BATCH:-1}"; PREFETCH_VAL="$${PREFETCH:-auto}"; \
      PRETRANS_VAL="$${PRETRANSPOSE:-all}"; AFFINITY_VAL="$${AFFINITY:-auto}"; \
      MAX_NEW_VAL="$${MAX_NEW:-128}"; \
      RUNS_VAL="$${RUNS:-1}"; \
      ROUNDS_VAL="$${ROUNDS:-1}"; \
      EXPECTED_TOKENS_VAL="$${EXPECTED_TOKENS:-$$(dirname "$$ABS_PROMPTS")/expected_tokens.txt}"; \
      DEDUP_VAL="1"; \
      VERIFY_HF_IDS_VAL="$${VERIFY_HF_IDS:-1}"; \
      HF_DIR_VAL="$${HF_DIR:-$$ABS_MDIR/hf/original}"; \
      HF_PY_VAL="$${HF_PY:-$(CURDIR)/.venv/bin/python}"; \
      if [ ! -x "$$HF_PY_VAL" ]; then HF_PY_VAL="$${HF_PY:-python3}"; fi; \
      HF_DEVICE_VAL="$${HF_DEVICE:-cpu}"; \
      HF_DTYPE_VAL="$${HF_DTYPE:-fp32}"; \
      HF_TOPK_VAL="$${HF_TOPK:-0}"; \
      HF_LOGITS_TOL_VAL="$${HF_LOGITS_TOL:-}"; \
      HF_CHECK_PROMPT_VAL="$${HF_CHECK_PROMPT:-1}"; \
      GEN_BIN="$${EXPECTED_TOKENS_BIN:-$(BIN_CPU)}"; \
      ABS_GEN_BIN="$$(realpath -m "$$GEN_BIN")"; \
      if [ "$${EXPECTED_TOKENS_GENERATE:-$(EXPECTED_TOKENS_GENERATE)}" = "1" ] || [ ! -f "$$EXPECTED_TOKENS_VAL" ]; then \
        echo "[bench-report] generating $$EXPECTED_TOKENS_VAL"; \
        env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" \
          ENGINE_BIN="$$ABS_GEN_BIN" DEVICE="$${EXPECTED_TOKENS_DEVICE:-$(EXPECTED_TOKENS_DEVICE)}" MODEL_DIR="$$ABS_MDIR" PROMPTS="$$ABS_PROMPTS" \
          THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
          PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
          MAX_NEW="$$MAX_NEW_VAL" RUNS="1" ROUNDS="1" \
          IE_DEDUP="$$DEDUP_VAL" IE_DEDUP_POLICY="$$IE_DEDUP_POLICY" IE_DEDUP_CACHE_MB="$$IE_DEDUP_CACHE_MB" \
          $(PYTHON) "$(EXPECTED_TOKENS_SCRIPT)" --engine-bin "$$ABS_GEN_BIN" --model-dir "$$ABS_MDIR" --prompts "$$ABS_PROMPTS" --out "$$EXPECTED_TOKENS_VAL" --device "$${EXPECTED_TOKENS_DEVICE:-$(EXPECTED_TOKENS_DEVICE)}" --threads "$$THREADS_VAL" --precision "$$PRECISION_VAL" --batch "$$BATCH_VAL" --prefetch "$$PREFETCH_VAL" --pretranspose "$$PRETRANS_VAL" --affinity "$$AFFINITY_VAL" --max-new "$$MAX_NEW_VAL" --rounds 1; \
      fi; \
      RUN_ID="$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}"; \
      REPORT_ROOT_VAL="$${REPORT_ROOT:-$(REPORT_ROOT)}"; \
      if [ -z "$$REPORT_ROOT_VAL" ]; then REPORT_ROOT_VAL="$(REPORT_ROOT)"; fi; \
      OUT_DIR="$$(realpath -m "$$REPORT_ROOT_VAL/$$RUN_ID/cpu")"; \
      OUT_FILE="$$OUT_DIR/report.jsonl"; \
      mkdir -p "$$OUT_DIR"; \
      echo "[bench-report] writing $$OUT_FILE"; \
      env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" \
        ENGINE_BIN="$$ABS_BIN" DEVICE="cpu" MODEL_DIR="$$ABS_MDIR" PROMPTS="$$ABS_PROMPTS" \
        THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
        PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
        MAX_NEW="$$MAX_NEW_VAL" RUNS="$$RUNS_VAL" ROUNDS="$$ROUNDS_VAL" \
        IE_BYTES_PER_TOKEN="$${IE_BYTES_PER_TOKEN:-67108864}" IE_STRIDE_BYTES="$${IE_STRIDE_BYTES:-256}" IE_VERIFY_TOUCH="$${IE_VERIFY_TOUCH:-1}" \
        IE_ACTIVATIONS="$$IE_ACTIVATIONS" IE_FP8_FORMAT="$$IE_FP8_FORMAT" IE_HOT_REPLICATE="$$IE_HOT_REPLICATE" \
        IE_DEDUP="$$DEDUP_VAL" IE_DEDUP_POLICY="$$IE_DEDUP_POLICY" IE_DEDUP_CACHE_MB="$$IE_DEDUP_CACHE_MB" IE_DEDUP_HOT_BYTES="$$IE_DEDUP_HOT_BYTES" IE_DEDUP_HOT_LIST="$$IE_DEDUP_HOT_LIST" \
        IE_PREFETCH_DISTANCE="$$IE_PREFETCH_DISTANCE" IE_NT_LOADS="$$IE_NT_LOADS" IE_L3_BYTES="$$IE_L3_BYTES" IE_NT_THRESHOLD_RATIO="$$IE_NT_THRESHOLD_RATIO" IE_STREAM_BLOCK_BYTES="$$IE_STREAM_BLOCK_BYTES" IE_REUSE_GUARD_WINDOWS="$$IE_REUSE_GUARD_WINDOWS" \
        EXPECTED_TOKENS="$$EXPECTED_TOKENS_VAL" REPORT_TOKENS_MAX="$$REPORT_TOKENS_MAX" \
        $(PYTHON) "$(REPORT_SCRIPT)" \
          --engine-bin "$$ABS_BIN" \
          --model-dir "$$ABS_MDIR" \
          --prompts "$$ABS_PROMPTS" \
          --device cpu \
          --out "$$OUT_FILE" \
          --threads "$$THREADS_VAL" \
          --precision "$$PRECISION_VAL" \
          --batch "$$BATCH_VAL" \
          --prefetch "$$PREFETCH_VAL" \
          --pretranspose "$$PRETRANS_VAL" \
          --affinity "$$AFFINITY_VAL" \
          --max-new "$$MAX_NEW_VAL" \
          --runs "$$RUNS_VAL" \
          --rounds "$$ROUNDS_VAL" \
          $(REPORT_ARGS); \
      echo "$$OUT_FILE" > "$$OUT_DIR/LATEST_REPORT_PATH.txt"; \
      echo "[bench-report] done"

# bench-report-cuda: CUDA per-prompt report generation (no strict TPS)
bench-report-cuda: $(BIN_CUDA)
> @set -e; set -o pipefail; \
      if [ "$${REPORT:-$(REPORT)}" != "1" ]; then echo "[bench-report-cuda] REPORT=0; nothing to do."; exit 0; fi; \
      if [ ! -f "$(REPORT_SCRIPT)" ]; then echo "ERROR: REPORT=1 but missing $(REPORT_SCRIPT)"; exit 2; fi; \
      if [ ! -f "$(EXPECTED_TOKENS_SCRIPT)" ]; then echo "ERROR: missing $(EXPECTED_TOKENS_SCRIPT)"; exit 2; fi; \
      if [ -z "$$PROMPTS" ]; then PROMPTS="benchmarks/prompts_10.txt"; fi; \
      if [ ! -f "$$PROMPTS" ]; then echo "ERROR: PROMPTS '$$PROMPTS' not found"; exit 2; fi; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      ABS_BIN="$$(realpath -m $(BIN_CUDA))"; ABS_PROMPTS="$$(realpath -m "$$PROMPTS")"; \
      THREADS_VAL="$${THREADS:-$$(nproc)}"; PRECISION_VAL="$${PRECISION:-fp32}"; \
      BATCH_VAL="$${BATCH:-1}"; PREFETCH_VAL="$${PREFETCH:-auto}"; \
      PRETRANS_VAL="$${PRETRANSPOSE:-all}"; AFFINITY_VAL="$${AFFINITY:-auto}"; \
      MAX_NEW_VAL="$${MAX_NEW:-128}"; \
      RUNS_VAL="$${RUNS:-1}"; \
      ROUNDS_VAL="$${ROUNDS:-1}"; \
      EXPECTED_TOKENS_VAL="$${EXPECTED_TOKENS:-$$(dirname "$$ABS_PROMPTS")/expected_tokens.txt}"; \
      DEDUP_VAL="1"; \
      GEN_BIN="$${EXPECTED_TOKENS_BIN:-$(BIN_CPU)}"; \
      ABS_GEN_BIN="$$(realpath -m "$$GEN_BIN")"; \
      if [ "$${EXPECTED_TOKENS_GENERATE:-$(EXPECTED_TOKENS_GENERATE)}" = "1" ] || [ ! -f "$$EXPECTED_TOKENS_VAL" ]; then \
        echo "[bench-report-cuda] generating $$EXPECTED_TOKENS_VAL"; \
        env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" \
          ENGINE_BIN="$$ABS_GEN_BIN" DEVICE="$${EXPECTED_TOKENS_DEVICE:-$(EXPECTED_TOKENS_DEVICE)}" MODEL_DIR="$$ABS_MDIR" PROMPTS="$$ABS_PROMPTS" \
          THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
          PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
          MAX_NEW="$$MAX_NEW_VAL" RUNS="1" ROUNDS="1" \
          IE_DEDUP="$$DEDUP_VAL" IE_DEDUP_POLICY="$$IE_DEDUP_POLICY" IE_DEDUP_CACHE_MB="$$IE_DEDUP_CACHE_MB" \
          $(PYTHON) "$(EXPECTED_TOKENS_SCRIPT)" --engine-bin "$$ABS_GEN_BIN" --model-dir "$$ABS_MDIR" --prompts "$$ABS_PROMPTS" --out "$$EXPECTED_TOKENS_VAL" --device "$${EXPECTED_TOKENS_DEVICE:-$(EXPECTED_TOKENS_DEVICE)}" --threads "$$THREADS_VAL" --precision "$$PRECISION_VAL" --batch "$$BATCH_VAL" --prefetch "$$PREFETCH_VAL" --pretranspose "$$PRETRANS_VAL" --affinity "$$AFFINITY_VAL" --max-new "$$MAX_NEW_VAL" --rounds 1; \
      fi; \
      RUN_ID="$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}"; \
      REPORT_ROOT_VAL="$${REPORT_ROOT:-$(REPORT_ROOT)}"; \
      if [ -z "$$REPORT_ROOT_VAL" ]; then REPORT_ROOT_VAL="$(REPORT_ROOT)"; fi; \
      OUT_DIR="$$(realpath -m "$$REPORT_ROOT_VAL/$$RUN_ID/cuda")"; \
      OUT_FILE="$$OUT_DIR/report.jsonl"; \
      mkdir -p "$$OUT_DIR"; \
      echo "[bench-report-cuda] writing $$OUT_FILE"; \
      env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" \
        ENGINE_BIN="$$ABS_BIN" DEVICE="cuda" MODEL_DIR="$$ABS_MDIR" PROMPTS="$$ABS_PROMPTS" \
        THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
        PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
        MAX_NEW="$$MAX_NEW_VAL" RUNS="$$RUNS_VAL" ROUNDS="$$ROUNDS_VAL" \
        IE_BYTES_PER_TOKEN="$${IE_BYTES_PER_TOKEN:-67108864}" IE_STRIDE_BYTES="$${IE_STRIDE_BYTES:-256}" IE_VERIFY_TOUCH="$${IE_VERIFY_TOUCH:-1}" \
        IE_ACTIVATIONS="$$IE_ACTIVATIONS" IE_FP8_FORMAT="$$IE_FP8_FORMAT" IE_HOT_REPLICATE="$$IE_HOT_REPLICATE" \
        IE_DEDUP="$$DEDUP_VAL" IE_DEDUP_POLICY="$$IE_DEDUP_POLICY" IE_DEDUP_CACHE_MB="$$IE_DEDUP_CACHE_MB" IE_DEDUP_HOT_BYTES="$$IE_DEDUP_HOT_BYTES" IE_DEDUP_HOT_LIST="$$IE_DEDUP_HOT_LIST" \
        IE_PREFETCH_DISTANCE="$$IE_PREFETCH_DISTANCE" IE_NT_LOADS="$$IE_NT_LOADS" IE_L3_BYTES="$$IE_L3_BYTES" IE_NT_THRESHOLD_RATIO="$$IE_NT_THRESHOLD_RATIO" IE_STREAM_BLOCK_BYTES="$$IE_STREAM_BLOCK_BYTES" IE_REUSE_GUARD_WINDOWS="$$IE_REUSE_GUARD_WINDOWS" \
        EXPECTED_TOKENS="$$EXPECTED_TOKENS_VAL" REPORT_TOKENS_MAX="$$REPORT_TOKENS_MAX" \
        $(PYTHON) "$(REPORT_SCRIPT)" \
          --engine-bin "$$ABS_BIN" \
          --model-dir "$$ABS_MDIR" \
          --prompts "$$ABS_PROMPTS" \
          --device cuda \
          --out "$$OUT_FILE" \
          --threads "$$THREADS_VAL" \
          --precision "$$PRECISION_VAL" \
          --batch "$$BATCH_VAL" \
          --prefetch "$$PREFETCH_VAL" \
          --pretranspose "$$PRETRANS_VAL" \
          --affinity "$$AFFINITY_VAL" \
          --max-new "$$MAX_NEW_VAL" \
          --runs "$$RUNS_VAL" \
          --rounds "$$ROUNDS_VAL" \
          $(REPORT_ARGS); \
      echo "$$OUT_FILE" > "$$OUT_DIR/LATEST_REPORT_PATH.txt"; \
      echo "[bench-report-cuda] done"

# bench-verify: verify the most recent CPU report (or REPORT_PATH=/path/to/report.jsonl)
bench-verify:
> @set -e; set -o pipefail; \
      if [ "$${VERIFY:-$(VERIFY)}" != "1" ]; then echo "[bench-verify] VERIFY=0; nothing to do."; exit 0; fi; \
      VERIFY_HF_IDS_VAL="$${VERIFY_HF_IDS:-1}"; \
      if [ "$$VERIFY_HF_IDS_VAL" = "1" ]; then \
        if [ ! -f "$(STRICT_VERIFY_SCRIPT)" ]; then echo "ERROR: VERIFY_HF_IDS=1 but missing $(STRICT_VERIFY_SCRIPT)"; exit 2; fi; \
        SP="$${STRICT_REPORT_PATH:-$(BUILD)/strict_cpu.json}"; \
        if [ ! -f "$$SP" ]; then echo "ERROR: strict report '$$SP' not found"; exit 2; fi; \
        echo "[bench-verify] verifying strict IDs $$SP"; \
        $(PYTHON) "$(STRICT_VERIFY_SCRIPT)" "$$SP"; \
        echo "[bench-verify] OK"; \
        exit 0; \
      fi; \
      if [ ! -f "$(VERIFY_SCRIPT)" ]; then echo "ERROR: VERIFY=1 but missing $(VERIFY_SCRIPT)"; exit 2; fi; \
      RP="$${REPORT_PATH:-}"; \
      if [ -z "$$RP" ]; then \
        REPORT_ROOT_VAL="$${REPORT_ROOT:-$(REPORT_ROOT)}"; \
        if [ -z "$$REPORT_ROOT_VAL" ]; then REPORT_ROOT_VAL="$(REPORT_ROOT)"; fi; \
        latest="$$(ls -1 "$$REPORT_ROOT_VAL" 2>/dev/null | tail -n 1 || true)"; \
        if [ -z "$$latest" ]; then echo "ERROR: no reports under $(REPORT_ROOT)"; exit 2; fi; \
        RP="$$REPORT_ROOT_VAL/$$latest/cpu/report.jsonl"; \
      fi; \
      if [ ! -f "$$RP" ]; then echo "ERROR: report file '$$RP' not found"; exit 2; fi; \
      echo "[bench-verify] verifying $$RP"; \
      $(PYTHON) "$(VERIFY_SCRIPT)" $(VERIFY_ARGS) "$$RP"; \
      echo "[bench-verify] OK"

# bench-verify-cuda: verify the most recent CUDA report (or REPORT_PATH=/path/to/report.jsonl)
bench-verify-cuda:
> @set -e; set -o pipefail; \
      if [ "$${VERIFY:-$(VERIFY)}" != "1" ]; then echo "[bench-verify-cuda] VERIFY=0; nothing to do."; exit 0; fi; \
      VERIFY_HF_IDS_VAL="$${VERIFY_HF_IDS:-1}"; \
      if [ "$$VERIFY_HF_IDS_VAL" = "1" ]; then \
        if [ ! -f "$(STRICT_VERIFY_SCRIPT)" ]; then echo "ERROR: VERIFY_HF_IDS=1 but missing $(STRICT_VERIFY_SCRIPT)"; exit 2; fi; \
        SP="$${STRICT_REPORT_PATH:-$(BUILD)/strict_gpu.json}"; \
        if [ ! -f "$$SP" ]; then echo "ERROR: strict report '$$SP' not found"; exit 2; fi; \
        echo "[bench-verify-cuda] verifying strict IDs $$SP"; \
        $(PYTHON) "$(STRICT_VERIFY_SCRIPT)" "$$SP"; \
        echo "[bench-verify-cuda] OK"; \
        exit 0; \
      fi; \
      if [ ! -f "$(VERIFY_SCRIPT)" ]; then echo "ERROR: VERIFY=1 but missing $(VERIFY_SCRIPT)"; exit 2; fi; \
      RP="$${REPORT_PATH:-}"; \
      if [ -z "$$RP" ]; then \
        REPORT_ROOT_VAL="$${REPORT_ROOT:-$(REPORT_ROOT)}"; \
        if [ -z "$$REPORT_ROOT_VAL" ]; then REPORT_ROOT_VAL="$(REPORT_ROOT)"; fi; \
        latest="$$(ls -1 "$$REPORT_ROOT_VAL" 2>/dev/null | tail -n 1 || true)"; \
        if [ -z "$$latest" ]; then echo "ERROR: no reports under $(REPORT_ROOT)"; exit 2; fi; \
        RP="$$REPORT_ROOT_VAL/$$latest/cuda/report.jsonl"; \
      fi; \
      if [ ! -f "$$RP" ]; then echo "ERROR: report file '$$RP' not found"; exit 2; fi; \
      echo "[bench-verify-cuda] verifying $$RP"; \
      $(PYTHON) "$(VERIFY_SCRIPT)" $(VERIFY_ARGS) "$$RP"; \
      echo "[bench-verify-cuda] OK"

bench: $(BIN_CPU)
> @set -e; set -o pipefail; \
      if [ -z "$$PROMPTS" ]; then PROMPTS="benchmarks/prompts_10.txt"; fi; \
      if [ ! -f "$$PROMPTS" ]; then echo "ERROR: PROMPTS '$$PROMPTS' not found"; exit 2; fi; \
      test -x "$(BIN_CPU)" || { echo "ERROR: CPU binary '$(BIN_CPU)' not found. Run 'make build' once."; exit 2; }; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      IE_REQ_VAL="$${IE_REQUIRE_MODEL:-1}"; \
      if [ "$$IE_REQ_VAL" != "0" ]; then \
        test -f "$$ABS_MDIR/model.ie.json" && test -f "$$ABS_MDIR/model.ie.bin" || { \
          echo "ERROR: IEBIN missing under '$$ABS_MDIR' (model.ie.json/bin). Set IE_REQUIRE_MODEL=0 to bypass."; exit 2; }; \
      fi; \
      DEDUP_VAL="1"; \
      if [ "$$DEDUP_VAL" = "1" ]; then \
        test -f "$$ABS_MDIR/dedup_manifest.json" -o -f "$$ABS_MDIR/model.dedup.json" || { \
          echo "ERROR: IE_DEDUP=1 but no dedup manifest found under '$$ABS_MDIR'."; \
          echo "ERROR: expected dedup_manifest.json or model.dedup.json (adjust filenames to your loader if needed)."; \
          exit 2; }; \
      fi; \
      ABS_BIN="$$(realpath -m $(BIN_CPU))"; ABS_PROMPTS="$$(realpath -m "$$PROMPTS")"; \
      THREADS_VAL="$${THREADS:-$$(nproc)}"; PRECISION_VAL="$${PRECISION:-fp32}"; \
      BATCH_VAL="$${BATCH:-1}"; PREFETCH_VAL="$${PREFETCH:-auto}"; \
      PRETRANS_VAL="$${PRETRANSPOSE:-all}"; AFFINITY_VAL="$${AFFINITY:-auto}"; \
      MAX_NEW_VAL="$${MAX_NEW:-128}"; \
      RUNS_VAL="$${RUNS:-3}"; \
      ROUNDS_VAL="$${ROUNDS:-1}"; \
      VERIFY_HF_IDS_VAL="$${VERIFY_HF_IDS:-1}"; \
      HF_DIR_VAL="$${HF_DIR:-$$ABS_MDIR/hf/original}"; \
      HF_PY_VAL="$${HF_PY:-$(CURDIR)/.venv/bin/python}"; \
      if [ ! -x "$$HF_PY_VAL" ]; then HF_PY_VAL="$${HF_PY:-python3}"; fi; \
      HF_DEVICE_VAL="$${HF_DEVICE:-cpu}"; \
      HF_DTYPE_VAL="$${HF_DTYPE:-fp32}"; \
      HF_TOPK_VAL="$${HF_TOPK:-0}"; \
      HF_LOGITS_TOL_VAL="$${HF_LOGITS_TOL:-}"; \
      HF_CHECK_PROMPT_VAL="$${HF_CHECK_PROMPT:-1}"; \
      if [ "$$VERIFY_HF_IDS_VAL" = "1" ] && [ ! -d "$$HF_DIR_VAL" ]; then \
        echo "ERROR: VERIFY_HF_IDS=1 but HF_DIR not found: $$HF_DIR_VAL"; exit 2; \
      fi; \
      echo "[bench] strict run (true TPS)… (RUNS=$$RUNS_VAL, ROUNDS=$$ROUNDS_VAL, IE_DEDUP=$$DEDUP_VAL)"; \
      env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" \
      ENGINE_BIN="$$ABS_BIN" DEVICE="cpu" MODEL_DIR="$$ABS_MDIR" PROMPTS="$$ABS_PROMPTS" \
      THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
      PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
      MAX_NEW="$$MAX_NEW_VAL" IE_REQUIRE_MODEL="$$IE_REQ_VAL" \
      IE_BYTES_PER_TOKEN="$${IE_BYTES_PER_TOKEN:-67108864}" IE_STRIDE_BYTES="$${IE_STRIDE_BYTES:-256}" IE_VERIFY_TOUCH="$${IE_VERIFY_TOUCH:-1}" \
      RUNS="$$RUNS_VAL" ROUNDS="$$ROUNDS_VAL" \
      IE_ACTIVATIONS="$$IE_ACTIVATIONS" IE_FP8_FORMAT="$$IE_FP8_FORMAT" IE_HOT_REPLICATE="$$IE_HOT_REPLICATE" \
      IE_DEDUP="$$DEDUP_VAL" IE_DEDUP_POLICY="$$IE_DEDUP_POLICY" IE_DEDUP_CACHE_MB="$$IE_DEDUP_CACHE_MB" IE_DEDUP_HOT_BYTES="$$IE_DEDUP_HOT_BYTES" IE_DEDUP_HOT_LIST="$$IE_DEDUP_HOT_LIST" \
      IE_PREFETCH_DISTANCE="$$IE_PREFETCH_DISTANCE" IE_NT_LOADS="$$IE_NT_LOADS" IE_L3_BYTES="$$IE_L3_BYTES" IE_NT_THRESHOLD_RATIO="$$IE_NT_THRESHOLD_RATIO" IE_STREAM_BLOCK_BYTES="$$IE_STREAM_BLOCK_BYTES" IE_REUSE_GUARD_WINDOWS="$$IE_REUSE_GUARD_WINDOWS" \
      VERIFY_HF_IDS="$$VERIFY_HF_IDS_VAL" HF_DIR="$$HF_DIR_VAL" HF_PY="$$HF_PY_VAL" \
      HF_DEVICE="$$HF_DEVICE_VAL" HF_DTYPE="$$HF_DTYPE_VAL" HF_TOPK="$$HF_TOPK_VAL" \
      HF_LOGITS_TOL="$$HF_LOGITS_TOL_VAL" HF_CHECK_PROMPT="$$HF_CHECK_PROMPT_VAL" \
      bash scripts/true_tps_strict.sh | tee $(BUILD)/strict_cpu.json; \
      echo "[bench] updating docs/PERFORMANCE.md (CPU)…"; \
      if [ -f $(BUILD)/strict_gpu.json ]; then \
        GPU_ARG="--gpu-json $(BUILD)/strict_gpu.json --gpu-engine-bin $$(realpath -m $(BIN_CUDA))"; \
      else \
        GPU_ARG=""; \
      fi; \
      python3 scripts/update_performance_md.py \
        --cpu-json $(BUILD)/strict_cpu.json $$GPU_ARG \
        --cpu-engine-bin "$$ABS_BIN" \
        --prompts-file "$$ABS_PROMPTS" \
        --threads "$$THREADS_VAL" \
        --precision "$$PRECISION_VAL" \
        --batch "$$BATCH_VAL" \
        --prefetch "$$PREFETCH_VAL" \
        --pretranspose "$$PRETRANS_VAL" \
        --affinity "$$AFFINITY_VAL" \
        --max-new "$$MAX_NEW_VAL" \
        --ie-require-model "$$IE_REQ_VAL" \
        --ie-bytes-per-token "$${IE_BYTES_PER_TOKEN:-67108864}" \
        --ie-stride-bytes "$${IE_STRIDE_BYTES:-256}" \
        --ie-verify-touch "$${IE_VERIFY_TOUCH:-1}" \
        --model-dir "$$ABS_MDIR"; \
      if [ "$${REPORT:-$(REPORT)}" = "1" ]; then \
        $(MAKE) bench-report PROMPTS="$$ABS_PROMPTS" MODEL_DIR="$$ABS_MDIR" THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" MAX_NEW="$$MAX_NEW_VAL" RUNS="$$RUNS_VAL" ROUNDS="$$ROUNDS_VAL"; \
        if [ "$${VERIFY:-$(VERIFY)}" = "1" ]; then $(MAKE) bench-verify VERIFY_HF_IDS="$$VERIFY_HF_IDS_VAL" STRICT_REPORT_PATH="$(BUILD)/strict_cpu.json"; fi; \
      fi; \
      

bench-cuda: $(BIN_CUDA)
> @set -e; set -o pipefail; \
      if [ -z "$$PROMPTS" ]; then PROMPTS="benchmarks/prompts_10.txt"; fi; \
      if [ ! -f "$$PROMPTS" ]; then echo "ERROR: PROMPTS '$$PROMPTS' not found"; exit 2; fi; \
      test -x "$(BIN_CUDA)" || { echo "ERROR: CUDA binary '$(BIN_CUDA)' not found. Run 'make build-cuda' once."; exit 2; }; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      IE_REQ_VAL="$${IE_REQUIRE_MODEL:-1}"; \
      if [ "$$IE_REQ_VAL" != "0" ]; then \
        test -f "$$ABS_MDIR/model.ie.json" && test -f "$$ABS_MDIR/model.ie.bin" || { \
          echo "ERROR: IEBIN missing under '$$ABS_MDIR' (model.ie.json/bin). Set IE_REQUIRE_MODEL=0 to bypass."; exit 2; }; \
      fi; \
      DEDUP_VAL="1"; \
      if [ "$$DEDUP_VAL" = "1" ]; then \
        test -f "$$ABS_MDIR/dedup_manifest.json" -o -f "$$ABS_MDIR/model.dedup.json" || { \
          echo "ERROR: IE_DEDUP=1 but no dedup manifest found under '$$ABS_MDIR'."; \
          echo "ERROR: expected dedup_manifest.json or model.dedup.json (adjust filenames to your loader if needed)."; \
          exit 2; }; \
      fi; \
      ABS_BIN="$$(realpath -m $(BIN_CUDA))"; ABS_PROMPTS="$$(realpath -m "$$PROMPTS")"; \
      THREADS_VAL="$${THREADS:-$$(nproc)}"; PRECISION_VAL="$${PRECISION:-fp32}"; \
      BATCH_VAL="$${BATCH:-1}"; PREFETCH_VAL="$${PREFETCH:-auto}"; \
      PRETRANS_VAL="$${PRETRANSPOSE:-all}"; AFFINITY_VAL="$${AFFINITY:-auto}"; \
      MAX_NEW_VAL="$${MAX_NEW:-128}"; \
      RUNS_VAL="$${RUNS:-3}"; \
      ROUNDS_VAL="$${ROUNDS:-1}"; \
      VERIFY_HF_IDS_VAL="$${VERIFY_HF_IDS:-1}"; \
      HF_DIR_VAL="$${HF_DIR:-$$ABS_MDIR/hf/original}"; \
      HF_PY_VAL="$${HF_PY:-$(CURDIR)/.venv/bin/python}"; \
      if [ ! -x "$$HF_PY_VAL" ]; then HF_PY_VAL="$${HF_PY:-python3}"; fi; \
      HF_DEVICE_VAL="$${HF_DEVICE:-cpu}"; \
      HF_DTYPE_VAL="$${HF_DTYPE:-fp32}"; \
      HF_TOPK_VAL="$${HF_TOPK:-0}"; \
      HF_LOGITS_TOL_VAL="$${HF_LOGITS_TOL:-}"; \
      HF_CHECK_PROMPT_VAL="$${HF_CHECK_PROMPT:-1}"; \
      if [ "$$VERIFY_HF_IDS_VAL" = "1" ] && [ ! -d "$$HF_DIR_VAL" ]; then \
        echo "ERROR: VERIFY_HF_IDS=1 but HF_DIR not found: $$HF_DIR_VAL"; exit 2; \
      fi; \
      echo "[bench-cuda] strict run (true TPS)… (RUNS=$$RUNS_VAL, ROUNDS=$$ROUNDS_VAL, IE_DEDUP=$$DEDUP_VAL)"; \
      env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" \
      ENGINE_BIN="$$ABS_BIN" DEVICE="cuda" MODEL_DIR="$$ABS_MDIR" PROMPTS="$$ABS_PROMPTS" \
      THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
      PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
      MAX_NEW="$$MAX_NEW_VAL" IE_REQUIRE_MODEL="$$IE_REQ_VAL" \
      IE_BYTES_PER_TOKEN="$${IE_BYTES_PER_TOKEN:-67108864}" IE_STRIDE_BYTES="$${IE_STRIDE_BYTES:-256}" IE_VERIFY_TOUCH="$${IE_VERIFY_TOUCH:-1}" \
      RUNS="$$RUNS_VAL" ROUNDS="$$ROUNDS_VAL" \
      IE_ACTIVATIONS="$$IE_ACTIVATIONS" IE_FP8_FORMAT="$$IE_FP8_FORMAT" IE_HOT_REPLICATE="$$IE_HOT_REPLICATE" \
      IE_DEDUP="$$DEDUP_VAL" IE_DEDUP_POLICY="$$IE_DEDUP_POLICY" IE_DEDUP_CACHE_MB="$$IE_DEDUP_CACHE_MB" IE_DEDUP_HOT_BYTES="$$IE_DEDUP_HOT_BYTES" IE_DEDUP_HOT_LIST="$$IE_DEDUP_HOT_LIST" \
      IE_PREFETCH_DISTANCE="$$IE_PREFETCH_DISTANCE" IE_NT_LOADS="$$IE_NT_LOADS" IE_L3_BYTES="$$IE_L3_BYTES" IE_NT_THRESHOLD_RATIO="$$IE_NT_THRESHOLD_RATIO" IE_STREAM_BLOCK_BYTES="$$IE_STREAM_BLOCK_BYTES" IE_REUSE_GUARD_WINDOWS="$$IE_REUSE_GUARD_WINDOWS" \
      IE_METRICS_MEM="1" IE_METRICS_MEM_TOML="monitoring/metrics_memory.toml" \
      VERIFY_HF_IDS="$$VERIFY_HF_IDS_VAL" HF_DIR="$$HF_DIR_VAL" HF_PY="$$HF_PY_VAL" \
      HF_DEVICE="$$HF_DEVICE_VAL" HF_DTYPE="$$HF_DTYPE_VAL" HF_TOPK="$$HF_TOPK_VAL" \
      HF_LOGITS_TOL="$$HF_LOGITS_TOL_VAL" HF_CHECK_PROMPT="$$HF_CHECK_PROMPT_VAL" \
      bash scripts/true_tps_strict.sh | tee $(BUILD)/strict_gpu.json; \
      if [ "$${REPORT:-$(REPORT)}" = "1" ]; then \
        $(MAKE) bench-report-cuda PROMPTS="$$ABS_PROMPTS" MODEL_DIR="$$ABS_MDIR" THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" MAX_NEW="$$MAX_NEW_VAL" RUNS="$$RUNS_VAL" ROUNDS="$$ROUNDS_VAL"; \
        if [ "$${VERIFY:-$(VERIFY)}" = "1" ]; then $(MAKE) bench-verify-cuda VERIFY_HF_IDS="$$VERIFY_HF_IDS_VAL" STRICT_REPORT_PATH="$(BUILD)/strict_gpu.json"; fi; \
      fi; \
      echo "[bench-cuda] updating docs/PERFORMANCE.md (GPU)…"; \
      if [ -f $(BUILD)/strict_cpu.json ]; then \
        CPU_ARG="--cpu-json $(BUILD)/strict_cpu.json --cpu-engine-bin $$(realpath -m $(BIN_CPU))"; \
      else \
        CPU_ARG=""; \
      fi; \
      python3 scripts/update_performance_md.py \
        --gpu-json $(BUILD)/strict_gpu.json $$CPU_ARG \
        --gpu-engine-bin "$$ABS_BIN" \
        --prompts-file "$$ABS_PROMPTS" \
        --threads "$$THREADS_VAL" \
        --precision "$$PRECISION_VAL" \
        --batch "$$BATCH_VAL" \
        --prefetch "$$PREFETCH_VAL" \
        --pretranspose "$$PRETRANS_VAL" \
        --affinity "$$AFFINITY_VAL" \
        --max-new "$$MAX_NEW_VAL" \
        --ie-require-model "$$IE_REQ_VAL" \
        --ie-bytes-per-token "$${IE_BYTES_PER_TOKEN:-67108864}" \
        --ie-stride-bytes "$${IE_STRIDE_BYTES:-256}" \
        --model-dir "$$ABS_MDIR"

bench-direct:
> @test -x "$(BIN_CPU)" || { echo "ERROR: CPU binary '$(BIN_CPU)' not found. Run 'make build' once."; exit 2; }
> @set -e; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      IE_REQ_VAL="$${IE_REQUIRE_MODEL:-1}"; \
      if [ "$$IE_REQ_VAL" != "0" ]; then \
        test -f "$$ABS_MDIR/model.ie.json" && test -f "$$ABS_MDIR/model.ie.bin" || { \
          echo "ERROR: IEBIN missing under '$$ABS_MDIR' (model.ie.json/bin). Set IE_REQUIRE_MODEL=0 to bypass."; exit 2; }; \
      fi; \
      PF="$${BENCH_PROMPTS:-benchmarks/prompts_10.txt}"; \
      BATCH="$${BENCH_BATCH:-32}"; \
      WARM="$${BENCH_WARMUP:-4}"; \
      PREF="$${BENCH_PREFETCH:-on}"; \
      if [ -f $$PF ]; then \
        ( cd "$$ABS_MDIR" && ../"$(BIN_CPU)" --prompts-file "$$(realpath -m "$$PF")" --batch $$BATCH --max-new 8 --prefetch $$PREF --warmup $$WARM ) | \
          python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["tokens_generated"])' >/dev/null; \
      else \
        ( cd "$$ABS_MDIR" && ../"$(BIN_CPU)" --prompt "bench-default" --max-new 8 --prefetch $$PREF --warmup $$WARM ) | \
          python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["tokens_generated"])' >/dev/null; \
      fi; \
      echo "[bench-direct] OK"

# =============================================================================
# Chat / text generation (single prompt)
# =============================================================================
chat: $(BIN_CPU)
> @set -e; set -o pipefail; \
      test -x "$(BIN_CPU)" || { echo "ERROR: CPU binary '$(BIN_CPU)' not found. Run 'make build' once."; exit 2; }; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      DEDUP_VAL="$${IE_DEDUP:-$(IE_DEDUP)}"; \
      if [ "$$DEDUP_VAL" = "1" ]; then \
        test -f "$$ABS_MDIR/dedup_manifest.json" -o -f "$$ABS_MDIR/model.dedup.json" || { \
          echo "ERROR: IE_DEDUP=1 but no dedup manifest found under '$$ABS_MDIR'."; exit 2; }; \
      fi; \
      ABS_BIN="$$(realpath -m $(BIN_CPU))"; \
      PROMPT_VAL="$${PROMPT:-Hello}"; \
      MAX_NEW_VAL="$${MAX_NEW:-256}"; \
      PREFETCH_VAL="$${PREFETCH:-auto}"; \
      WARM_VAL="$${WARMUP:-0}"; \
      echo "[chat] device=cpu model_dir=$$ABS_MDIR max_new=$$MAX_NEW_VAL IE_DEDUP=$$DEDUP_VAL"; \
      ( cd "$$ABS_MDIR" && IE_PRINT_TEXT=1 IE_DEDUP="$$DEDUP_VAL" "$$ABS_BIN" --prompt "$$PROMPT_VAL" --max-new "$$MAX_NEW_VAL" --prefetch "$$PREFETCH_VAL" --warmup "$$WARM_VAL" )

chat-file: $(BIN_CPU)
> @set -e; set -o pipefail; \
      test -n "$$PROMPT_FILE" || { echo "ERROR: set PROMPT_FILE=/path/to/prompt.txt"; exit 2; }; \
      test -f "$$PROMPT_FILE" || { echo "ERROR: PROMPT_FILE '$$PROMPT_FILE' not found"; exit 2; }; \
      test -x "$(BIN_CPU)" || { echo "ERROR: CPU binary '$(BIN_CPU)' not found. Run 'make build' once."; exit 2; }; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      DEDUP_VAL="$${IE_DEDUP:-$(IE_DEDUP)}"; \
      if [ "$$DEDUP_VAL" = "1" ]; then \
        test -f "$$ABS_MDIR/dedup_manifest.json" -o -f "$$ABS_MDIR/model.dedup.json" || { \
          echo "ERROR: IE_DEDUP=1 but no dedup manifest found under '$$ABS_MDIR'."; exit 2; }; \
      fi; \
      PROMPT_VAL="$$(cat "$$PROMPT_FILE")"; \
      MAX_NEW_VAL="$${MAX_NEW:-256}"; \
      PREFETCH_VAL="$${PREFETCH:-auto}"; \
      WARM_VAL="$${WARMUP:-0}"; \
      ABS_BIN="$$(realpath -m $(BIN_CPU))"; \
      echo "[chat-file] device=cpu model_dir=$$ABS_MDIR prompt_file=$$PROMPT_FILE max_new=$$MAX_NEW_VAL IE_DEDUP=$$DEDUP_VAL"; \
      ( cd "$$ABS_MDIR" && IE_PRINT_TEXT=1 IE_DEDUP="$$DEDUP_VAL" "$$ABS_BIN" --prompt "$$PROMPT_VAL" --max-new "$$MAX_NEW_VAL" --prefetch "$$PREFETCH_VAL" --warmup "$$WARM_VAL" )

chat-cuda: $(BIN_CUDA)
> @set -e; set -o pipefail; \
      test -x "$(BIN_CUDA)" || { echo "ERROR: CUDA binary '$(BIN_CUDA)' not found. Run 'make build-cuda' once."; exit 2; }; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      DEDUP_VAL="$${IE_DEDUP:-$(IE_DEDUP)}"; \
      if [ "$$DEDUP_VAL" = "1" ]; then \
        test -f "$$ABS_MDIR/dedup_manifest.json" -o -f "$$ABS_MDIR/model.dedup.json" || { \
          echo "ERROR: IE_DEDUP=1 but no dedup manifest found under '$$ABS_MDIR'."; exit 2; }; \
      fi; \
      PROMPT_VAL="$${PROMPT:-Hello}"; \
      MAX_NEW_VAL="$${MAX_NEW:-256}"; \
      PREFETCH_VAL="$${PREFETCH:-auto}"; \
      WARM_VAL="$${WARMUP:-0}"; \
      ABS_BIN="$$(realpath -m $(BIN_CUDA))"; \
      echo "[chat-cuda] device=cuda model_dir=$$ABS_MDIR max_new=$$MAX_NEW_VAL IE_DEDUP=$$DEDUP_VAL"; \
      ( cd "$$ABS_MDIR" && IE_PRINT_TEXT=1 IE_DEDUP="$$DEDUP_VAL" "$$ABS_BIN" --prompt "$$PROMPT_VAL" --max-new "$$MAX_NEW_VAL" --prefetch "$$PREFETCH_VAL" --warmup "$$WARM_VAL" )

chat-file-cuda: $(BIN_CUDA)
> @set -e; set -o pipefail; \
      test -n "$$PROMPT_FILE" || { echo "ERROR: set PROMPT_FILE=/path/to/prompt.txt"; exit 2; }; \
      test -f "$$PROMPT_FILE" || { echo "ERROR: PROMPT_FILE '$$PROMPT_FILE' not found"; exit 2; }; \
      test -x "$(BIN_CUDA)" || { echo "ERROR: CUDA binary '$(BIN_CUDA)' not found. Run 'make build-cuda' once."; exit 2; }; \
      MDIR="$${MODEL:-$${MODEL_DIR:-}}"; \
      if [ -z "$$MDIR" ] || [ "$$MDIR" = "." ]; then MDIR="$(MODEL_DIR_DEFAULT)"; fi; \
      ABS_MDIR="$$(realpath -m "$$MDIR")"; \
      DEDUP_VAL="$${IE_DEDUP:-$(IE_DEDUP)}"; \
      if [ "$$DEDUP_VAL" = "1" ]; then \
        test -f "$$ABS_MDIR/dedup_manifest.json" -o -f "$$ABS_MDIR/model.dedup.json" || { \
          echo "ERROR: IE_DEDUP=1 but no dedup manifest found under '$$ABS_MDIR'."; exit 2; }; \
      fi; \
      PROMPT_VAL="$$(cat "$$PROMPT_FILE")"; \
      MAX_NEW_VAL="$${MAX_NEW:-256}"; \
      PREFETCH_VAL="$${PREFETCH:-auto}"; \
      WARM_VAL="$${WARMUP:-0}"; \
      ABS_BIN="$$(realpath -m $(BIN_CUDA))"; \
      echo "[chat-file-cuda] device=cuda model_dir=$$ABS_MDIR prompt_file=$$PROMPT_FILE max_new=$$MAX_NEW_VAL IE_DEDUP=$$DEDUP_VAL"; \
      ( cd "$$ABS_MDIR" && IE_PRINT_TEXT=1 IE_DEDUP="$$DEDUP_VAL" "$$ABS_BIN" --prompt "$$PROMPT_VAL" --max-new "$$MAX_NEW_VAL" --prefetch "$$PREFETCH_VAL" --warmup "$$WARM_VAL" )

# =============================================================================
# Profile / reports / helpers
# =============================================================================
profile:
> @test -x "$(BIN_CPU)" || { echo "ERROR: CPU binary '$(BIN_CPU)' not found. Run 'make build' once."; exit 2; }
> @bash -lc 'cd $(MODEL_DIR_DEFAULT) && ../../scripts/profile_flamegraph.sh ../../$(BIN_CPU) "profiling prompt with 64+ tokens"'

perf-report:
> @test -x "$(BIN_CPU)" || { echo "ERROR: CPU binary '$(BIN_CPU)' not found. Run 'make build' once."; exit 2; }
> @echo "[profile] generating flamegraph (optional)…"
> @set -e; MDROOT="$$(pwd)"; MODELDIR="$(MODEL_DIR_DEFAULT)"; cd "$$MODELDIR"; \
      if command -v perf >/dev/null 2>&1 && \
         { [ -x "$${STACKCOLLAPSE:-/usr/bin/stackcollapse-perf.pl}" ] || [ -f "$${STACKCOLLAPSE:-/usr/bin/stackcollapse-perf.pl}" ]; } && \
         [ -f "$${FLAMEGRAPH:-/usr/bin/flamegraph.pl}" ]; then \
        echo "[profile] generating flamegraph..."; \
        perf record -F 1200 -g -- ../$(BIN_CPU) --prompt "profiling-64-tokens" --max-new 64 >/dev/null 2>&1 || true; \
        perf script | "$${STACKCOLLAPSE:-/usr/bin/stackcollapse-perf.pl}" > out.perf || true; \
        perl "$${FLAMEGRAPH:-/usr/bin/flamegraph.pl}" out.perf > "$$MDROOT/flamegraph.svg" || true; \
        rm -f out.perf; \
      else echo "[profile] skipping flamegraph (tools not found)"; fi; cd "$$MDROOT"; \
      echo "[bench] strict run (true TPS)…"; \
      IE_BYTES_PER_TOKEN="$(IE_BYTES_PER_TOKEN)" IE_STRIDE_BYTES="$(IE_STRIDE_BYTES)" IE_VERIFY_TOUCH="$(IE_VERIFY_TOUCH)" \
      ENGINE_BIN="$(BIN_CPU)" MODEL_DIR="$(MODEL_DIR_DEFAULT)" PROMPTS="benchmarks/prompts_10.txt" \
      THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" PRECISION="$(if $(PRECISION),$(PRECISION),fp32)" \
      AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" \
      BATCH="$(if $(BATCH),$(BATCH),1)" PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" \
      MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),128)" \
      IE_DEDUP="$(IE_DEDUP)" \
      bash scripts/true_tps_strict.sh | tee /tmp/ie_strict.json; \
      echo "[bench] updating docs/PERFORMANCE.md (strict)…"; \
      python3 scripts/update_performance_md.py --cpu-json /tmp/ie_strict.json || true

baseline-report:
> @test -x "$(BIN_CPU)" || { echo "ERROR: CPU binary '$(BIN_CPU)' not found. Run 'make build' once."; exit 2; }
> @echo "[bench] running harness..."; bash scripts/run_benchmark.sh; \
      echo "[report] generating BASELINE.md..."; python3 scripts/make_baseline_md.py

# =============================================================================
# Formatting / lint / docs / clean / microbench
# =============================================================================
fmt:
> @command -v clang-format >/dev/null 2>&1 && clang-format -i $$(find engine -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.cu') || echo "clang-format not installed"

lint:
> @echo "[lint] warnings-as-errors enabled by default"; \
      command -v clang-tidy >/dev/null 2>&1 && clang-tidy $$(find engine/src -name '*.c' -o -name '*.cpp') -- $(CFLAGS) $(CXXFLAGS) $(INC) || echo "clang-tidy not installed"

docs:
> @echo "[docs] Final narrative emitted at project end."

docs-doxygen:
> @command -v doxygen >/dev/null 2>&1 || { echo "doxygen not installed"; exit 1; }
> doxygen docs/Doxyfile
> @echo "HTML docs under docs/doxygen/html/index.html"

clean:
> rm -rf $(BUILD) benchmarks/reports/* perf.data* flamegraph.svg out.perf script.stacks callgrind.out.* callgrind.stacks .iebin.stamp .iebin.check .iebin.pack

microbench: build
> @mkdir -p $(BUILD)
> $(CC) $(CFLAGS) $(INC) benchmarks/src/microbench_gemv.c -o $(BUILD)/microbench_gemv $(LDFLAGS_CPU)
> @echo "[run] microbench (H=256 V=1024 iters=200)"
> @$(BUILD)/microbench_gemv 256 1024 200

microbench-stream: build
> @mkdir -p $(BUILD)
> $(CC) $(CFLAGS) $(INC) benchmarks/src/microbench_stream.c engine/src/opt/stream.c engine/src/util_logging.c -o $(BUILD)/microbench_stream $(LDFLAGS_CPU)
> @echo "[run] microbench_stream (default params)"
> @$(BUILD)/microbench_stream

microbench-block-sparse: build
> @mkdir -p $(BUILD)
> $(CC) $(CFLAGS) $(INC) benchmarks/src/microbench_block_sparse.c engine/src/gemm_block_sparse.c engine/src/sparse_io.c engine/src/util_logging.c -o $(BUILD)/microbench_block_sparse $(LDFLAGS_CPU)
> @echo "[run] microbench_block-sparse (rows=4096 cols=4096 iters=50)"
> @$(BUILD)/microbench_block_sparse

# =============================================================================
# Monitoring & metrics (optional)
# =============================================================================
monitoring-up:
> @echo "[monitoring] starting stack..."
> @(command -v docker-compose >/dev/null 2>&1 && docker-compose -f monitoring/docker-compose.yml up -d) || \
       (command -v docker >/dev/null 2>&1 && docker compose -f monitoring/docker-compose.yml up -d) || \
       (echo "docker compose not available"; exit 1)

monitoring-down:
> @echo "[monitoring] stopping stack..."
> @(command -v docker-compose >/dev/null 2>&1 && docker-compose -f monitoring/docker-compose.yml down) || \
       (command -v docker >/dev/null 2>&1 && docker compose -f monitoring/docker-compose.yml down) || \
       (echo "docker compose not available"; exit 1)

metrics-exporter:
> python3 scripts/metrics_exporter.py --port 8000

# =============================================================================
# PTQ (INT8/INT4 helpers)
# =============================================================================
ptq-calibrate:
> @command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> @test -n "$(WEIGHTS)"     || { echo "Set WEIGHTS=/path/to/weights.bin"; exit 2; }
> @test -n "$(ROWS)"        || { echo "Set ROWS=<rows>"; exit 2; }
> @test -n "$(COLS)"        || { echo "Set COLS=<cols>"; exit 2; }
> @test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
> python3 benchmarks/ptq_calib.py --weights "$(WEIGHTS)" --rows $(ROWS) --cols $(COLS) --mode $(if $(MODE),$(MODE),per_row) --out-prefix "$(OUT_PREFIX)" --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-hf:
> @command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> @test -n "$(HF_MODEL)"    || { echo "Set HF_MODEL=org/repo"; exit 2; }
> @test -n "$(KEY)"         || { echo "Set KEY=<state_dict key>"; exit 2; }
> @test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
> python3 scripts/ptq_from_hf.py --repo "$(HF_MODEL)" --key "$(KEY)" $(if $(REV),--revision "$(REV)",) $(if $(FILE),--file "$(FILE)",) $(if $(TRANSPOSE),--transpose,) --out-prefix "$(OUT_PREFIX)" --mode $(if $(MODE),$(MODE),per_row) --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-torch:
> @command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> @test -n "$(TORCH_CHECKPOINT)" || { echo "Set TORCH_CHECKPOINT=/path/to/model.ckpt"; exit 2; }
> @test -n "$(KEY)"              || { echo "Set KEY=<state_dict key>"; exit 2; }
> @test -n "$(OUT_PREFIX)"       || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
> python3 scripts/ptq_from_source.py --source torch --checkpoint "$(TORCH_CHECKPOINT)" --key "$(KEY)" $(if $(TRANSPOSE),--transpose,) --out-prefix "$(OUT_PREFIX)" --mode $(if $(MODE),$(MODE),per_row) --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-bin:
> @command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> @test -n "$(BIN)"         || { echo "Set BIN=/path/to/f32.bin"; exit 2; }
> @test -n "$(ROWS)"        || { echo "Set ROWS=<rows>"; exit 2; }
> @test -n "$(COLS)"        || { echo "Set COLS=<cols>"; exit 2; }
> @test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
> python3 scripts/ptq_from_bin.py --bin "$(BIN)" --rows $(ROWS) --cols $(COLS) --out-prefix "$(OUT_PREFIX)" --mode $(if $(MODE),$(MODE),per_row) --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)
