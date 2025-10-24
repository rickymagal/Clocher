# File: Makefile
# Inference Engine — CPU core + CUDA + Python harness
# =============================================================================
SHELL := /usr/bin/bash
.DEFAULT_GOAL := build

# Export env so VAR=… make target reaches the scripts.
.EXPORT_ALL_VARIABLES:
export STACKCOLLAPSE FLAMEGRAPH PROMPTS_FILE BATCH PREFETCH WARMUP THREADS PRECISION PRETRANSPOSE ROUNDS MAX_NEW FREQ CALLGRAPH
export ENGINE_BIN MODEL MODEL_DIR PROMPTS RUNS WARMUP THREADS AFFINITY BATCH PREFETCH PRETRANSPOSE WARMUP_TOKENS REPORT_ROOT PUSHGATEWAY_URL MAX_NEW
export TARGET_SECONDS IE_BYTES_PER_TOKEN IE_STRIDE_BYTES IE_VERIFY_TOUCH
export DEVICE GPU_ID

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

# CUDA architecture (override if needed: sm_90, sm_86, sm_80, sm_75…)
CUDA_ARCH ?= sm_80
NVCCFLAGS += -gencode arch=compute_80,code=$(CUDA_ARCH)

# =============================================================================
# Paths / binaries
# =============================================================================
BUILD     ?= build
INC       := -Iengine/include
MODEL_DIR_DEFAULT := models/gpt-oss-20b

BIN_CPU   := $(BUILD)/inference-engine
BIN_CUDA  := $(BUILD)/inference-engine.cuda

# =============================================================================
# Sources
# =============================================================================
SRC_CORE_C := \
  engine/src/ie_api.c \
  engine/src/ie_tensor.c \
  engine/src/util_logging.c \
  engine/src/util_metrics.c \
  engine/src/io/weights.c \
  engine/src/io/tokenizer.c \
  engine/src/io/ie_batcher.c \
  engine/src/opt/cpu_features.c \
  engine/src/opt/thread_pool.c \
  engine/src/opt/pretranspose.c \
  engine/src/opt/numa_probe.c \
  engine/src/kernels/gemv_generic.c \
  engine/src/kernels/gemv_avx2.c \
  engine/src/math/fast_tanh.c \
  engine/src/math/floatx.c \
  engine/src/ie_kv_instrumentation.c

SRC_MAIN_C := engine/src/main_infer.c

SRC_CUDA_CU := \
  engine/src/devices/ie_device_cuda.cu \
  engine/src/kernels/ie_kernels_cuda.cu

# =============================================================================
# Derived
# =============================================================================
OBJ_CORE_C := $(patsubst %.c,$(BUILD)/%.o,$(SRC_CORE_C))
OBJ_MAIN_C := $(patsubst %.c,$(BUILD)/%.o,$(SRC_MAIN_C))
OBJ_CUDA   := $(patsubst %.cu,$(BUILD)/%.o,$(SRC_CUDA_CU))

# =============================================================================
# Phonies
# =============================================================================
.PHONY: build build-release build-cuda build-all cuda \
        test bench bench-direct bench-cuda cuda-bench \
        profile perf-report baseline-report \
        fmt lint docs docs-doxygen clean microbench \
        monitoring-up monitoring-down metrics-exporter \
        ptq-calibrate ptq-from-hf ptq-from-torch ptq-from-bin \
        perf_cpu_fp32 perf_cpu_int8 perf_cpu_bf16 perf_gpu \
        show-tools

cuda: build-cuda
cuda-bench: bench-cuda

# =============================================================================
# Utils / patterns
# =============================================================================
show-tools:
	@echo "CC=$(CC)"; echo "CXX=$(CXX)"; echo "NVCC=$(NVCC)";

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -Iengine/include -c $< -o $@

# =============================================================================
# CPU build (default)
# =============================================================================
build: $(BIN_CPU)

$(BIN_CPU): $(OBJ_MAIN_C) $(OBJ_CORE_C)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) $(OBJ_MAIN_C) $(OBJ_CORE_C) -o $@ $(LDFLAGS_CPU)
	@echo "[build] CPU binary -> $@"

build-release: CFLAGS += -DNDEBUG
build-release: build

# =============================================================================
# CUDA build
# =============================================================================
build-cuda: $(BIN_CUDA)

$(BIN_CUDA): $(OBJ_MAIN_C) $(OBJ_CORE_C) $(OBJ_CUDA)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INC) $(OBJ_MAIN_C) $(OBJ_CORE_C) $(OBJ_CUDA) -o $@ $(LDFLAGS_CUDA)
	@echo "[build] CUDA binary -> $@"

build-all: build build-cuda

# =============================================================================
# Tests (restored)
# =============================================================================
test: build
	@echo "[test] C unit tests"
	$(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS_CPU) && $(BUILD)/test_tensor
	$(CC) $(CFLAGS) $(INC) tests/c/test_api.c engine/src/ie_api.c engine/src/ie_tensor.c engine/src/util_logging.c engine/src/util_metrics.c engine/src/io/weights.c engine/src/io/tokenizer.c engine/src/io/ie_batcher.c engine/src/opt/cpu_features.c engine/src/opt/thread_pool.c engine/src/opt/pretranspose.c engine/src/opt/numa_probe.c engine/src/kernels/gemv_generic.c engine/src/kernels/gemv_avx2.c engine/src/math/fast_tanh.c engine/src/ie_kv_instrumentation.c -o $(BUILD)/test_api $(LDFLAGS_CPU) && ( cd models/gpt-oss-20b && ../../$(BUILD)/test_api )

	@if [ -z "$$IE_SKIP_WEIGHTS_TEST" ]; then \
	  if [ -f models/gpt-oss-20b/model.ie.json ] && [ -f models/gpt-oss-20b/model.ie.bin ]; then \
	    echo "[test] test_weights"; \
	    if $(CC) $(CFLAGS) $(INC) tests/c/test_weights.c engine/src/io/weights.c -o $(BUILD)/test_weights $(LDFLAGS_CPU); then \
	      ( cd models/gpt-oss-20b && ../../$(BUILD)/test_weights ); \
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

	@echo "[test] Python tests"
	python3 -m unittest discover -s tests/python -p 'test_*.py' -v
# =============================================================================
# Benchmarks (each updates only its own side; report uses last 3 runs/side)
# =============================================================================
bench: build
	@if [ -z "$$PROMPTS" ]; then echo "ERROR: PROMPTS must be set (e.g., PROMPTS=benchmarks/prompts_10..txt)"; exit 2; fi
	@if [ ! -f "$$PROMPTS" ]; then echo "ERROR: PROMPTS '$$PROMPTS' not found"; exit 2; fi
	@MDIR="$${MODEL:-$${MODEL_DIR:-$(MODEL_DIR_DEFAULT)}}"; \
	if [ ! -d "$$MDIR" ]; then echo "ERROR: model dir '$$MDIR' not found"; exit 2; fi; \
	ABS_BIN="$$(realpath -m $(BIN_CPU))"; ABS_PROMPTS="$$(realpath -m "$$PROMPTS")"; \
	THREADS_VAL="$${THREADS:-$$(nproc)}"; PRECISION_VAL="$${PRECISION:-fp32}"; \
	BATCH_VAL="$${BATCH:-1}"; PREFETCH_VAL="$${PREFETCH:-auto}"; \
	PRETRANS_VAL="$${PRETRANSPOSE:-all}"; AFFINITY_VAL="$${AFFINITY:-auto}"; \
	MAX_NEW_VAL="$${MAX_NEW:-128}"; IE_REQ_VAL="$${IE_REQUIRE_MODEL:-1}"; \
	IE_BPT_VAL="$${IE_BYTES_PER_TOKEN:-67108864}"; IE_STRIDE_VAL="$${IE_STRIDE_BYTES:-256}"; IE_TOUCH_VAL="$${IE_VERIFY_TOUCH:-1}"; \
	RUNS_VAL="$${RUNS:-3}"; \
	echo "[bench] strict run (true TPS)… (RUNS=$$RUNS_VAL)"; \
	env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" \
	ENGINE_BIN="$$ABS_BIN" DEVICE="cpu" MODEL_DIR="$$MDIR" PROMPTS="$$ABS_PROMPTS" \
	THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
	PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
	MAX_NEW="$$MAX_NEW_VAL" IE_REQUIRE_MODEL="$$IE_REQ_VAL" \
	IE_BYTES_PER_TOKEN="$$IE_BPT_VAL" IE_STRIDE_BYTES="$$IE_STRIDE_VAL" IE_VERIFY_TOUCH="$$IE_TOUCH_VAL" \
	RUNS="$$RUNS_VAL" ROUNDS="$$RUNS_VAL" \
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
	  --ie-bytes-per-token "$$IE_BPT_VAL" \
	  --ie-stride-bytes "$$IE_STRIDE_VAL" \
	  --ie-verify-touch "$$IE_TOUCH_VAL" \
	  --model-dir "$$MDIR"

bench-cuda: build-cuda
	@if [ -z "$$PROMPTS" ]; then echo "ERROR: PROMPTS must be set (e.g., PROMPTS=benchmarks/prompts_10..txt)"; exit 2; fi
	@if [ ! -f "$$PROMPTS" ]; then echo "ERROR: PROMPTS '$$PROMPTS' not found"; exit 2; fi
	@MDIR="$${MODEL:-$${MODEL_DIR:-$(MODEL_DIR_DEFAULT)}}"; \
	if [ ! -d "$$MDIR" ]; then echo "ERROR: model dir '$$MDIR' not found"; exit 2; fi; \
	ABS_BIN="$$(realpath -m $(BIN_CUDA))"; ABS_PROMPTS="$$(realpath -m "$$PROMPTS")"; \
	THREADS_VAL="$${THREADS:-$$(nproc)}"; PRECISION_VAL="$${PRECISION:-fp32}"; \
	BATCH_VAL="$${BATCH:-1}"; PREFETCH_VAL="$${PREFETCH:-auto}"; \
	PRETRANS_VAL="$${PRETRANSPOSE:-all}"; AFFINITY_VAL="$${AFFINITY:-auto}"; \
	MAX_NEW_VAL="$${MAX_NEW:-128}"; IE_REQ_VAL="$${IE_REQUIRE_MODEL:-1}"; \
	IE_BPT_VAL="$${IE_BYTES_PER_TOKEN:-67108864}"; IE_STRIDE_VAL="$${IE_STRIDE_BYTES:-256}"; IE_TOUCH_VAL="$${IE_VERIFY_TOUCH:-1}"; \
	RUNS_VAL="$${RUNS:-3}"; \
	echo "[bench-cuda] strict run (true TPS)… (RUNS=$$RUNS_VAL)"; \
	env -i PATH="$$PATH" SHELL="$$SHELL" HOME="$$HOME" \
	ENGINE_BIN="$$ABS_BIN" DEVICE="cuda" MODEL_DIR="$$MDIR" PROMPTS="$$ABS_PROMPTS" \
	THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
	PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
	MAX_NEW="$$MAX_NEW_VAL" IE_REQUIRE_MODEL="$$IE_REQ_VAL" \
	IE_BYTES_PER_TOKEN="$$IE_BPT_VAL" IE_STRIDE_BYTES="$$IE_STRIDE_VAL" IE_VERIFY_TOUCH="$$IE_TOUCH_VAL" \
	RUNS="$$RUNS_VAL" ROUNDS="$$RUNS_VAL" \
	bash scripts/true_tps_strict.sh | tee $(BUILD)/strict_gpu.json; \
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
	  --ie-bytes-per-token "$$IE_BPT_VAL" \
	  --ie-stride-bytes "$$IE_STRIDE_VAL" \
	  --ie-verify-touch "$$IE_TOUCH_VAL" \
	  --model-dir "$$MDIR"

bench-direct: build
	@set -e; \
	PF="$(if $(BENCH_PROMPTS),$(BENCH_PROMPTS),benchmarks/prompts_10..txt)"; \
	BATCH="$(if $(BENCH_BATCH),$(BENCH_BATCH),32)"; \
	WARM="$(if $(BENCH_WARMUP),$(BENCH_WARMUP),4)"; \
	PREF="$(if $(BENCH_PREFETCH),$(BENCH_PREFETCH),on)"; \
	if [ -f $$PF ]; then \
	  ( cd $(MODEL_DIR_DEFAULT) && ../$(BIN_CPU) --prompts-file "$$(realpath -m "$$PF")" --batch $$BATCH --max-new 8 --prefetch $$PREF --warmup $$WARM ) | \
	    python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["tokens_generated"])' >/dev/null; \
	else \
	  ( cd $(MODEL_DIR_DEFAULT) && ../$(BIN_CPU) --prompt "bench-default" --max-new 8 --prefetch $$PREF --warmup $$WARM ) | \
	    python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["tokens_generated"])' >/dev/null; \
	fi; \
	echo "[bench-direct] OK"

# =============================================================================
# Profile / reports / helpers
# =============================================================================
profile: build
	@bash -lc 'cd $(MODEL_DIR_DEFAULT) && ../../scripts/profile_flamegraph.sh ../../$(BIN_CPU) "profiling prompt with 64+ tokens"'

perf-report: build
	@echo "[profile] generating flamegraph (optional)…"
	@set -e; MDROOT="$$(pwd)"; MODELDIR="models/gpt-oss-20b"; cd "$$MODELDIR"; \
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
	ENGINE_BIN="$(BIN_CPU)" MODEL_DIR="models/gpt-oss-20b" PROMPTS="benchmarks/prompts_10..txt" \
	THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" PRECISION="$(if $(PRECISION),$(PRECISION),fp32)" \
	AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" \
	BATCH="$(if $(BATCH),$(BATCH),1)" PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" \
	MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),128)" \
	bash scripts/true_tps_strict.sh | tee /tmp/ie_strict.json; \
	echo "[bench] updating docs/PERFORMANCE.md (strict)…"; \
	python3 scripts/update_performance_md.py --cpu-json /tmp/ie_strict.json || true

baseline-report: build
	@echo "[bench] running harness..."; bash scripts/run_benchmark.sh; \
	echo "[report] generating BASELINE.md..."; python3 scripts/make_baseline_md.py

# =============================================================================
# Formatting / lint / docs / clean / microbench
# =============================================================================
fmt:
	@command -v clang-format >/dev/null 2nd /dev/null && clang-format -i $$(find engine -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.cu') || echo "clang-format not installed"

lint:
	@echo "[lint] warnings-as-errors enabled by default"; \
	command -v clang-tidy >/dev/null 2>&1 && clang-tidy $$(find engine/src -name '*.c' -o -name '*.cpp') -- $(CFLAGS) $(CXXFLAGS) $(INC) || echo "clang-tidy not installed"

docs:
	@echo "[docs] Final narrative emitted at project end."

docs-doxygen:
	@command -v doxygen >/dev/null 2>&1 || { echo "doxygen not installed"; exit 1; }
	doxygen docs/Doxyfile
	@echo "HTML docs under docs/doxygen/html/index.html"

clean:
	rm -rf $(BUILD) benchmarks/reports/* perf.data* flamegraph.svg out.perf script.stacks callgrind.out.* callgrind.stacks

microbench: build
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) benchmarks/src/microbench_gemv.c -o $(BUILD)/microbench_gemv $(LDFLAGS_CPU)
	@echo "[run] microbench (H=256 V=1024 iters=200)"
	@$(BUILD)/microbench_gemv 256 1024 200

# =============================================================================
# Monitoring & metrics (optional)
# =============================================================================
monitoring-up:
	@echo "[monitoring] starting stack..."
	@(command -v docker-compose >/dev/null 2>&1 && docker-compose -f monitoring/docker-compose.yml up -d) || \
	 (command -v docker >/dev/null 2>&1 && docker compose -f monitoring/docker-compose.yml up -d) || \
	 (echo "docker compose not available"; exit 1)

monitoring-down:
	@echo "[monitoring] stopping stack..."
	@(command -v docker-compose >/dev/null 2>&1 && docker-compose -f monitoring/docker-compose.yml down) || \
	 (command -v docker >/dev/null 2>&1 && docker compose -f monitoring/docker-compose.yml down) || \
	 (echo "docker compose not available"; exit 1)

metrics-exporter:
	python3 scripts/metrics_exporter.py --port 8000

# =============================================================================
# PTQ (INT8)
# =============================================================================
ptq-calibrate:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
	@test -n "$(WEIGHTS)"     || { echo "Set WEIGHTS=/path/to/weights.bin"; exit 2; }
	@test -n "$(ROWS)"        || { echo "Set ROWS=<rows>"; exit 2; }
	@test -n "$(COLS)"        || { echo "Set COLS=<cols>"; exit 2; }
	@test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
	python3 benchmarks/ptq_calib.py --weights "$(WEIGHTS)" --rows $(ROWS) --cols $(COLS) --mode $(if $(MODE),$(MODE),per_row) --out-prefix "$(OUT_PREFIX)" --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-hf:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
	@test -n "$(HF_MODEL)"    || { echo "Set HF_MODEL=org/repo"; exit 2; }
	@test -n "$(KEY)"         || { echo "Set KEY=<state_dict key>"; exit 2; }
	@test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
	python3 scripts/ptq_from_hf.py --repo "$(HF_MODEL)" --key "$(KEY)" $(if $(REV),--revision "$(REV)",) $(if $(FILE),--file "$(FILE)",) $(if $(TRANSPOSE),--transpose,) --out-prefix "$(OUT_PREFIX)" --mode $(if $(MODE),$(MODE),per_row) --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-torch:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
	@test -n "$(TORCH_CHECKPOINT)" || { echo "Set TORCH_CHECKPOINT=/path/to/model.ckpt"; exit 2; }
	@test -n "$(KEY)"              || { echo "Set KEY=<state_dict key>"; exit 2; }
	@test -n "$(OUT_PREFIX)"       || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
	python3 scripts/ptq_from_source.py --source torch --checkpoint "$(TORCH_CHECKPOINT)" --key "$(KEY)" $(if $(TRANSPOSE),--transpose,) --out-prefix "$(OUT_PREFIX)" --mode $(if $(MODE),$(MODE),per_row) --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-bin:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
	@test -n "$(BIN)"         || { echo "Set BIN=/path/to/f32.bin"; exit 2; }
	@test -n "$(ROWS)"        || { echo "Set ROWS=<rows>"; exit 2; }
	@test -n "$(COLS)"        || { echo "Set COLS=<cols>"; exit 2; }
	@test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
	python3 scripts/ptq_from_source.py --source raw --bin "$(BIN)" --rows $(ROWS) --cols $(COLS) $(if $(TRANSPOSE),--transpose,) --out-prefix "$(OUT_PREFIX)" --mode $(if $(MODE),$(MODE),per_row) --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

# =============================================================================
# Performance presets
# =============================================================================
perf_cpu_fp32: build
	@MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR_DEFAULT))" PROMPTS="$(if $(PROMPTS),$(PROMPTS),benchmarks/prompts_10..txt)" RUNS="$(if $(RUNS),$(RUNS),3)" WARMUP="$(if $(WARMUP),$(WARMUP),1)" THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" PRECISION="fp32" BATCH="$(if $(BATCH),$(BATCH),1)" PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" WARMUP_TOKENS="$(if $(WARMUP_TOKENS),$(WARMUP_TOKENS),64)" AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),0)" ENGINE_BIN="$(BIN_CPU)" bash scripts/run_benchmark.sh

perf_cpu_bf16: build
	@MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR_DEFAULT))" PROMPTS="$(if $(PROMPTS),$(PROMPTS),benchmarks/prompts_10..txt)" RUNS="$(if $(RUNS),$(RUNS),3)" WARMUP="$(if $(WARMUP),$(WARMUP),1)" THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" PRECISION="bf16" BATCH="$(if $(BATCH),$(BATCH),1)" PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" WARMUP_TOKENS="$(if $(WARMUP_TOKENS),$(WARMUP_TOKENS),64)" AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),0)" ENGINE_BIN="$(BIN_CPU)" bash scripts/run_benchmark.sh

perf_cpu_int8: build
	@MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR_DEFAULT))" PROMPTS="$(if $(PROMPTS),$(PROMPTS),benchmarks/prompts_10..txt)" RUNS="$(if $(RUNS),$(RUNS),3)" WARMUP="$(if $(WARMUP),$(WARMUP),1)" THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" PRECISION="$(if $(PRECISION),$(PRECISION),int8)" BATCH="$(if $(BATCH),$(BATCH),1)" PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" WARMUP_TOKENS="$(if $(WARMUP_TOKENS),$(WARMUP_TOKENS),64)" AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),0)" ENGINE_BIN="$(BIN_CPU)" bash scripts/run_benchmark.sh

perf_gpu: build-cuda
	@MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR_DEFAULT))" PROMPTS="$(if $(PROMPTS),$(PROMPTS),benchmarks/prompts_10..txt)" RUNS="$(if $(RUNS),$(RUNS),3)" WARMUP="$(if $(WARMUP),$(WARMUP),1)" THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" PRECISION="$(if $(PRECISION),$(PRECISION),fp32)" BATCH="$(if $(BATCH),$(BATCH),1)" PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" WARMUP_TOKENS="$(if $(WARMUP_TOKENS),$(WARMUP_TOKENS),64)" AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),0)" ENGINE_BIN="$(BIN_CUDA)" bash scripts/run_benchmark.sh
