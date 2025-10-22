# File: Makefile
# Inference Engine — C core + Python stdlib harness
# =================================================
# Complete Makefile (English-only)

SHELL := /usr/bin/bash

# Export env so VAR=… make target reaches the scripts.
.EXPORT_ALL_VARIABLES:
export STACKCOLLAPSE FLAMEGRAPH PROMPTS_FILE BATCH PREFETCH WARMUP THREADS PRECISION PRETRANSPOSE ROUNDS MAX_NEW FREQ CALLGRAPH
export ENGINE_BIN MODEL MODEL_DIR PROMPTS RUNS WARMUP THREADS AFFINITY BATCH PREFETCH PRETRANSPOSE WARMUP_TOKENS REPORT_ROOT PUSHGATEWAY_URL MAX_NEW
export TARGET_SECONDS IE_BYTES_PER_TOKEN IE_STRIDE_BYTES IE_VERIFY_TOUCH
# NEW: device metadata (safe no-ops for current CPU-only tree)
export DEVICE GPU_ID

# -------- toolchain ----------------------------------------------------------
CC       ?= gcc
CFLAGS   ?= -std=c11 -O3 -Wall -Wextra -Werror -pedantic
LDFLAGS  ?= -lpthread -lm

# -------- paths / binaries ---------------------------------------------------
BUILD    ?= build
BIN      ?= $(BUILD)/inference-engine
INC      := -Iengine/include
MODEL_DIR_DEFAULT := models/gpt-oss-20b

# -------- core sources (DO NOT add main here) --------------------------------
SRC_CORE := \
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

# CLI entry point
SRC_MAIN := engine/src/main_infer.c

# =================================================
.PHONY: setup build build-release build-cuda build-l0 \
        test bench bench-direct profile perf-report baseline-report \
        fmt lint docs docs-doxygen clean microbench \
        monitoring-up monitoring-down metrics-exporter \
        ptq-calibrate ptq-from-hf ptq-from-torch ptq-from-bin \
        perf_cpu_fp32 perf_cpu_int8 perf_cpu_bf16 perf_gpu
# =================================================

setup:
	@echo "[setup] Optional developer tools."

# -------- build --------------------------------------------------------------
build: $(BIN)

$(BIN): $(SRC_MAIN) $(SRC_CORE)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LDFLAGS)

build-release: CFLAGS += -DNDEBUG
build-release: build

# NEW: placeholder GPU targets (do not change your current build)
build-cuda: build
	@echo "[info] build-cuda: current tree is CPU-only; integrate nvcc later."

build-l0: build
	@echo "[info] build-l0: current tree is CPU-only; integrate Level Zero later."

# -------- tests --------------------------------------------------------------
test: build
	@echo "[test] C unit tests"
	$(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS) && $(BUILD)/test_tensor

	# IMPORTANT: link ie_kv_instrumentation.c because ie_api.c calls ie_kv_on_token
	$(CC) $(CFLAGS) $(INC) tests/c/test_api.c engine/src/ie_api.c engine/src/ie_tensor.c engine/src/util_logging.c engine/src/util_metrics.c engine/src/io/weights.c engine/src/io/tokenizer.c engine/src/io/ie_batcher.c engine/src/opt/cpu_features.c engine/src/opt/thread_pool.c engine/src/opt/pretranspose.c engine/src/opt/numa_probe.c engine/src/kernels/gemv_generic.c engine/src/kernels/gemv_avx2.c engine/src/math/fast_tanh.c engine/src/math/floatx.c engine/src/ie_kv_instrumentation.c -o $(BUILD)/test_api $(LDFLAGS) && ( cd models/gpt-oss-20b && ../../$(BUILD)/test_api )

	# test_weights: só roda se IEBIN existir OU se IE_SKIP_WEIGHTS_TEST não estiver setada
	@if [ -z "$$IE_SKIP_WEIGHTS_TEST" ]; then \
	  if [ -f models/gpt-oss-20b/model.ie.json ] && [ -f models/gpt-oss-20b/model.ie.bin ]; then \
	    echo "[test] test_weights"; \
	    $(CC) $(CFLAGS) $(INC) tests/c/test_weights.c engine/src/io/weights.c -o $(BUILD)/test_weights $(LDFLAGS) && ( cd models/gpt-oss-20b && ../../$(BUILD)/test_weights ); \
	  else \
	    echo "[skip] test_weights (IEBIN not present; export IE_SKIP_WEIGHTS_TEST=1 to silence)"; \
	  fi; \
	else \
	  echo "[skip] test_weights (IE_SKIP_WEIGHTS_TEST=1)"; \
	fi

	$(CC) $(CFLAGS) $(INC) tests/c/test_tokenizer.c engine/src/io/tokenizer.c -o $(BUILD)/test_tokenizer $(LDFLAGS) && $(BUILD)/test_tokenizer
	$(CC) $(CFLAGS) $(INC) tests/c/test_kernels.c engine/src/kernels/gemv_generic.c engine/src/kernels/gemv_avx2.c -o $(BUILD)/test_kernels $(LDFLAGS) && $(BUILD)/test_kernels
	$(CC) $(CFLAGS) $(INC) tests/c/test_threadpool.c engine/src/opt/thread_pool.c -o $(BUILD)/test_threadpool $(LDFLAGS) && $(BUILD)/test_threadpool
	$(CC) $(CFLAGS) $(INC) tests/c/test_math.c engine/src/math/fast_tanh.c -o $(BUILD)/test_math $(LDFLAGS) && $(BUILD)/test_math
	$(CC) $(CFLAGS) $(INC) tests/c/test_cpu_features.c engine/src/opt/cpu_features.c -o $(BUILD)/test_cpu_features $(LDFLAGS) && $(BUILD)/test_cpu_features
	$(CC) $(CFLAGS) $(INC) tests/c/test_batcher.c engine/src/io/ie_batcher.c -o $(BUILD)/test_batcher $(LDFLAGS) && $(BUILD)/test_batcher

	@echo "[test] Python tests"
	python3 -m unittest discover -s tests/python -p 'test_*.py' -v

# -------- benchmark (shell harness) ------------------------------------------
# IMPORTANT: You MUST pass PROMPTS explicitly. No autodetection.
# Example:
#   make bench PROMPTS=benchmarks/prompts_10..txt
bench: build
	@# Require PROMPTS
	@if [ -z "$$PROMPTS" ]; then echo "ERROR: PROMPTS must be set (e.g., PROMPTS=benchmarks/prompts_10..txt)"; exit 2; fi
	@if [ ! -f "$$PROMPTS" ]; then echo "ERROR: PROMPTS '$$PROMPTS' not found"; exit 2; fi
	@if [ ! -x "$(BIN)" ]; then echo "ERROR: engine binary '$(BIN)' not found or not executable"; exit 2; fi
	@MDIR="$${MODEL:-$${MODEL_DIR:-$(MODEL_DIR_DEFAULT)}}"; \
	if [ ! -d "$$MDIR" ]; then echo "ERROR: model dir '$$MDIR' not found"; exit 2; fi; \
	ABS_BIN="$$(realpath -m $(BIN))"; \
	ABS_PROMPTS="$$(realpath -m "$$PROMPTS")"; \
	THREADS_VAL="$${THREADS:-$$(nproc)}"; \
	PRECISION_VAL="$${PRECISION:-fp32}"; \
	BATCH_VAL="$${BATCH:-1}"; \
	PREFETCH_VAL="$${PREFETCH:-auto}"; \
	PRETRANS_VAL="$${PRETRANSPOSE:-all}"; \
	AFFINITY_VAL="$${AFFINITY:-auto}"; \
	MAX_NEW_VAL="$${MAX_NEW:-128}"; \
	IE_REQ_VAL="$${IE_REQUIRE_MODEL:-1}"; \
	IE_BPT_VAL="$${IE_BYTES_PER_TOKEN:-67108864}"; \
	IE_STRIDE_VAL="$${IE_STRIDE_BYTES:-256}"; \
	IE_TOUCH_VAL="$${IE_VERIFY_TOUCH:-1}"; \
	echo "[bench] using PROMPTS=$$ABS_PROMPTS"; \
	echo "[bench] strict run (true TPS)…"; \
	ENGINE_BIN="$$ABS_BIN" MODEL_DIR="$$MDIR" PROMPTS="$$ABS_PROMPTS" \
	THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
	PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
	MAX_NEW="$$MAX_NEW_VAL" \
	IE_REQUIRE_MODEL="$$IE_REQ_VAL" IE_BYTES_PER_TOKEN="$$IE_BPT_VAL" \
	IE_STRIDE_BYTES="$$IE_STRIDE_VAL" IE_VERIFY_TOUCH="$$IE_TOUCH_VAL" \
	bash scripts/true_tps_strict.sh | tee $(BUILD)/strict_latest.json; \
	echo "[bench] updating docs/PERFORMANCE.md (strict)…"; \
	ENGINE_BIN="$$ABS_BIN" PROMPTS="$$ABS_PROMPTS" \
	THREADS="$$THREADS_VAL" PRECISION="$$PRECISION_VAL" BATCH="$$BATCH_VAL" \
	PREFETCH="$$PREFETCH_VAL" PRETRANSPOSE="$$PRETRANS_VAL" AFFINITY="$$AFFINITY_VAL" \
	MAX_NEW="$$MAX_NEW_VAL" \
	IE_REQUIRE_MODEL="$$IE_REQ_VAL" IE_BYTES_PER_TOKEN="$$IE_BPT_VAL" \
	IE_STRIDE_BYTES="$$IE_STRIDE_VAL" IE_VERIFY_TOUCH="$$IE_TOUCH_VAL" \
	python3 scripts/update_performance_md.py --strict-json $(BUILD)/strict_latest.json

# -------- benchmark without harness (quick check) ----------------------------
bench-direct: build
	@set -e; \
	PF="$(if $(BENCH_PROMPTS),$(BENCH_PROMPTS),benchmarks/prompts_10.txt)"; \
	BATCH="$(if $(BENCH_BATCH),$(BENCH_BATCH),32)"; \
	WARM="$(if $(BENCH_WARMUP),$(BENCH_WARMUP),4)"; \
	PREF="$(if $(BENCH_PREFETCH),$(BENCH_PREFETCH),on)"; \
	if [ -f $$PF ]; then \
	  ( cd $(MODEL_DIR_DEFAULT) && ../$(BIN) --prompts-file "$$(realpath -m "$$PF")" --batch $$BATCH --max-new 8 --prefetch $$PREF --warmup $$WARM ) | \
	    python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["tokens_generated"])' >/dev/null; \
	else \
	  ( cd $(MODEL_DIR_DEFAULT) && ../$(BIN) --prompt "bench-default" --max-new 8 --prefetch $$PREF --warmup $$WARM ) | \
	    python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["tokens_generated"])' >/dev/null; \
	fi; \
	echo "[bench-direct] OK"

# -------- profiling / flamegraph --------------------------------------------
profile: build
	@bash -lc 'cd $(MODEL_DIR_DEFAULT) && ../../scripts/profile_flamegraph.sh ../../$(BIN) "profiling prompt with 64+ tokens"'

# --- perf-report: strictly update PERFORMANCE.md; FlameGraph is optional ---
perf-report: build
	@echo "[profile] generating flamegraph (optional)…"
	@set -e; \
	MDROOT="$$(pwd)"; \
	MODELDIR="models/gpt-oss-20b"; \
	cd "$$MODELDIR"; \
	if command -v perf >/dev/null 2>&1 && \
	   { [ -x "$${STACKCOLLAPSE:-/usr/bin/stackcollapse-perf.pl}" ] || [ -f "$${STACKCOLLAPSE:-/usr/bin/stackcollapse-perf.pl}" ]; } && \
	   [ -f "$${FLAMEGRAPH:-/usr/bin/flamegraph.pl}" ]; then \
	  echo "[profile] generating flamegraph..."; \
	  perf record -F 1200 -g -- ../$(BIN) --prompt "profiling-64-tokens" --max-new 64 >/dev/null 2>&1 || true; \
	  perf script | "$${STACKCOLLAPSE:-/usr/bin/stackcollapse-perf.pl}" > out.perf || true; \
	  perl "$${FLAMEGRAPH:-/usr/bin/flamegraph.pl}" out.perf > "$$MDROOT/flamegraph.svg" || true; \
	  rm -f out.perf; \
	else \
	  echo "[profile] skipping flamegraph (tools not found)"; \
	fi; \
	cd "$$MDROOT"

	@echo "[bench] strict run (true TPS)…"
	@IE_BYTES_PER_TOKEN="$(IE_BYTES_PER_TOKEN)" IE_STRIDE_BYTES="$(IE_STRIDE_BYTES)" IE_VERIFY_TOUCH="$(IE_VERIFY_TOUCH)" \
	ENGINE_BIN="$(BIN)" MODEL_DIR="models/gpt-oss-20b" PROMPTS="benchmarks/prompts_10.txt" \
	THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" PRECISION="$(if $(PRECISION),$(PRECISION),fp32)" \
	AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" \
	BATCH="$(if $(BATCH),$(BATCH),1)" PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" \
	MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),128)" \
	bash scripts/true_tps_strict.sh | tee /tmp/ie_strict.json

	@echo "[bench] updating docs/PERFORMANCE.md (strict)…"
	@python3 scripts/update_performance_md.py --strict-json /tmp/ie_strict.json || true

baseline-report: build
	@echo "[bench] running harness..."
	@bash scripts/run_benchmark.sh
	@echo "[report] generating BASELINE.md..."
	@python3 scripts/make_baseline_md.py

# -------- dev helpers --------------------------------------------------------
fmt:
	@command -v clang-format >/dev/null 2>&1 && clang-format -i $$(find engine -name '*.c' -o -name '*.h') || echo "clang-format not installed"

lint:
	@echo "[lint] warnings-as-errors enabled by default"
	@command -v clang-tidy >/dev/null 2>&1 && clang-tidy $$(find engine/src -name '*.c') -- $(CFLAGS) $(INC) || echo "clang-tidy not installed"

docs:
	@echo "[docs] Final narrative emitted at project end (README/report)."

docs-doxygen:
	@command -v doxygen >/dev/null 2>&1 || { echo "doxygen not installed"; exit 1; }
	doxygen docs/Doxyfile
	@echo "HTML docs under docs/doxygen/html/index.html"

# -------- clean --------------------------------------------------------------
clean:
	rm -rf $(BUILD) benchmarks/reports/* perf.data* flamegraph.svg out.perf script.stacks callgrind.out.* callgrind.stacks

# -------- microbench ---------------------------------------------------------
microbench: build
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) benchmarks/src/microbench_gemv.c -o $(BUILD)/microbench_gemv $(LDFLAGS)
	@echo "[run] microbench (H=256 V=1024 iters=200)"
	@$(BUILD)/microbench_gemv 256 1024 200

# -------- monitoring / metrics ----------------------------------------------
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

# -------- PTQ (INT8) ---------------------------------------------------------
ptq-calibrate:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
	@test -n "$(WEIGHTS)"     || { echo "Set WEIGHTS=/path/to/weights.bin"; exit 2; }
	@test -n "$(ROWS)"        || { echo "Set ROWS=<rows>"; exit 2; }
	@test -n "$(COLS)"        || { echo "Set COLS=<cols>"; exit 2; }
	@test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
	python3 benchmarks/ptq_calib.py \
	  --weights "$(WEIGHTS)" --rows $(ROWS) --cols $(COLS) \
	  --mode $(if $(MODE),$(MODE),per_row) \
	  --out-prefix "$(OUT_PREFIX)" \
	  --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-hf:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
	@test -n "$(HF_MODEL)"    || { echo "Set HF_MODEL=org/repo"; exit 2; }
	@test -n "$(KEY)"         || { echo "Set KEY=<state_dict key>"; exit 2; }
	@test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
	python3 scripts/ptq_from_hf.py \
	  --repo "$(HF_MODEL)" \
	  --key "$(KEY)" \
	  $(if $(REV),--revision "$(REV)",) \
	  $(if $(FILE),--file "$(FILE)",) \
	  $(if $(TRANSPOSE),--transpose,) \
	  --out-prefix "$(OUT_PREFIX)" \
	  --mode $(if $(MODE),$(MODE),per_row) \
	  --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-torch:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
	@test -n "$(TORCH_CHECKPOINT)" || { echo "Set TORCH_CHECKPOINT=/path/to/model.ckpt"; exit 2; }
	@test -n "$(KEY)"              || { echo "Set KEY=<state_dict key>"; exit 2; }
	@test -n "$(OUT_PREFIX)"       || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
	python3 scripts/ptq_from_source.py \
	  --source torch \
	  --checkpoint "$(TORCH_CHECKPOINT)" \
	  --key "$(KEY)" \
	  $(if $(TRANSPOSE),--transpose,) \
	  --out-prefix "$(OUT_PREFIX)" \
	  --mode $(if $(MODE),$(MODE),per_row) \
	  --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

ptq-from-bin:
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
	@test -n "$(BIN)"         || { echo "Set BIN=/path/to/f32.bin"; exit 2; }
	@test -n "$(ROWS)"        || { echo "Set ROWS=<rows>"; exit 2; }
	@test -n "$(COLS)"        || { echo "Set COLS=<cols>"; exit 2; }
	@test -n "$(OUT_PREFIX)"  || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
	python3 scripts/ptq_from_source.py \
	  --source raw \
	  --bin "$(BIN)" --rows $(ROWS) --cols $(COLS) \
	  $(if $(TRANSPOSE),--transpose,) \
	  --out-prefix "$(OUT_PREFIX)" \
	  --mode $(if $(MODE),$(MODE),per_row) \
	  --accuracy-threshold $(if $(THRESH),$(THRESH),0.995)

# -------- performance targets (20B real) -------------------------------------
# Just run: make perf_cpu_fp32   (defaults to models/gpt-oss-20b)
perf_cpu_fp32: build
	@MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR_DEFAULT))" \
	PROMPTS="$(if $(PROMPTS),$(PROMPTS),benchmarks/prompts_10.txt)" \
	RUNS="$(if $(RUNS),$(RUNS),3)" \
	WARMUP="$(if $(WARMUP),$(WARMUP),1)" \
	THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" \
	PRECISION="fp32" \
	BATCH="$(if $(BATCH),$(BATCH),1)" \
	PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" \
	PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" \
	WARMUP_TOKENS="$(if $(WARMUP_TOKENS),$(WARMUP_TOKENS),64)" \
	AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" \
	MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),0)" \
	ENGINE_BIN="$(BIN)" \
	bash scripts/run_benchmark.sh

perf_cpu_bf16: build
	@MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR_DEFAULT))" \
	PROMPTS="$(if $(PROMPTS),$(PROMPTS),benchmarks/prompts_10.txt)" \
	RUNS="$(if $(RUNS),$(RUNS),3)" \
	WARMUP="$(if $(WARMUP),$(WARMUP),1)" \
	THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" \
	PRECISION="bf16" \
	BATCH="$(if $(BATCH),$(BATCH),1)" \
	PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" \
	PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" \
	WARMUP_TOKENS="$(if $(WARMUP_TOKENS),$(WARMUP_TOKENS),64)" \
	AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" \
	MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),0)" \
	ENGINE_BIN="$(BIN)" \
	bash scripts/run_benchmark.sh

perf_cpu_int8: build
	@MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR_DEFAULT))" \
	PROMPTS="$(if $(PROMPTS),$(PROMPTS),benchmarks/prompts_10.txt)" \
	RUNS="$(if $(RUNS),$(RUNS),3)" \
	WARMUP="$(if $(WARMUP),$(WARMUP),1)" \
	THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" \
	PRECISION="$(if $(PRECISION),$(PRECISION),int8)" \
	BATCH="$(if $(BATCH),$(BATCH),1)" \
	PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" \
	PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" \
	WARMUP_TOKENS="$(if $(WARMUP_TOKENS),$(WARMUP_TOKENS),64)" \
	AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" \
	MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),0)" \
	ENGINE_BIN="$(BIN)" \
	bash scripts/run_benchmark.sh

# GPU/iGPU variant: harness is identical, once kernels exist
perf_gpu: build
	@MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR_DEFAULT))" \
	PROMPTS="$(if $(PROMPTS),$(PROMPTS),benchmarks/prompts_10.txt)" \
	RUNS="$(if $(RUNS),$(RUNS),3)" \
	WARMUP="$(if $(WARMUP),$(WARMUP),1)" \
	THREADS="$(if $(THREADS),$(THREADS),$(shell nproc))" \
	PRECISION="$(if $(PRECISION),$(PRECISION),fp32)" \
	BATCH="$(if $(BATCH),$(BATCH),1)" \
	PREFETCH="$(if $(PREFETCH),$(PREFETCH),auto)" \
	PRETRANSPOSE="$(if $(PRETRANSPOSE),$(PRETRANSPOSE),all)" \
	WARMUP_TOKENS="$(if $(WARMUP_TOKENS),$(WARMUP_TOKENS),64)" \
	AFFINITY="$(if $(AFFINITY),$(AFFINITY),auto)" \
	MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),0)" \
	ENGINE_BIN="$(BIN)" \
	bash scripts/run_benchmark.sh
