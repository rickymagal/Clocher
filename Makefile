# Inference Engine — C core + Python stdlib harness
# =================================================
# Complete Makefile (English-only)

SHELL := /usr/bin/bash

# Export env so VAR=… make target reaches the scripts.
.EXPORT_ALL_VARIABLES:
export STACKCOLLAPSE FLAMEGRAPH PROMPTS_FILE BATCH PREFETCH WARMUP THREADS PRECISION PRETRANSPOSE ROUNDS MAX_NEW FREQ CALLGRAPH
export ENGINE_BIN MODEL MODEL_DIR PROMPTS RUNS WARMUP THREADS AFFINITY BATCH PREFETCH PRETRANSPOSE WARMUP_TOKENS REPORT_ROOT PUSHGATEWAY_URL MAX_NEW
export TARGET_SECONDS IE_BYTES_PER_TOKEN IE_STRIDE_BYTES IE_VERIFY_TOUCH

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
  engine/src/math/floatx.c

# CLI entry point
SRC_MAIN := engine/src/main_infer.c

# =================================================
.PHONY: setup build build-release test bench bench-direct profile perf-report baseline-report \
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

# -------- tests --------------------------------------------------------------
test: build
	@echo "[test] C unit tests"
	$(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS) && $(BUILD)/test_tensor
	$(CC) $(CFLAGS) $(INC) tests/c/test_api.c $(SRC_CORE) -o $(BUILD)/test_api $(LDFLAGS) && $(BUILD)/test_api
	$(CC) $(CFLAGS) $(INC) tests/c/test_weights.c engine/src/io/weights.c -o $(BUILD)/test_weights $(LDFLAGS) && $(BUILD)/test_weights
	$(CC) $(CFLAGS) $(INC) tests/c/test_tokenizer.c engine/src/io/tokenizer.c -o $(BUILD)/test_tokenizer $(LDFLAGS) && $(BUILD)/test_tokenizer
	$(CC) $(CFLAGS) $(INC) tests/c/test_kernels.c engine/src/kernels/gemv_generic.c engine/src/kernels/gemv_avx2.c -o $(BUILD)/test_kernels $(LDFLAGS) && $(BUILD)/test_kernels
	$(CC) $(CFLAGS) $(INC) tests/c/test_threadpool.c engine/src/opt/thread_pool.c -o $(BUILD)/test_threadpool $(LDFLAGS) && $(BUILD)/test_threadpool
	$(CC) $(CFLAGS) $(INC) tests/c/test_math.c engine/src/math/fast_tanh.c -o $(BUILD)/test_math $(LDFLAGS) && $(BUILD)/test_math
	$(CC) $(CFLAGS) $(INC) tests/c/test_cpu_features.c engine/src/opt/cpu_features.c -o $(BUILD)/test_cpu_features $(LDFLAGS) && $(BUILD)/test_cpu_features
	$(CC) $(CFLAGS) $(INC) tests/c/test_batcher.c engine/src/io/ie_batcher.c -o $(BUILD)/test_batcher $(LDFLAGS) && $(BUILD)/test_batcher
	@echo "[test] Python tests"
	python3 -m unittest discover -s tests/python -p 'test_*.py' -v

# -------- benchmark (shell harness) ------------------------------------------
# Default model directory in this repo: models/gpt-oss-20b
bench: build
	@MODEL_DIR="$(if $(MODEL_DIR),$(MODEL_DIR),$(MODEL_DIR_DEFAULT))" \
	  MODEL="$(if $(MODEL),$(MODEL),$(MODEL_DIR))" \
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
	  MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),128)" \
	  TARGET_SECONDS="$(if $(TARGET_SECONDS),$(TARGET_SECONDS),10)" \
	  ENGINE_BIN="$(BIN)" \
	  IE_BYTES_PER_TOKEN="$(if $(IE_BYTES_PER_TOKEN),$(IE_BYTES_PER_TOKEN),0)" \
	  IE_STRIDE_BYTES="$(if $(IE_STRIDE_BYTES),$(IE_STRIDE_BYTES),64)" \
	  IE_VERIFY_TOUCH="$(if $(IE_VERIFY_TOUCH),$(IE_VERIFY_TOUCH),0)" \
	  bash -lc '\
	    mkdir -p $(BUILD) benchmarks/reports; \
	    echo "[bench] strict run (true TPS)…"; \
	    out=$$(ENGINE_BIN="$$ENGINE_BIN" MODEL_DIR="$$MODEL_DIR" PROMPTS="$$PROMPTS" THREADS="$$THREADS" PRECISION="$$PRECISION" AFFINITY="$$AFFINITY" PRETRANSPOSE="$$PRETRANSPOSE" BATCH="$$BATCH" PREFETCH="$$PREFETCH" MAX_NEW="$$MAX_NEW" TARGET_SECONDS="$$TARGET_SECONDS" IE_BYTES_PER_TOKEN="$$IE_BYTES_PER_TOKEN" IE_STRIDE_BYTES="$$IE_STRIDE_BYTES" IE_VERIFY_TOUCH="$$IE_VERIFY_TOUCH" scripts/true_tps_strict.sh); \
	    echo "$$out" | tee $(BUILD)/strict_latest.json; \
	    echo "[bench] updating docs/PERFORMANCE.md (strict)…"; \
	    python3 scripts/update_performance_md.py --strict-json $(BUILD)/strict_latest.json || { echo "[warn] strict run failed: strict run did not yield JSON."; echo "$$out"; }; \
	  '

# -------- benchmark without harness (quick check) ----------------------------
bench-direct: build
	@set -e; \
	PF="$(if $(BENCH_PROMPTS),$(BENCH_PROMPTS),benchmarks/prompts_10.txt)"; \
	BATCH="$(if $(BENCH_BATCH),$(BENCH_BATCH),32)"; \
	WARM="$(if $(BENCH_WARMUP),$(BENCH_WARMUP),4)"; \
	PREF="$(if $(BENCH_PREFETCH),$(BENCH_PREFETCH),on)"; \
	if [ -f $$PF ]; then \
	  ( cd $(MODEL_DIR_DEFAULT) && ../$(BIN) --prompts-file "../$$PF" --batch $$BATCH --max-new 8 --prefetch $$PREF --warmup $$WARM ) | \
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
	# Só gera flamegraph se perf + stackcollapse + flamegraph.pl existirem
	if command -v perf >/dev/null 2>&1 && \
	   { [ -x "$${STACKCOLLAPSE:-/usr/bin/stackcollapse-perf.pl}" ] || [ -f "$${STACKCOLLAPSE:-/usr/bin/stackcollapse-perf.pl}" ]; } && \
	   [ -f "$${FLAMEGRAPH:-/usr/bin/flamegraph.pl}" ]; then \
	  echo "[profile] generating flamegraph..."; \
	  PERF_ROUNDS="$${FREQ:-120}"; \
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
	MAX_NEW="$(if $(MAX_NEW),$(MAX_NEW),128)" TARGET_SECONDS="$(if $(TARGET_SECONDS),$(TARGET_SECONDS),10)" \
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
	PRECISION="int8" \
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
