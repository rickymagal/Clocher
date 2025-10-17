# Inference Engine — C core + Python stdlib harness
# =================================================
# Full Makefile with all targets used so far.

# -------- toolchain --------
CC       ?= gcc
CFLAGS   ?= -std=c11 -O3 -Wall -Wextra -Werror -pedantic
LDFLAGS  ?= -lpthread -lm

# -------- paths / binaries --------
BUILD    ?= build
BIN      ?= $(BUILD)/inference-engine
INC      := -Iengine/include

# -------- core sources (NO main here) --------
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
.PHONY: setup build build-release test bench profile fmt lint docs docs-doxygen clean \
        microbench perf-report baseline-report \
        monitoring-up monitoring-down metrics-exporter \
        ptq-calibrate ptq-from-hf ptq-from-torch ptq-from-bin \
        bench-direct
# =================================================

setup:
	@echo "[setup] Optional dev tools; runtime remains dependency-free."

# -------- build binaries --------
build: $(BIN)

$(BIN): $(SRC_MAIN) $(SRC_CORE)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LDFLAGS)

build-release: CFLAGS += -DNDEBUG
build-release: build

# -------- unit tests (C + Python) --------
test: build
	@echo "[test] C unit tests"
	$(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS) && $(BUILD)/test_tensor
	# DO NOT link main_infer.c here
	$(CC) $(CFLAGS) $(INC) tests/c/test_api.c $(SRC_CORE) -o $(BUILD)/test_api $(LDFLAGS) && $(BUILD)/test_api
	$(CC) $(CFLAGS) $(INC) tests/c/test_weights.c engine/src/io/weights.c -o $(BUILD)/test_weights $(LDFLAGS) && $(BUILD)/test_weights
	$(CC) $(CFLAGS) $(INC) tests/c/test_tokenizer.c engine/src/io/tokenizer.c -o $(BUILD)/test_tokenizer $(LDFLAGS) && $(BUILD)/test_tokenizer
	$(CC) $(CFLAGS) $(INC) tests/c/test_kernels.c engine/src/kernels/gemv_generic.c engine/src/kernels/gemv_avx2.c -o $(BUILD)/test_kernels $(LDFLAGS) && $(BUILD)/test_kernels
	$(CC) $(CFLAGS) $(INC) tests/c/test_threadpool.c engine/src/opt/thread_pool.c -o $(BUILD)/test_threadpool $(LDFLAGS) && $(BUILD)/test_threadpool
	# extras
	$(CC) $(CFLAGS) $(INC) tests/c/test_math.c engine/src/math/fast_tanh.c -o $(BUILD)/test_math $(LDFLAGS) && $(BUILD)/test_math
	$(CC) $(CFLAGS) $(INC) tests/c/test_cpu_features.c engine/src/opt/cpu_features.c -o $(BUILD)/test_cpu_features $(LDFLAGS) && $(BUILD)/test_cpu_features
	$(CC) $(CFLAGS) $(INC) tests/c/test_batcher.c engine/src/io/ie_batcher.c -o $(BUILD)/test_batcher $(LDFLAGS) && $(BUILD)/test_batcher
	@echo "[test] Python tests"
	python3 -m unittest discover -s tests/python -p 'test_*.py' -v

# -------- benchmark harness --------
bench: build
	@BENCH_PROMPTS="$(if $(BENCH_PROMPTS),$(BENCH_PROMPTS),benchmarks/prompts.txt)" \
	  BENCH_BATCH="$(if $(BENCH_BATCH),$(BENCH_BATCH),32)" \
	  BENCH_WARMUP="$(if $(BENCH_WARMUP),$(BENCH_WARMUP),4)" \
	  BENCH_PREFETCH="$(if $(BENCH_PREFETCH),$(BENCH_PREFETCH),on)" \
	  bash scripts/run_benchmark.sh

# Quick direct bench without the shell harness (optional)
bench-direct: build
	@set -e; \
	PF="$(if $(BENCH_PROMPTS),$(BENCH_PROMPTS),benchmarks/prompts.txt)"; \
	BATCH="$(if $(BENCH_BATCH),$(BENCH_BATCH),32)"; \
	WARM="$(if $(BENCH_WARMUP),$(BENCH_WARMUP),4)"; \
	PREF="$(if $(BENCH_PREFETCH),$(BENCH_PREFETCH),on)"; \
	if [ -f $$PF ]; then \
	  $(BIN) --prompts-file $$PF --batch $$BATCH --max-new 8 --prefetch $$PREF --warmup $$WARM | \
	    python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["tokens_generated"])' >/dev/null; \
	else \
	  $(BIN) --prompt "bench-default" --max-new 8 --prefetch $$PREF --warmup $$WARM | \
	    python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["tokens_generated"])' >/dev/null; \
	fi; \
	echo "[bench-direct] OK"

# -------- perf/Flamegraph --------
profile: build
	@scripts/profile_flamegraph.sh $(BIN) "profiling prompt with 64+ tokens"

perf-report: build
	@echo "[profile] generating flamegraph..."
	@scripts/profile_flamegraph.sh $(BIN) "profiling prompt with 64+ tokens"
	@echo "[report] updating docs/PERFORMANCE.md..."
	@python3 scripts/update_performance_md.py

baseline-report: build
	@echo "[bench] running harness..."
	@bash scripts/run_benchmark.sh
	@echo "[report] generating BASELINE.md..."
	@python3 scripts/make_baseline_md.py

# -------- dev helpers --------
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

# -------- clean artifacts --------
clean:
	rm -rf $(BUILD) benchmarks/reports/* perf.data* flamegraph.svg out.perf script.stacks callgrind.out.* callgrind.stacks

# =================================================
# Step 3 helpers: microbench + perf-report + baseline-report
# =================================================
microbench: build
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) benchmarks/src/microbench_gemv.c -o $(BUILD)/microbench_gemv $(LDFLAGS)
	@echo "[run] microbench (H=256 V=1024 iters=200)"
	@$(BUILD)/microbench_gemv 256 1024 200

# =================================================
# Monitoring / metrics (optional)
# =================================================
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

# =================================================
# PTQ pipeline (INT8) — calibration/export helpers
# =================================================
# Usage examples:
#  make ptq-calibrate WEIGHTS=bin/Wxh.bin ROWS=512 COLS=512 MODE=per_row OUT_PREFIX=out/Wxh_int8
#  make ptq-from-hf HF_MODEL=facebook/opt-125m KEY=model.decoder.layers.0.self_attn.q_proj.weight OUT_PREFIX=out/qproj_int8
#  make ptq-from-torch TORCH_CHECKPOINT=/abs/path/model.ckpt KEY=rnn.weight_hh_l0 OUT_PREFIX=out/Wxh_int8
#  make ptq-from-bin BIN=in.bin ROWS=768 COLS=768 OUT_PREFIX=out/W_int8 MODE=per_row

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
