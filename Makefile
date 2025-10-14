# Inference Engine — C core + Python stdlib harness
# -------------------------------------------------
CC       ?= gcc
CFLAGS   ?= -std=c11 -O3 -Wall -Wextra -Werror -pedantic
LDFLAGS  ?= -lpthread -lm
BUILD    ?= build
BIN      ?= $(BUILD)/inference-engine
INC      := -Iengine/include

SRC := \
  engine/src/main_infer.c \
  engine/src/ie_api.c \
  engine/src/ie_tensor.c \
  engine/src/util_logging.c \
  engine/src/util_metrics.c \
  engine/src/io/weights.c \
  engine/src/io/tokenizer.c

.PHONY: setup build build-release test bench profile fmt lint docs docs-doxygen clean \
        microbench perf-report baseline-report

# --- setup (optional dev helpers) ---
setup:
	@echo "[setup] dev helpers are optional; runtime stays dependency-free."

# --- build binaries ---
build: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LDFLAGS)

build-release: CFLAGS += -DNDEBUG
build-release: build

# --- unit tests (C + Python) ---
test: build
	@echo "[test] C unit tests"
	$(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS) && $(BUILD)/test_tensor
	$(CC) $(CFLAGS) $(INC) tests/c/test_api.c engine/src/ie_api.c engine/src/ie_tensor.c engine/src/util_logging.c engine/src/util_metrics.c engine/src/io/weights.c engine/src/io/tokenizer.c -o $(BUILD)/test_api $(LDFLAGS) && $(BUILD)/test_api
	$(CC) $(CFLAGS) $(INC) tests/c/test_weights.c engine/src/io/weights.c -o $(BUILD)/test_weights $(LDFLAGS) && $(BUILD)/test_weights
	$(CC) $(CFLAGS) $(INC) tests/c/test_tokenizer.c engine/src/io/tokenizer.c -o $(BUILD)/test_tokenizer $(LDFLAGS) && $(BUILD)/test_tokenizer
	@echo "[test] Python tests"
	python3 -m unittest discover -s tests/python -p 'test_*.py' -v

# --- benchmark harness ---
bench: build
	@bash scripts/run_benchmark.sh

# --- perf/Flamegraph (uses scripts/profile_flamegraph.sh) ---
profile: build
	@scripts/profile_flamegraph.sh $(BIN) "profiling prompt with 64+ tokens"

# --- format & lint (best-effort; optional tools) ---
fmt:
	@command -v clang-format >/dev/null 2>&1 && clang-format -i $$(find engine -name '*.c' -o -name '*.h') || echo "clang-format not installed"

lint:
	@echo "[lint] warnings-as-errors enabled by default"
	@command -v clang-tidy >/dev/null 2>&1 && clang-tidy $$(find engine/src -name '*.c') -- $(CFLAGS) $(INC) || echo "clang-tidy not installed"

# --- docs placeholders ---
docs:
	@echo "[docs] Final narrative emitted at project end (README/report)."

docs-doxygen:
	@command -v doxygen >/dev/null 2>&1 || { echo "doxygen not installed"; exit 1; }
	doxygen docs/Doxyfile
	@echo "HTML docs under docs/doxygen/html/index.html"

# --- clean artifacts ---
clean:
	rm -rf $(BUILD) benchmarks/reports/* perf.data* flamegraph.svg out.perf script.stacks callgrind.out.* callgrind.stacks

# ============================================================
# Step 3 helpers: microbench + perf-report + baseline-report
# ============================================================

# Standalone microbenchmark for GEMV/tanh/embed (builds and runs)
microbench: build
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) benchmarks/src/microbench_gemv.c -o $(BUILD)/microbench_gemv $(LDFLAGS)
	@echo "[run] microbench (H=256 V=1024 iters=200)"
	@$(BUILD)/microbench_gemv 256 1024 200

# Full profiling pipeline: flamegraph + PERFORMANCE.md
perf-report: build
	@echo "[profile] generating flamegraph..."
	@scripts/profile_flamegraph.sh $(BIN) "profiling prompt with 64+ tokens"
	@echo "[report] updating PERFORMANCE.md..."
	@python3 scripts/update_performance_md.py

# Full baseline report: bench + BASELINE.md
baseline-report: build
	@echo "[bench] running harness..."
	@bash scripts/run_benchmark.sh
	@echo "[report] generating BASELINE.md..."
	@python3 scripts/make_baseline_md.py > benchmarks/reports/BASELINE.md
