# Inference Engine — C core + Python stdlib harness
# -------------------------------------------------
CC       ?= gcc
CFLAGS   ?= -std=c11 -O3 -Wall -Wextra -Werror -pedantic
LDFLAGS  ?= -lpthread -lm
BUILD    ?= build
BIN      ?= $(BUILD)/inference-engine
INC      := -Iengine/include

# Core sources (NO main here)
SRC_CORE := \
  engine/src/ie_api.c \
  engine/src/ie_tensor.c \
  engine/src/util_logging.c \
  engine/src/util_metrics.c \
  engine/src/io/weights.c \
  engine/src/io/tokenizer.c \
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

.PHONY: setup build build-release test bench profile fmt lint docs docs-doxygen clean \
        microbench perf-report baseline-report

setup:
	@echo "[setup] dev helpers are optional; runtime stays dependency-free."

# --- build binaries ---
build: $(BIN)

$(BIN): $(SRC_MAIN) $(SRC_CORE)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LDFLAGS)

build-release: CFLAGS += -DNDEBUG
build-release: build

# --- unit tests (C + Python) ---
test: build
	@echo "[test] C unit tests"
	$(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS) && $(BUILD)/test_tensor
	# DO NOT link main_infer.c here
	$(CC) $(CFLAGS) $(INC) tests/c/test_api.c $(SRC_CORE) -o $(BUILD)/test_api $(LDFLAGS) && $(BUILD)/test_api
	$(CC) $(CFLAGS) $(INC) tests/c/test_weights.c engine/src/io/weights.c -o $(BUILD)/test_weights $(LDFLAGS) && $(BUILD)/test_weights
	$(CC) $(CFLAGS) $(INC) tests/c/test_tokenizer.c engine/src/io/tokenizer.c -o $(BUILD)/test_tokenizer $(LDFLAGS) && $(BUILD)/test_tokenizer
	$(CC) $(CFLAGS) $(INC) tests/c/test_kernels.c engine/src/kernels/gemv_generic.c engine/src/kernels/gemv_avx2.c -o $(BUILD)/test_kernels $(LDFLAGS) && $(BUILD)/test_kernels
	$(CC) $(CFLAGS) $(INC) tests/c/test_threadpool.c engine/src/opt/thread_pool.c -o $(BUILD)/test_threadpool $(LDFLAGS) && $(BUILD)/test_threadpool
	# extras step 4+
	$(CC) $(CFLAGS) $(INC) tests/c/test_math.c engine/src/math/fast_tanh.c -o $(BUILD)/test_math $(LDFLAGS) && $(BUILD)/test_math
	$(CC) $(CFLAGS) $(INC) tests/c/test_cpu_features.c engine/src/opt/cpu_features.c -o $(BUILD)/test_cpu_features $(LDFLAGS) && $(BUILD)/test_cpu_features
	@echo "[test] Python tests"
	python3 -m unittest discover -s tests/python -p 'test_*.py' -v

# --- benchmark harness ---
bench: build
	@bash scripts/run_benchmark.sh

# --- perf/Flamegraph ---
profile: build
	@scripts/profile_flamegraph.sh $(BIN) "profiling prompt with 64+ tokens"

# --- format & lint ---
fmt:
	@command -v clang-format >/dev/null 2>&1 && clang-format -i $$(find engine -name '*.c' -o -name '*.h') || echo "clang-format not installed"

lint:
	@echo "[lint] warnings-as-errors enabled by default"
	@command -v clang-tidy >/dev/null 2>&1 && clang-tidy $$(find engine/src -name '*.c') -- $(CFLAGS) $(INC) || echo "clang-tidy not installed"

# --- docs ---
docs:
	@echo "[docs] Final narrative emitted at project end (README/report)."

docs-doxygen:
	@command -v doxygen >/dev/null 2>&1 || { echo "doxygen not installed"; exit 1; }
	doxygen docs/Doxyfile
	@echo "HTML docs under docs/doxygen/html/index.html"

# --- clean ---
clean:
	rm -rf $(BUILD) benchmarks/reports/* perf.data* flamegraph.svg out.perf script.stacks callgrind.out.* callgrind.stacks
