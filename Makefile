# Inference Engine — C core + Python stdlib harness
CC       ?= gcc
CFLAGS   ?= -std=c11 -O3 -Wall -Wextra -Werror -pedantic
LDFLAGS  ?= -lpthread
BUILD    ?= build
BIN      ?= $(BUILD)/inference-engine
SRC      := engine/src/main_infer.c engine/src/ie_api.c engine/src/ie_tensor.c engine/src/util_logging.c engine/src/util_metrics.c
INC      := -Iengine/include

.PHONY: setup build build-release test bench profile fmt lint docs clean

setup:
	@echo "[setup] dev helpers are optional; runtime stays dependency-free."

build: $(BIN)
$(BIN): $(SRC)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LDFLAGS)

build-release: CFLAGS += -DNDEBUG
build-release: build

test: build
	@echo "[test] C unit tests"
	$(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS) && $(BUILD)/test_tensor
	$(CC) $(CFLAGS) $(INC) tests/c/test_api.c engine/src/ie_api.c engine/src/ie_tensor.c engine/src/util_logging.c engine/src/util_metrics.c -o $(BUILD)/test_api $(LDFLAGS) && $(BUILD)/test_api
	@echo "[test] Python tests"
	python3 -m unittest discover -s tests/python -p 'test_*.py' -v

bench: build
	@bash scripts/run_benchmark.sh

profile: build
	@bash scripts/profile_flamegraph.sh $(BIN)

fmt:
	@command -v clang-format >/dev/null 2>&1 && clang-format -i $$(find engine -name '*.c' -o -name '*.h') || echo "clang-format not installed"

lint:
	@echo "[lint] warnings-as-errors enabled by default"
	@command -v clang-tidy >/dev/null 2>&1 && clang-tidy $$(find engine/src -name '*.c') -- $(CFLAGS) $(INC) || echo "clang-tidy not installed"

docs:
	@echo "[docs] Final narrative emitted at project end (README/report)."

clean:
	rm -rf $(BUILD) benchmarks/reports/* perf.data* flamegraph.svg
