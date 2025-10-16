# Inference Engine — C core + Python stdlib harness
# -------------------------------------------------
# Use '>' (not TAB) to start recipe lines; copy-paste safe.
.RECIPEPREFIX := >

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
  engine/src/math/floatx.c \
  engine/src/quant/int8_ptq.c

# CLI entry point
SRC_MAIN := engine/src/main_infer.c

.PHONY: setup build build-release test bench profile fmt lint docs docs-doxygen clean \
        microbench perf-report baseline-report \
        weights-list-torch weights-export-torch \
        weights-list-onnx  weights-export-onnx \
        ptq-calibrate ptq-calibrate-auto \
        ptq-from-torch ptq-from-onnx ptq-from-hf \
        deps-ptq monitoring-up monitoring-down metrics-exporter

setup:
> echo "[setup] dev helpers are optional; runtime stays dependency-free."

# --- build binaries ---
build: $(BIN)

$(BIN): $(SRC_MAIN) $(SRC_CORE)
> mkdir -p $(BUILD)
> $(CC) $(CFLAGS) $(INC) $^ -o $@ $(LDFLAGS)

build-release: CFLAGS += -DNDEBUG
build-release: build

# --- unit tests (C + Python) ---
test: build
> echo "[test] C unit tests"
> $(CC) $(CFLAGS) $(INC) tests/c/test_tensor.c engine/src/ie_tensor.c engine/src/util_logging.c -o $(BUILD)/test_tensor $(LDFLAGS) && $(BUILD)/test_tensor
> # DO NOT link main_infer.c here
> $(CC) $(CFLAGS) $(INC) tests/c/test_api.c $(SRC_CORE) -o $(BUILD)/test_api $(LDFLAGS) && $(BUILD)/test_api
> $(CC) $(CFLAGS) $(INC) tests/c/test_weights.c engine/src/io/weights.c -o $(BUILD)/test_weights $(LDFLAGS) && $(BUILD)/test_weights
> $(CC) $(CFLAGS) $(INC) tests/c/test_tokenizer.c engine/src/io/tokenizer.c -o $(BUILD)/test_tokenizer $(LDFLAGS) && $(BUILD)/test_tokenizer
> $(CC) $(CFLAGS) $(INC) tests/c/test_kernels.c engine/src/kernels/gemv_generic.c engine/src/kernels/gemv_avx2.c -o $(BUILD)/test_kernels $(LDFLAGS) && $(BUILD)/test_kernels
> $(CC) $(CFLAGS) $(INC) tests/c/test_threadpool.c engine/src/opt/thread_pool.c -o $(BUILD)/test_threadpool $(LDFLAGS) && $(BUILD)/test_threadpool
> echo "[test] Python tests"
> python3 -m unittest discover -s tests/python -p 'test_*.py' -v

# --- benchmark harness ---
bench: build
> bash scripts/run_benchmark.sh

# --- perf/Flamegraph ---
profile: build
> scripts/profile_flamegraph.sh $(BIN) "profiling prompt with 64+ tokens"

# --- format & lint (best-effort; optional tools) ---
fmt:
> command -v clang-format >/dev/null 2>&1 && clang-format -i $$(find engine -name '*.c' -o -name '*.h') || echo "clang-format not installed"

lint:
> echo "[lint] warnings-as-errors enabled by default"
> command -v clang-tidy >/dev/null 2>&1 && clang-tidy $$(find engine/src -name '*.c') -- $(CFLAGS) $(INC) || echo "clang-tidy not installed"

# --- docs ---
docs:
> echo "[docs] Final narrative emitted at project end (README/report)."

docs-doxygen:
> command -v doxygen >/dev/null 2>&1 || { echo "doxygen not installed"; exit 1; }
> doxygen docs/Doxyfile
> echo "HTML docs under docs/doxygen/html/index.html"

# --- clean artifacts ---
clean:
> rm -rf $(BUILD) benchmarks/reports/* perf.data* flamegraph.svg out.perf script.stacks callgrind.out.* callgrind.stacks out

# ============================================================
# Step 3 helpers: microbench + perf-report + baseline-report
# ============================================================

microbench: build
> mkdir -p $(BUILD)
> $(CC) $(CFLAGS) $(INC) benchmarks/src/microbench_gemv.c -o $(BUILD)/microbench_gemv $(LDFLAGS)
> echo "[run] microbench (H=256 V=1024 iters=200)"
> $(BUILD)/microbench_gemv 256 1024 200

perf-report: build
> echo "[profile] generating flamegraph..."
> scripts/profile_flamegraph.sh $(BIN) "profiling prompt with 64+ tokens"
> echo "[report] updating PERFORMANCE.md..."
> python3 scripts/update_performance_md.py

baseline-report: build
> echo "[bench] running harness..."
> bash scripts/run_benchmark.sh
> echo "[report] generating BASELINE.md..."
> python3 scripts/make_baseline_md.py

# ============================================================
# Real weights export (manual) + classic PTQ
# ============================================================

# PyTorch (manual export)
weights-list-torch:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(TORCH_CHECKPOINT)" || { echo "Set TORCH_CHECKPOINT=/path/to/model.ckpt"; exit 2; }
> python3 scripts/export_tensors_torch.py --checkpoint "$(TORCH_CHECKPOINT)" --list

weights-export-torch:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(TORCH_CHECKPOINT)" || { echo "Set TORCH_CHECKPOINT=/path/to/model.ckpt"; exit 2; }
> test -n "$(TORCH_TENSORS)"    || { echo "Set TORCH_TENSORS='state.k:out.bin,...'"; exit 2; }
> { \
>   ARGS="--checkpoint \"$(TORCH_CHECKPOINT)\""; \
>   IFS=, ; for t in $(TORCH_TENSORS); do ARGS="$$ARGS --tensor \"$$t\""; done ; \
>   if [ -n "$(TORCH_ALIASES)" ]; then for a in $(TORCH_ALIASES); do ARGS="$$ARGS --alias \"$$a\""; done ; fi ; \
>   if [ -n "$(TORCH_TRANSPOSE)" ]; then for z in $(TORCH_TRANSPOSE); do ARGS="$$ARGS --transpose \"$$z\""; done ; fi ; \
>   eval python3 scripts/export_tensors_torch.py $$ARGS ; \
> }

# ONNX (manual export)
weights-list-onnx:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(ONNX_MODEL)" || { echo "Set ONNX_MODEL=/path/to/model.onnx"; exit 2; }
> python3 scripts/export_tensors_onnx.py --onnx "$(ONNX_MODEL)" --list

weights-export-onnx:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(ONNX_MODEL)"  || { echo "Set ONNX_MODEL=/path/to/model.onnx"; exit 2; }
> test -n "$(ONNX_TENSORS)"|| { echo "Set ONNX_TENSORS='init.name:out.bin,...'"; exit 2; }
> { \
>   ARGS="--onnx \"$(ONNX_MODEL)\""; \
>   IFS=, ; for t in $(ONNX_TENSORS); do ARGS="$$ARGS --tensor \"$$t\""; done ; \
>   if [ -n "$(ONNX_ALIASES)" ]; then for a in $(ONNX_ALIASES); do ARGS="$$ARGS --alias \"$$a\""; done ; fi ; \
>   if [ -n "$(ONNX_TRANSPOSE)" ]; then for z in $(ONNX_TRANSPOSE); do ARGS="$$ARGS --transpose \"$$z\""; done ; fi ; \
>   eval python3 scripts/export_tensors_onnx.py $$ARGS ; \
> }

# Classic PTQ (explicit rows/cols)
ptq-calibrate:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(WEIGHTS)"    || { echo "Set WEIGHTS=bin/Wxh.bin"; exit 2; }
> test -n "$(ROWS)"       || { echo "Set ROWS=<rows>"; exit 2; }
> test -n "$(COLS)"       || { echo "Set COLS=<cols>"; exit 2; }
> test -n "$(OUT_PREFIX)" || { echo "Set OUT_PREFIX=out/Wxh_int8"; exit 2; }
> python3 benchmarks/ptq_calib.py \
>   --weights "$(WEIGHTS)" --rows $(ROWS) --cols $(COLS) \
>   --mode $(if $(MODE),$(MODE),per_row) \
>   --out-prefix "$(OUT_PREFIX)" \
>   --accuracy-threshold $(if $(ACC),$(ACC),0.995)

# Auto-size fallback (kept)
ptq-calibrate-auto:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(WEIGHTS)"    || { echo "Set WEIGHTS=bin/Wxh.bin"; exit 2; }
> test -n "$(OUT_PREFIX)" || { echo "Set OUT_PREFIX=out/Wxh_int8"; exit 2; }
> python3 scripts/infer_shape_and_ptq.py \
>   --weights "$(WEIGHTS)" \
>   --mode $(if $(MODE),$(MODE),per_row) \
>   --out-prefix "$(OUT_PREFIX)" \
>   --accuracy-threshold $(if $(ACC),$(ACC),0.995)

# ============================================================
# One-shot PTQ directly from source model (reads TRUE shape)
# ============================================================

# PyTorch local checkpoint
ptq-from-torch:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(TORCH_CHECKPOINT)" || { echo "Set TORCH_CHECKPOINT=/path/to/model.ckpt"; exit 2; }
> test -n "$(KEY)"              || { echo "Set KEY=<state_dict key>"; exit 2; }
> test -n "$(OUT_PREFIX)"       || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
> python3 scripts/ptq_from_source.py \
>   --source torch \
>   --checkpoint "$(TORCH_CHECKPOINT)" \
>   --key "$(KEY)" \
>   $$( [ "$(TRANSPOSE)" = "1" ] && echo "--transpose" ) \
>   --out-prefix "$(OUT_PREFIX)" \
>   --mode $(if $(MODE),$(MODE),per_row) \
>   --accuracy-threshold $(if $(ACC),$(ACC),0.995)

# ONNX local model
ptq-from-onnx:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(ONNX_MODEL)" || { echo "Set ONNX_MODEL=/path/to/model.onnx"; exit 2; }
> test -n "$(INIT)"       || { echo "Set INIT=<initializer name>"; exit 2; }
> test -n "$(OUT_PREFIX)" || { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
> python3 scripts/ptq_from_source.py \
>   --source onnx \
>   --onnx "$(ONNX_MODEL)" \
>   --init "$(INIT)" \
>   $$( [ "$(TRANSPOSE)" = "1" ] && echo "--transpose" ) \
>   --out-prefix "$(OUT_PREFIX)" \
>   --mode $(if $(MODE),$(MODE),per_row) \
>   --accuracy-threshold $(if $(ACC),$(ACC),0.995)

# Hugging Face repo (downloads model, finds tensor, exports, PTQ)
# Required:
#   HF_MODEL=repo_id (e.g., "facebook/opt-125m")
#   KEY=state_dict key (e.g., "decoder.layers.0.self_attn.q_proj.weight")
# Optional:
#   REV=branch_or_commit
#   FILE=checkpoint filename override (if repo has multiple ckpt files)
#   TRANSPOSE=1 to transpose 2D tensor
ptq-from-hf:
> command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
> test -n "$(HF_MODEL)"  || { echo "Set HF_MODEL=org/repo"; exit 2; }
> test -n "$(KEY)"       || { echo "Set KEY=<state_dict key>"; exit 2; }
> test -n "$(OUT_PREFIX)"|| { echo "Set OUT_PREFIX=out/W_int8"; exit 2; }
> python3 scripts/ptq_from_hf.py \
>   --repo "$(HF_MODEL)" \
>   --key "$(KEY)" \
>   $$( [ -n "$(REV)" ] && echo "--revision \"$(REV)\"" ) \
>   $$( [ -n "$(FILE)" ] && echo "--file \"$(FILE)\"" ) \
>   $$( [ "$(TRANSPOSE)" = "1" ] && echo "--transpose" ) \
>   --out-prefix "$(OUT_PREFIX)" \
>   --mode $(if $(MODE),$(MODE),per_row) \
>   --accuracy-threshold $(if $(ACC),$(ACC),0.995)

# ============================================================
# Monitoring (Prometheus + Grafana) and Metrics Exporter
# ============================================================
monitoring-up:
> echo "[monitoring] bring up stack via docker compose"
> docker compose -f monitoring/docker-compose.yml up -d

monitoring-down:
> echo "[monitoring] tear down stack"
> docker compose -f monitoring/docker-compose.yml down

metrics-exporter:
> python3 scripts/metrics_exporter.py --port 8000

# ============================================================
# Python deps for PTQ pipelines
# ============================================================
deps-ptq:
> python3 -m pip install --upgrade pip
> python3 -m pip install --no-cache-dir numpy torch onnx safetensors huggingface_hub
