# scripts/run_gptoss.sh
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-models/gpt-oss-20b}"
PROMPT="${2:-Hello.}"
MAX_NEW="${3:-32}"

export IE_VERIFY_TOUCH="${IE_VERIFY_TOUCH:-1}"
export IE_BYTES_PER_TOKEN="${IE_BYTES_PER_TOKEN:-67108864}"
export IE_STRIDE_BYTES="${IE_STRIDE_BYTES:-256}"

./build/inference-engine \
  --model-dir "$MODEL_DIR" \
  --prompt "$PROMPT" \
  --max-new "$MAX_NEW"

echo "exit=$?"
