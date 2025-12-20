#!/usr/bin/env bash
set -euo pipefail
python3 tools/hf_to_iebin.py
python3 tools/dedup_prepare_and_extract_all.py
# Move outputs up one level so the runtime picks them up automatically.
cp -f models/gpt-oss-20b/dedup_out/model.defaults.bin   models/gpt-oss-20b/model.defaults.bin
cp -f models/gpt-oss-20b/dedup_out/model.masks.bin      models/gpt-oss-20b/model.masks.bin
cp -f models/gpt-oss-20b/dedup_out/model.exceptions.bin models/gpt-oss-20b/model.exceptions.bin
echo "[dedup] artifacts ready under models/gpt-oss-20b/"
