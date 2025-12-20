#!/usr/bin/env bash
set -euo pipefail
MD="models/gpt-oss-20b"
rm -rf "$MD/dedup_out" "$MD/dedup_tmp"
rm -f  "$MD/model.defaults.bin" "$MD/model.masks.bin" "$MD/model.exceptions.bin"
rm -f  "$MD/model.dedup.groups.for_dedup.json" "$MD/model.dedup.groups.indices.json"
rm -f  "$MD/tensor_map.for_dedup.json"
echo "[clean] done."
