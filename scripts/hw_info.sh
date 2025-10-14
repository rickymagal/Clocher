#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# @file hw_info.sh
# @brief Emit compact hardware & OS info to stdout (Markdown friendly).
# -----------------------------------------------------------------------------
set -euo pipefail

echo "## Hardware & OS"
echo ""
echo "- Hostname: $(hostname)"
echo "- Kernel: $(uname -sr)"
if command -v lsb_release >/dev/null 2>&1; then
  echo "- Distro: $(lsb_release -ds)"
fi
echo "- CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
echo "- Cores (logical/physical guess): $(nproc) / $(lscpu | awk -F: '/^Core\(s\) per socket/ {c=$2} /^Socket\(s\)/ {s=$2} END{gsub(/ /, "", c); gsub(/ /, "", s); print c*s}')"
echo "- Flags: $(lscpu | awk -F: '/^Flags/ {print $2}' | sed 's/^ //')"
echo "- Memory (MB): $(free -m | awk '/^Mem:/ {print $2}')"
