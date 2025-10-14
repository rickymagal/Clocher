#!/usr/bin/env python3
"""
Update docs/PERFORMANCE.md from the latest report + optional flamegraph stacks.

- Reads newest benchmarks/reports/<ts>/{summary.json,samples.csv}
- If available, parses 'script.stacks' (perf) or 'callgrind.stacks' to list top hot paths
- Writes/overwrites docs/PERFORMANCE.md with a concise, standardized section (ENGLISH ONLY)
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmarks" / "reports"
DOC = ROOT / "docs" / "PERFORMANCE.md"
FLAMEGRAPH = ROOT / "flamegraph.svg"

def newest_report_dir() -> Path:
    """Return the newest benchmark report directory under benchmarks/reports."""
    dirs = [p for p in REPORTS.glob("*") if p.is_dir()]
    if not dirs:
        raise SystemExit("No report dirs found. Run `make bench` first.")
    return sorted(dirs)[-1]

def load_summary(d: Path) -> dict:
    """Load summary.json from a report directory."""
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))

def load_samples(d: Path):
    """Load samples.csv rows from a report directory."""
    with (d / "samples.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def mean(xs):
    """Mean over a sequence of numbers (coerced to float)."""
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else 0.0

def parse_stacks() -> list[tuple[str, int]]:
    """
    Parse folded stacks from perf or callgrind:
      - script.stacks (perf)
      - callgrind.stacks (valgrind fallback)
    Return top (leaf_function, samples) pairs sorted descending.
    """
    for name in ("script.stacks", "callgrind.stacks"):
        p = ROOT / name
        if p.exists():
            counts: dict[str, int] = {}
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if " " not in line:
                    continue
                stack, cnt = line.rsplit(" ", 1)
                try:
                    c = int(cnt)
                except ValueError:
                    continue
                leaf = stack.split(";")[-1]
                counts[leaf] = counts.get(leaf, 0) + c
            return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return []

def main():
    d = newest_report_dir()
    summary = load_summary(d)
    samples = load_samples(d)

    p50s = [float(r.get("latency_p50_ms", 0.0)) for r in samples]
    p95s = [float(r.get("latency_p95_ms", 0.0)) for r in samples]
    mean_p50 = mean(p50s)
    mean_p95 = mean(p95s)

    hot = parse_stacks()
    total = sum(c for _, c in hot) or 1
    top5 = [(f, (c / total) * 100.0) for f, c in hot[:5]]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# PERFORMANCE")
    lines.append("")
    lines.append("This document records policies and reproduction guidance for CPU performance.")
    lines.append("All content is generated in English by this script.")
    lines.append("")
    lines.append("## Current Profile (baseline FP32)")
    lines.append(f"- Experiment timestamp: **{now}**")
    lines.append("- Prompt/runner: `benchmarks/prompts.jsonl` via `benchmarks/harness.py`")
    lines.append("")
    lines.append("### Metrics (latest report)")
    lines.append(f"- **Avg True TPS (harness)**: **{summary.get('avg_tps_true', 0.0):.2f}** tok/s")
    lines.append(f"- **p50**: **{mean_p50:.3f}** ms | **p95**: **{mean_p95:.3f}** ms")
    lines.append(f"- **Total tokens**: **{int(summary.get('total_tokens', 0))}** | **samples**: **{int(summary.get('samples', 0))}**")
    lines.append("")
    lines.append("### Hot Paths (flamegraph)")
    if top5:
        for (fn, pct) in top5:
            lines.append(f"- `{fn}` — **{pct:.1f}%**")
    else:
        lines.append("- N/A (generate `flamegraph.svg` and `script.stacks` with `scripts/profile_flamegraph.sh`)")
    if FLAMEGRAPH.exists():
        rel = os.path.relpath(FLAMEGRAPH, start=DOC.parent)
        lines.append(f"\n> Flamegraph: `{rel}`")
    lines.append("")
    lines.append("## Next Optimizations")
    lines.append("- **GEMV**: AVX2/AVX-512 micro-kernels, blocking and FMA epilogues.")
    lines.append("- **Threading/NUMA**: contiguous row sharding + CPU pinning (compact/scatter).")
    lines.append("- **tanh**: polynomial/LUT approximation for faster nonlinearities.")
    lines.append("- **Embedding**: avoid `sinf` or precompute token-dependent patterns.")
    lines.append("")
    lines.append("## Reproduction & Policies")
    lines.append("- **Threads**: `--threads N` (default 1 for stable CI).")
    lines.append("- **CPU Affinity (Linux)**: enable per run with `IE_TP_USE_AFFINITY=1` and choose `--affinity {auto,compact,scatter}`.")
    lines.append("- **NUMA (Linux)**: use `scripts/set_numa.sh {compact|interleave|node:X} -- <CMD>` (external helper; engine remains runtime-only).")
    lines.append("")
    lines.append("## Evolution Table")
    lines.append("| Build/tag | Precision | Threads | Avg True TPS | p50 (ms) | p95 (ms) | Notes |")
    lines.append("|-----------|:---------:|:-------:|-------------:|---------:|---------:|:------|")
    lines.append(f"| baseline  | fp32      | auto    | {summary.get('avg_tps_true', 0.0):.2f} | {mean_p50:.3f} | {mean_p95:.3f} | Initial baseline |")

    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {DOC}")

if __name__ == "__main__":
    main()
