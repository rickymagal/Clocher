#!/usr/bin/env python3
"""
Create a Markdown baseline report from the latest benchmarks report directory.

- Finds the newest folder under benchmarks/reports/<timestamp>/
- Reads summary.json and samples.csv
- Prints BASELINE.md to stdout (so you can redirect to the file)
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "benchmarks" / "reports"

def newest_report_dir() -> Path:
    candidates = [p for p in REPORTS_DIR.glob("*") if p.is_dir()]
    if not candidates:
        raise SystemExit("No report directories found. Run `make bench` first.")
    return sorted(candidates)[-1]

def read_summary(d: Path) -> dict:
    s = d / "summary.json"
    return json.loads(s.read_text(encoding="utf-8"))

def read_samples(d: Path) -> list[dict]:
    rows = []
    with (d / "samples.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def main() -> int:
    d = newest_report_dir()
    summary = read_summary(d)
    samples = read_samples(d)

    # Derive p50/p95 from samples if available per-token latency; otherwise use CLI metrics columns
    # Our samples.csv contains engine metrics per run line, not per-token. We'll show means.
    p50s = [float(r.get("latency_p50_ms", 0.0)) for r in samples]
    p95s = [float(r.get("latency_p95_ms", 0.0)) for r in samples]
    mean_p50 = sum(p50s) / len(p50s) if p50s else 0.0
    mean_p95 = sum(p95s) / len(p95s) if p95s else 0.0

    md = []
    md.append(f"# Baseline Report — {d.name}")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- **Avg True TPS (harness)**: {summary.get('avg_tps_true', 0.0):.2f} tok/s")
    md.append(f"- **Total tokens**: {int(summary.get('total_tokens', 0))}")
    md.append(f"- **Runs (samples)**: {int(summary.get('samples', 0))}")
    md.append(f"- **Mean p50 latency**: {mean_p50:.3f} ms")
    md.append(f"- **Mean p95 latency**: {mean_p95:.3f} ms")
    md.append("")
    md.append("## Engine/Build flags")
    md.append("")
    md.append("- Precision: `fp32` (baseline)")
    md.append("- Threads: `auto` (baseline stub)")
    md.append("- Affinity: `auto` (baseline stub)")
    md.append("")
    md.append("## Observations")
    md.append("")
    md.append("- Replace dummy wait with real FP32 math ✅")
    md.append("- Latency is measured per token; p50/p95 reported by CLI; harness computes True TPS.")
    md.append("- Next steps: profile flamegraph, identify hotspots (gemv_rowmajor, embed_token, tanhf), plan AVX2 kernels, threading.")
    md.append("")
    md.append("## Raw files")
    md.append("")
    md.append(f"- `samples.csv` — {d / 'samples.csv'}")
    md.append(f"- `summary.json` — {d / 'summary.json' }")
    print("\n".join(md))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
