#!/usr/bin/env python3
"""
Update docs/PERFORMANCE.md from the newest benchmarks/reports/<STAMP>/ directory
(or --report-dir). Works even if the binary emitted zeros by using samples.csv
computed/fallback rows.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, statistics as st
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "PERFORMANCE.md"
REPORTS = ROOT / "benchmarks" / "reports"

def newest_report_dir():
    dirs = sorted((d for d in REPORTS.glob("*/") if d.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None

def load_params(dir_: Path) -> dict:
    p = {}
    f = dir_ / "params.json"
    if f.exists():
        p = json.loads(f.read_text())
    # last-resort defaults
    p.setdefault("threads","n/a")
    p.setdefault("precision","n/a")
    p.setdefault("pretranspose","n/a")
    p.setdefault("affinity","n/a")
    return p

def load_summary(dir_: Path) -> dict:
    s = {}
    f = dir_ / "summary.json"
    if f.exists():
        try:
            s = json.loads(f.read_text())
        except Exception:
            s = {}
    return s

def compute_from_csv(dir_: Path) -> dict:
    csvf = dir_ / "samples.csv"
    if not csvf.exists():
        return {}
    rows=[]
    with csvf.open(newline='') as f:
        r = csv.reader(f)
        next(r, None)  # header comment
        hdr = next(r, None)
        for row in r:
            if not row or row[0].startswith('[dbg]'):
                continue
            try:
                tg = float(row[0]); wt = float(row[1]); tps = float(row[2])
                p50 = float(row[3]); p95 = float(row[4])
            except Exception:
                continue
            if tg>0 and wt>0:
                rows.append((tg,wt,tps,p50,p95))
    if not rows:
        return {"tps_true": None, "latency_p50_ms": None, "latency_p95_ms": None}
    tps_vals = [r[2] for r in rows]
    p50_vals = [r[3] for r in rows if r[3]>0]
    p95_vals = [r[4] for r in rows if r[4]>0]
    return {
        "tps_true": st.median(tps_vals),
        "latency_p50_ms": (st.median(p50_vals) if p50_vals else None),
        "latency_p95_ms": (st.median(p95_vals) if p95_vals else None),
    }

def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=str, default=None)
    args = ap.parse_args()

    if args.report_dir:
        dir_ = Path(args.report_dir).resolve()
    else:
        dir_ = newest_report_dir()
    if not dir_ or not dir_.exists():
        print("no report dir found", file=sys.stderr)
        sys.exit(1)

    params = load_params(dir_)
    summary = load_summary(dir_)
    # If summary missing or zeros, recompute from CSV
    recompute = (
        not summary or
        summary.get("tps_true") in (0, None) or
        summary.get("latency_p50_ms") in (0, None) and summary.get("latency_p95_ms") in (0, None)
    )
    if recompute:
        summary = compute_from_csv(dir_)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    flame = (dir_/ "flamegraph.svg").exists()
    perf  = (dir_/ "perf.data").exists()

    md = f"""# Performance Notes

_Last updated: **{ts}**_

## Summary (latest run)
- Reports directory: `{dir_.as_posix()}`
- TPS (true): **{fmt(summary.get('tps_true'))}**
- Latency p50: **{fmt(summary.get('latency_p50_ms'))}**
- Latency p95: **{fmt(summary.get('latency_p95_ms'))}**

## Run parameters
- Threads: **{params.get('threads','n/a')}**
- Precision: **{params.get('precision','n/a')}**
- Pretranspose: **{params.get('pretranspose','n/a')}**
- Affinity policy (CLI): **{params.get('affinity','n/a')}**

## Profiling Artifacts
- `flamegraph.svg`: **{'present' if flame else 'absent'}**
- `perf.data`: **{'present' if perf else 'absent'}**

## Hot Paths (annotated)
- GEMV (`ie_gemv_f32`): AVX2 microkernel if available; otherwise generic path.
- Activation (`tanh` fast path): clamped polynomial/table approximation.
- Thread pool scheduling: contiguous shard with grainsize control and optional pinning.

## Next optimization actions
- Validate NUMA policy impacts using `scripts/set_numa.sh` (`interleave|node:X|strict`).
- Explore epilogue fusion (bias + activation) in GEMV output.
- Extend blocked-K packing and prefetch distances based on flamegraph evidence.
"""
    DOCS.write_text(md)
    print(f"[ok] wrote: {DOCS}")

if __name__ == "__main__":
    main()
