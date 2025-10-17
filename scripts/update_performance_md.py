#!/usr/bin/env python3
"""
update_performance_md.py
Reads the latest harness report under benchmarks/reports/ and rewrites docs/PERFORMANCE.md
with a compact summary. Also notes whether flamegraph.svg and perf.data are present.
"""

import json
import os
import time
from glob import glob

REPORTS_DIR = os.path.join("benchmarks", "reports")
OUT_MD = os.path.join("docs", "PERFORMANCE.md")

def _latest_report_dir() -> str | None:
    candidates = [d for d in glob(os.path.join(REPORTS_DIR, "*")) if os.path.isdir(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _fmt(v, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}{suffix}"
    return f"{v}{suffix}"

def main():
    rep = _latest_report_dir()
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    if rep is None:
        content = f"""# Performance Notes

_Last updated: **{ts}**_

No reports found under `{REPORTS_DIR}`.
"""
        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_MD, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[ok] wrote: {OUT_MD} (no reports)")
        return

    summary = _load_json(os.path.join(rep, "summary.json"))
    params  = _load_json(os.path.join(rep, "params.json"))

    # Summary fields (be tolerant to missing keys)
    tps_true = summary.get("tps_true")
    lat_p50  = summary.get("latency_p50_ms")
    lat_p95  = summary.get("latency_p95_ms")

    # Params (tolerant)
    threads      = params.get("threads")
    precision    = params.get("precision", "fp32")
    pretranspose = params.get("pretranspose")
    affinity_cli = params.get("affinity_cli")
    affinity_env = params.get("affinity_env_toggle")
    cpu_features = params.get("cpu_features")  # e.g. "avx2, fma, sse4.2"

    # Artifacts presence
    flamegraph_present = os.path.exists("flamegraph.svg")
    perf_present       = os.path.exists("perf.data")

    content = f"""# Performance Notes

_Last updated: **{ts}**_

## Summary (latest run)
- Reports directory: `{rep}`
- TPS (true): **{_fmt(tps_true)}**
- Latency p50: **{_fmt(lat_p50, ' ms')}**
- Latency p95: **{_fmt(lat_p95, ' ms')}**

## Run parameters
- Threads: **{_fmt(threads)}**
- Precision: **{precision}**
- Pretranspose: **{_fmt(pretranspose)}**
- Affinity policy (CLI): **{_fmt(affinity_cli)}**
- Affinity env toggle (`IE_TP_USE_AFFINITY`): **{_fmt(affinity_env)}**
- Detected CPU features: **{_fmt(cpu_features)}**

## Profiling Artifacts
- `flamegraph.svg`: **{"present" if flamegraph_present else "absent"}**
- `perf.data`: **{"present" if perf_present else "absent"}**

## Hot Paths (annotated)
- GEMV (`ie_gemv_f32`): AVX2 microkernel if available; otherwise generic path.
- Activation (`tanh` fast path): clamped polynomial/table approximation.
- Thread pool scheduling: contiguous shard with grainsize control and optional pinning.

## Next optimization actions
- Validate NUMA policy impacts using `scripts/set_numa.sh` (`interleave|node:X|strict`).
- Explore epilogue fusion (bias + activation) in GEMV output.
- Extend blocked-K packing and prefetch distances based on flamegraph evidence.
"""

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[ok] wrote: {OUT_MD}")

if __name__ == "__main__":
    main()
