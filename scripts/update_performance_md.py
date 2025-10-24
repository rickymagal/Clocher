#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate docs/PERFORMANCE.md from strict benchmark JSON logs.

Behavior:
- Accepts CPU and/or GPU JSONL files (each produced by scripts/true_tps_strict.sh).
- If only one side is provided, it still renders both sections using any existing
  counterpart JSON found under build/ (best effort).
- Aggregations and the comparative table use ONLY the last 3 runs for each device.
- "Best true TPS" is the best aggregate TPS across the currently included CPU/GPU sections.
- "Run Parameters & Conditions" come from CLI overrides first, then CPU summary, then GPU summary.
- Spatial metrics include MB/token, total bytes touched, coverage vs model.bin, and effective bandwidth.
- System & Model Info is discovered locally (Linux-friendly).

Inputs (per device JSONL):
- Per-run rows: include tokens_generated, wall_time_s, tps_true, latency_p50_ms, latency_p95_ms, rss_peak_mb, kv_hits, kv_misses.
- Final summary row: includes aggregated fields and configuration (threads, precision, prompts, model_dir, etc.).
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

Run = Dict[str, Any]
Agg = Dict[str, Any]

# --------------------------- IO helpers ---------------------------

def _now_utc_str() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _read_json_lines(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                # Ignore non-JSON lines from the harness
                pass
    return out

def _partition_runs_and_summary(objs: List[Dict[str, Any]]) -> Tuple[List[Run], Optional[Dict[str, Any]]]:
    runs: List[Run] = []
    summary: Optional[Dict[str, Any]] = None
    for o in objs:
        is_summary = any(k in o for k in ("threads", "precision", "model_dir", "prompts", "runs", "rss_peak_mb_max"))
        is_run = all(k in o for k in ("tokens_generated", "wall_time_s", "tps_true"))
        if is_summary:
            summary = o
        elif is_run:
            runs.append(o)
    return runs, summary

# ------------------------- math / formatting ------------------------

def _fmt_float(x: Optional[float], digits: int = 3, unit: str = "", na: str = "n/a") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return na
    s = f"{x:.{digits}f}"
    return f"{s} {unit}".strip()

def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None

def _sum_i(xs: List[int]) -> int:
    return int(sum(xs)) if xs else 0

def _sum_f(xs: List[float]) -> float:
    return float(sum(xs)) if xs else 0.0

def _pick_first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _bytes_to_gb(n: Optional[float]) -> Optional[float]:
    if n is None:
        return None
    return n / 1_000_000_000.0

def _bytes_to_mb(n: Optional[float]) -> Optional[float]:
    if n is None:
        return None
    return n / 1_000_000.0

# ---------------------- system discovery ---------------------------

def _discover_cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            m = re.search(r"model name\s*:\s*(.+)", f.read())
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or None

def _discover_mem_total_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            m = re.search(r"MemTotal:\s+(\d+)\s+kB", f.read())
            if m:
                kb = float(m.group(1))
                return kb * 1024.0 / 1_000_000_000.0
    except Exception:
        pass
    return None

def _discover_os() -> Optional[str]:
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            kv = dict(
                (line.split("=", 1)[0], line.split("=", 1)[1].strip().strip('"'))
                for line in f if "=" in line
            )
        return kv.get("PRETTY_NAME") or platform.platform()
    except Exception:
        return platform.platform()

def _discover_kernel() -> Optional[str]:
    try:
        u = platform.uname()
        return f"{u.release}-{u.machine}"
    except Exception:
        return platform.version()

def _discover_git() -> Optional[str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--ignore-submodules"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"{sha} {'DIRTY' if dirty != 0 else ''}".strip()
    except Exception:
        return None

def _find_model_bin(model_dir: Optional[str]) -> Optional[str]:
    if not model_dir:
        return None
    for name in ("model.ie.bin", "model.bin", "weights.bin"):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    return None

def _filesize(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except Exception:
        return None

# ---------------------- aggregation logic --------------------------

def _take_last3(runs: List[Run]) -> List[Run]:
    return runs[-3:] if len(runs) > 3 else runs

def _agg(runs: List[Run]) -> Agg:
    tokens = [int(r.get("tokens_generated", 0)) for r in runs]
    wall   = [float(r.get("wall_time_s", 0.0)) for r in runs]
    tps    = [float(r.get("tps_true", 0.0)) for r in runs]
    p50    = [float(r.get("latency_p50_ms", r.get("p50_ms", 0.0))) for r in runs if ("latency_p50_ms" in r or "p50_ms" in r)]
    p95    = [float(r.get("latency_p95_ms", r.get("p95_ms", 0.0))) for r in runs if ("latency_p95_ms" in r or "p95_ms" in r)]
    rss    = [float(r.get("rss_peak_mb", 0.0)) for r in runs if "rss_peak_mb" in r]
    kvh    = [int(r.get("kv_hits", 0)) for r in runs]
    kvm    = [int(r.get("kv_misses", 0)) for r in runs]

    tokens_sum = _sum_i(tokens)
    wall_sum   = _sum_f(wall)
    tps_true   = (float(tokens_sum) / wall_sum) if wall_sum > 0 else None

    return {
        "runs": len(runs),
        "tokens_sum": tokens_sum,
        "wall_sum": wall_sum,
        "tps_true": tps_true,
        "p50_mean": _mean(p50),
        "p95_mean": _mean(p95),
        "rss_mean": _mean(rss),
        "rss_max": max(rss) if rss else None,
        "kv_hits": _sum_i(kvh),
        "kv_misses": _sum_i(kvm),
    }

def _mb_per_token(bpt: Optional[int]) -> str:
    if bpt is None:
        return "n/a"
    return f"{_fmt_float(bpt / 1_000_000.0, 1)} MB/token"

def _bytes_touched(tokens_sum: int, bpt: Optional[int]) -> Optional[float]:
    if not bpt or tokens_sum <= 0:
        return None
    return float(tokens_sum) * float(bpt)

def _bandwidth_gbps(bytes_touched: Optional[float], wall_sum: Optional[float]) -> Optional[float]:
    if bytes_touched is None or wall_sum is None or wall_sum <= 0:
        return None
    return _bytes_to_gb(bytes_touched) / wall_sum

# ------------------------- rendering helpers -----------------------

def _render_device(title: str, agg: Agg, bpt: Optional[int], model_size: Optional[int]) -> str:
    bt = _bytes_touched(agg["tokens_sum"], bpt)
    bw = _bandwidth_gbps(bt, agg["wall_sum"])
    coverage = "n/a"
    if bpt and model_size and model_size > 0:
        coverage = f"{(float(bpt)/float(model_size)):.3f}"
    return f"""## {title} — Summary (latest benchmark)
- Runs: **{agg['runs']}**
- Tokens generated (Σ): **{agg['tokens_sum']}**
- Wall time (Σ): **{_fmt_float(agg['wall_sum'], 3, 's')}**
- True TPS (Σ tokens / Σ time): **{_fmt_float(agg['tps_true'], 3)}**

## Latency
- p50 (mean across runs): **{_fmt_float(agg['p50_mean'], 3, 'ms')}**
- p95 (mean across runs): **{_fmt_float(agg['p95_mean'], 3, 'ms')}**

## Spatial Complexity (Memory & Cache)
- RSS peak (mean): **{_fmt_float(agg['rss_mean'], 3, 'MB')}**
- RSS peak (max): **{_fmt_float(agg['rss_max'], 3, 'MB')}**
- KV cache: **{agg['kv_hits']} hits / {agg['kv_misses']} misses**
- IE_BYTES_PER_TOKEN: **{_mb_per_token(bpt)}**
- Bytes touched (Σ): **{_fmt_float(_bytes_to_gb(bt), 1, 'GB')}**
- Working-set coverage (bytes_per_token / model.bin): **{coverage}**
- Effective bandwidth: **{_fmt_float(bw, 2, 'GB/s')}**
"""

def _render_shared(shared: Dict[str, Any]) -> str:
    return f"""## Run Parameters & Conditions
- Engine bin: `{shared.get('engine_bin','n/a')}`
- Prompts file: `{shared.get('prompts','n/a')}`
- Threads: **{shared.get('threads','n/a')}**
- Precision: **{shared.get('precision','n/a')}**
- Batch: **{shared.get('batch','n/a')}**
- Prefetch: **{shared.get('prefetch','n/a')}**
- Pretranspose: **{shared.get('pretranspose','n/a')}**
- Affinity: **{shared.get('affinity','n/a')}**
- Max new tokens: **{shared.get('max_new','n/a')}**
- IE_REQUIRE_MODEL: **{shared.get('ie_require_model','n/a')}**
- IE_BYTES_PER_TOKEN: **{shared.get('ie_bytes_per_token','n/a')}**
- IE_STRIDE_BYTES: **{shared.get('ie_stride_bytes','n/a')}**
- IE_VERIFY_TOUCH: **{shared.get('ie_verify_touch','n/a')}**
"""

def _render_system(model_dir: Optional[str]) -> str:
    cpu = _discover_cpu_model() or "n/a"
    cores = os.cpu_count() or 0
    mem = _discover_mem_total_gb()
    os_name = _discover_os() or "n/a"
    kernel = _discover_kernel() or "n/a"
    git = _discover_git() or "n/a"
    return f"""## System & Model Info
- CPU: **{cpu}**
- Logical cores: **{cores if cores else 'n/a'}**
- RAM (MemTotal): **{_fmt_float(mem, 1, 'GB')}**
- OS: **{os_name}**
- Kernel: **{kernel}**
- Git commit: **{git}**
"""

def _render_table(cpu_runs: List[Run], gpu_runs: List[Run]) -> str:
    lines = []
    lines.append("## Comparative Runs\n")
    lines.append("| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | KV hits | KV misses |")
    lines.append("|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------:|----------:|")
    def row(dv: str, r: Run, idx: int) -> str:
        return (
            f"| {dv} | {idx} | "
            f"{int(r.get('tokens_generated', 0))} | "
            f"{_fmt_float(float(r.get('wall_time_s', 0.0)), 3)} | "
            f"{_fmt_float(float(r.get('tps_true', 0.0)), 3)} | "
            f"{_fmt_float(float(r.get('latency_p50_ms', r.get('p50_ms', 0.0))), 3)} | "
            f"{_fmt_float(float(r.get('latency_p95_ms', r.get('p95_ms', 0.0))), 3)} | "
            f"{_fmt_float(float(r.get('rss_peak_mb', 0.0)), 3)} | "
            f"{int(r.get('kv_hits', 0))} | {int(r.get('kv_misses', 0))} |"
        )
    for i, r in enumerate(cpu_runs, 1):
        lines.append(row("CPU", r, i))
    for i, r in enumerate(gpu_runs, 1):
        lines.append(row("GPU", r, i))
    return "\n".join(lines) + "\n"

def _merge_shared(cli: Dict[str, Any], cpu_sum: Optional[Dict[str, Any]], gpu_sum: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    def g(k: str) -> Any:
        if cli.get(k) is not None:
            return cli[k]
        if cpu_sum and cpu_sum.get(k) is not None:
            return cpu_sum.get(k)
        if gpu_sum and gpu_sum.get(k) is not None:
            return gpu_sum.get(k)
        return None

    keys = [
        "engine_bin","prompts","threads","precision","batch","prefetch","pretranspose",
        "affinity","max_new","ie_require_model","ie_bytes_per_token","ie_stride_bytes",
        "ie_verify_touch","model_dir"
    ]
    return {k: g(k) for k in keys}

# ------------------------------ CLI -------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update PERFORMANCE.md from strict JSON logs (last 3 runs per device).")
    p.add_argument("--cpu-json", default=None, help="Path to strict CPU JSONL")
    p.add_argument("--gpu-json", default=None, help="Path to strict GPU JSONL")
    p.add_argument("--out", default="docs/PERFORMANCE.md", help="Output Markdown file")

    # Shared/override params
    p.add_argument("--engine-bin", dest="engine_bin", default=None)
    p.add_argument("--cpu-engine-bin", dest="cpu_engine_bin", default=None)
    p.add_argument("--gpu-engine-bin", dest="gpu_engine_bin", default=None)
    p.add_argument("--prompts-file", dest="prompts", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--precision", default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--prefetch", default=None)
    p.add_argument("--pretranspose", default=None)
    p.add_argument("--affinity", default=None)
    p.add_argument("--max-new", dest="max_new", type=int, default=None)
    p.add_argument("--ie-require-model", dest="ie_require_model", type=int, default=None)
    p.add_argument("--ie-bytes-per-token", dest="ie_bytes_per_token", type=int, default=None)
    p.add_argument("--ie-stride-bytes", dest="ie_stride_bytes", type=int, default=None)
    p.add_argument("--ie-verify-touch", dest="ie_verify_touch", type=int, default=None)
    p.add_argument("--model-dir", dest="model_dir", default=None)
    return p.parse_args()

# ------------------------------ main ------------------------------

def main() -> int:
    args = _parse()

    # If args not provided, try defaults under build/
    cpu_json = args.cpu_json or (os.path.exists("build/strict_cpu.json") and "build/strict_cpu.json") or None
    gpu_json = args.gpu_json or (os.path.exists("build/strict_gpu.json") and "build/strict_gpu.json") or None

    # Load CPU
    cpu_runs: List[Run] = []
    cpu_sum: Optional[Dict[str, Any]] = None
    if cpu_json and os.path.isfile(cpu_json):
        c_objs = _read_json_lines(cpu_json)
        c_runs_all, cpu_sum = _partition_runs_and_summary(c_objs)
        cpu_runs = _take_last3(c_runs_all)

    # Load GPU
    gpu_runs: List[Run] = []
    gpu_sum: Optional[Dict[str, Any]] = None
    if gpu_json and os.path.isfile(gpu_json):
        g_objs = _read_json_lines(gpu_json)
        g_runs_all, gpu_sum = _partition_runs_and_summary(g_objs)
        gpu_runs = _take_last3(g_runs_all)

    # Engine bin preference (side-specific first, then generic)
    cpu_engine_bin = _pick_first(args.cpu_engine_bin, args.engine_bin)
    gpu_engine_bin = _pick_first(args.gpu_engine_bin, args.engine_bin)

    # Build merged shared params
    cli_shared = {
        "engine_bin": args.engine_bin,
        "prompts": args.prompts,
        "threads": args.threads,
        "precision": args.precision,
        "batch": args.batch,
        "prefetch": args.prefetch,
        "pretranspose": args.pretranspose,
        "affinity": args.affinity,
        "max_new": args.max_new,
        "ie_require_model": args.ie_require_model,
        "ie_bytes_per_token": args.ie_bytes_per_token,
        "ie_stride_bytes": args.ie_stride_bytes,
        "ie_verify_touch": args.ie_verify_touch,
        "model_dir": args.model_dir,
    }
    shared = _merge_shared(cli_shared, cpu_sum, gpu_sum)
    if cpu_engine_bin:
        shared["engine_bin"] = cpu_engine_bin
    elif gpu_engine_bin and not shared.get("engine_bin"):
        shared["engine_bin"] = gpu_engine_bin

    # Parse BPT as int if present
    bpt_val = shared.get("ie_bytes_per_token")
    try:
        bpt = int(bpt_val) if bpt_val is not None else None
    except Exception:
        bpt = None

    # Aggregations
    cpu_agg = _agg(cpu_runs) if cpu_runs else None
    gpu_agg = _agg(gpu_runs) if gpu_runs else None

    # Model size and coverage
    model_bin_path = _find_model_bin(shared.get("model_dir"))
    model_size = _filesize(model_bin_path)

    # Best TPS (current snapshot)
    best_label = "n/a"
    best_tps = None
    if cpu_agg and gpu_agg:
        if (cpu_agg["tps_true"] or 0) >= (gpu_agg["tps_true"] or 0):
            best_label, best_tps = "CPU", cpu_agg["tps_true"]
        else:
            best_label, best_tps = "GPU", gpu_agg["tps_true"]
    elif cpu_agg:
        best_label, best_tps = "CPU", cpu_agg["tps_true"]
    elif gpu_agg:
        best_label, best_tps = "GPU", gpu_agg["tps_true"]

    # Render
    out_lines: List[str] = []
    out_lines.append("# Performance Notes\n")
    out_lines.append(f"_Last updated: **{_now_utc_str()}**_\n")
    out_lines.append(f"\n**Best true TPS:** **{best_label} — {_fmt_float(best_tps, 3)}**.\n")

    if cpu_agg:
        out_lines.append(_render_device("CPU", cpu_agg, bpt, model_size))
    if gpu_agg:
        out_lines.append(_render_device("GPU", gpu_agg, bpt, model_size))

    out_lines.append(_render_shared(shared))
    out_lines.append(_render_system(shared.get("model_dir")))
    out_lines.append(_render_table(cpu_runs, gpu_runs))

    # Write file
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"[update_performance_md] Wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
