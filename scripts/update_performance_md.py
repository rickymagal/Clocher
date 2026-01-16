#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate docs/PERFORMANCE.md from strict benchmark JSON logs.

Key points:
- Uses only the last 3 runs for each device (for aggregate sections).
- Aggregations use Σ tokens / Σ time.
- Includes Memory Details when present in per-run logs.
- Includes a Deduplication section that reports *real* artifact sizes
  by following symlinks (reports target sizes, not link lengths).
- Adds a Per-Prompt CPU vs GPU section when per-prompt metrics are available
  in the latest run object (supports several common field names).
- Prints per-prompt token-ID verification status when present:
  - token_id_check_ok (or variants)
  - first mismatch step + differing IDs (best-effort extraction)
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
PromptRec = Dict[str, Any]


def _now_utc_str() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _read_json_lines(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                pass
    return out


def _partition_runs_and_summary(objs: List[Dict[str, Any]]) -> Tuple[List[Run], Optional[Dict[str, Any]]]:
    runs: List[Run] = []
    summary: Optional[Dict[str, Any]] = None
    for o in objs:
        is_summary = any(k in o for k in ("threads", "precision", "model_dir", "prompts", "runs", "rounds", "device"))
        is_run = all(k in o for k in ("tokens_generated", "wall_time_s", "tps_true"))
        if is_summary:
            summary = o
        elif is_run:
            runs.append(o)
    return runs, summary


def _fmt_float(x: Optional[float], digits: int = 3, unit: str = "", na: str = "n/a") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return na
    s = f"{x:.{digits}f}"
    return f"{s} {unit}".strip()


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _max_or_none(xs: List[float]) -> Optional[float]:
    return max(xs) if xs else None


def _min_or_none(xs: List[float]) -> Optional[float]:
    return min(xs) if xs else None


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
                for line in f
                if "=" in line
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
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--ignore-submodules"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
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


def _filesize_follow(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        rp = os.path.realpath(path)
        return os.path.getsize(rp)
    except Exception:
        return None


def _take_last3(runs: List[Run]) -> List[Run]:
    return runs[-3:] if len(runs) > 3 else runs


def _agg_numeric(runs: List[Run], key: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [float(r.get(key)) for r in runs if r.get(key) is not None]
    if not vals:
        return None, None, None
    return _mean(vals), _min_or_none(vals), _max_or_none(vals)


def _agg_sum(runs: List[Run], key: str) -> Optional[float]:
    vals = [float(r.get(key)) for r in runs if r.get(key) is not None]
    return _sum_f(vals) if vals else None


def _agg(runs: List[Run]) -> Agg:
    tokens = [int(r.get("tokens_generated", 0)) for r in runs]
    wall = [float(r.get("wall_time_s", 0.0)) for r in runs]
    p50 = [
        float(r.get("latency_p50_ms", r.get("p50_ms", 0.0)))
        for r in runs
        if ("latency_p50_ms" in r or "p50_ms" in r)
    ]
    p95 = [
        float(r.get("latency_p95_ms", r.get("p95_ms", 0.0)))
        for r in runs
        if ("latency_p95_ms" in r or "p95_ms" in r)
    ]

    tokens_sum = _sum_i(tokens)
    wall_sum = _sum_f(wall)
    tps_true = (float(tokens_sum) / wall_sum) if wall_sum > 0 else None

    rss_mean, _, rss_max = _agg_numeric(runs, "rss_peak_mb")
    pss_mean, _, pss_max = _agg_numeric(runs, "pss_peak_mb")
    vms_mean, _, vms_max = _agg_numeric(runs, "vms_peak_mb")

    rss_floor_mean, _, rss_floor_max = _agg_numeric(runs, "rss_floor_mb")
    rss_delta_mean, _, rss_delta_max = _agg_numeric(runs, "rss_delta_mb")

    minflt_sum = _agg_sum(runs, "minflt")
    majflt_sum = _agg_sum(runs, "majflt")

    swap_in_sum_mb = _agg_sum(runs, "swap_in_mb")
    swap_out_sum_mb = _agg_sum(runs, "swap_out_mb")

    psi_some_mean, _, psi_some_max = _agg_numeric(runs, "psi_mem_some_pct")
    psi_full_mean, _, psi_full_max = _agg_numeric(runs, "psi_mem_full_pct")

    numa_loc_mean, _, numa_loc_max = _agg_numeric(runs, "numa_locality_pct")

    mem_avail_mb_mean, _, _ = _agg_numeric(runs, "mem_available_mb")
    mem_avail_pct_mean, _, _ = _agg_numeric(runs, "mem_available_pct")

    kvh = [int(r.get("kv_hits", 0)) for r in runs]
    kvm = [int(r.get("kv_misses", 0)) for r in runs]

    return {
        "runs": len(runs),
        "tokens_sum": tokens_sum,
        "wall_sum": wall_sum,
        "tps_true": tps_true,
        "p50_mean": _mean(p50),
        "p95_mean": _mean(p95),
        "rss_mean": rss_mean,
        "rss_max": rss_max,
        "pss_mean": pss_mean,
        "pss_max": pss_max,
        "vms_mean": vms_mean,
        "vms_max": vms_max,
        "rss_floor_mean": rss_floor_mean,
        "rss_floor_max": rss_floor_max,
        "rss_delta_mean": rss_delta_mean,
        "rss_delta_max": rss_delta_max,
        "minflt_sum": minflt_sum,
        "majflt_sum": majflt_sum,
        "swap_in_sum_mb": swap_in_sum_mb,
        "swap_out_sum_mb": swap_out_sum_mb,
        "psi_some_mean": psi_some_mean,
        "psi_some_max": psi_some_max,
        "psi_full_mean": psi_full_mean,
        "psi_full_max": psi_full_max,
        "numa_locality_mean": numa_loc_mean,
        "numa_locality_max": numa_loc_max,
        "mem_available_mb_mean": mem_avail_mb_mean,
        "mem_available_pct_mean": mem_avail_pct_mean,
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


def _render_memory_details(agg: Agg) -> str:
    lines: List[str] = []
    lines.append("### Memory Details")

    if agg.get("pss_mean") is not None or agg.get("pss_max") is not None:
        lines.append(
            f"- PSS peak (mean / max): **{_fmt_float(agg.get('pss_mean'), 3, 'MB')} / {_fmt_float(agg.get('pss_max'), 3, 'MB')}**"
        )
    if agg.get("vms_mean") is not None or agg.get("vms_max") is not None:
        lines.append(
            f"- VMS peak (mean / max): **{_fmt_float(agg.get('vms_mean'), 3, 'MB')} / {_fmt_float(agg.get('vms_max'), 3, 'MB')}**"
        )
    if agg.get("rss_floor_mean") is not None or agg.get("rss_floor_max") is not None:
        lines.append(
            f"- RSS floor (mean / max): **{_fmt_float(agg.get('rss_floor_mean'), 3, 'MB')} / {_fmt_float(agg.get('rss_floor_max'), 3, 'MB')}**"
        )
    if agg.get("rss_delta_mean") is not None or agg.get("rss_delta_max") is not None:
        lines.append(
            f"- RSS delta vs baseline (mean / max): **{_fmt_float(agg.get('rss_delta_mean'), 3, 'MB')} / {_fmt_float(agg.get('rss_delta_max'), 3, 'MB')}**"
        )

    if agg.get("minflt_sum") is not None or agg.get("majflt_sum") is not None:
        lines.append(
            f"- Page faults (minor Σ / major Σ): **{_fmt_float(agg.get('minflt_sum'), 0)} / {_fmt_float(agg.get('majflt_sum'), 0)}**"
        )

    if agg.get("swap_in_sum_mb") is not None or agg.get("swap_out_sum_mb") is not None:
        lines.append(
            f"- Swap I/O (in Σ / out Σ): **{_fmt_float(agg.get('swap_in_sum_mb'), 1, 'MB')} / {_fmt_float(agg.get('swap_out_sum_mb'), 1, 'MB')}**"
        )

    if agg.get("psi_some_mean") is not None or agg.get("psi_some_max") is not None:
        lines.append(
            f"- PSI memory 'some' (mean / max): **{_fmt_float(agg.get('psi_some_mean'), 2, '%')} / {_fmt_float(agg.get('psi_some_max'), 2, '%')}**"
        )
    if agg.get("psi_full_mean") is not None or agg.get("psi_full_max") is not None:
        lines.append(
            f"- PSI memory 'full' (mean / max): **{_fmt_float(agg.get('psi_full_mean'), 2, '%')} / {_fmt_float(agg.get('psi_full_max'), 2, '%')}**"
        )

    if agg.get("numa_locality_mean") is not None or agg.get("numa_locality_max") is not None:
        lines.append(
            f"- NUMA locality ratio (mean / max): **{_fmt_float(agg.get('numa_locality_mean'), 1, '%')} / {_fmt_float(agg.get('numa_locality_max'), 1, '%')}**"
        )

    if agg.get("mem_available_mb_mean") is not None or agg.get("mem_available_pct_mean") is not None:
        lines.append(
            f"- System MemAvailable (mean): **{_fmt_float(agg.get('mem_available_mb_mean'), 1, 'MB')}**"
            f" — **{_fmt_float(agg.get('mem_available_pct_mean'), 1, '%')} of MemTotal**"
        )

    if len(lines) == 1:
        lines.append("- No extended memory metrics were present in the logs.")
    return "\n".join(lines) + "\n"


def _dedup_artifact_stats(model_dir: Optional[str], model_bin_size: Optional[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "defaults_path": None,
        "masks_path": None,
        "exceptions_path": None,
        "defaults_realpath": None,
        "masks_realpath": None,
        "exceptions_realpath": None,
        "defaults_size": None,
        "masks_size": None,
        "exceptions_size": None,
        "total_size": None,
        "ratio_vs_model_bin": None,
    }
    if not model_dir:
        return out

    defaults = os.path.join(model_dir, "model.defaults.bin")
    masks = os.path.join(model_dir, "model.masks.bin")
    exceptions = os.path.join(model_dir, "model.exceptions.bin")

    out["defaults_path"] = defaults
    out["masks_path"] = masks
    out["exceptions_path"] = exceptions

    def _one(p: str) -> Tuple[Optional[str], Optional[int]]:
        try:
            rp = os.path.realpath(p)
            if not os.path.exists(rp):
                return None, None
            return rp, os.path.getsize(rp)
        except Exception:
            return None, None

    d_rp, d_sz = _one(defaults)
    m_rp, m_sz = _one(masks)
    e_rp, e_sz = _one(exceptions)

    out["defaults_realpath"] = d_rp
    out["masks_realpath"] = m_rp
    out["exceptions_realpath"] = e_rp

    out["defaults_size"] = d_sz
    out["masks_size"] = m_sz
    out["exceptions_size"] = e_sz

    sizes = [x for x in (d_sz, m_sz, e_sz) if isinstance(x, int)]
    if sizes:
        total = int(sum(sizes))
        out["total_size"] = total
        if model_bin_size and model_bin_size > 0:
            out["ratio_vs_model_bin"] = float(total) / float(model_bin_size)
    return out


def _render_dedup_section(shared: Dict[str, Any], model_bin_size: Optional[int]) -> str:
    model_dir = shared.get("model_dir")
    st = _dedup_artifact_stats(model_dir, model_bin_size)

    ie_dedup = shared.get("ie_dedup")
    ie_dedup_strict = shared.get("ie_dedup_strict")
    ie_dedup_policy = shared.get("ie_dedup_policy")
    ie_dedup_cache_mb = shared.get("ie_dedup_cache_mb")

    def _fmt_bytes(n: Optional[int]) -> str:
        if n is None:
            return "n/a"
        return f"{n}"

    def _fmt_mb(n: Optional[int]) -> str:
        if n is None:
            return "n/a"
        return _fmt_float(_bytes_to_mb(float(n)), 2, "MB")

    model_bin_sz = model_bin_size
    model_bin_sz_mb = _fmt_float(_bytes_to_mb(float(model_bin_sz)) if model_bin_sz else None, 2, "MB")
    ratio = st.get("ratio_vs_model_bin")
    ratio_s = _fmt_float(float(ratio), 6, "", na="n/a") if ratio is not None else "n/a"

    lines: List[str] = []
    lines.append("### Deduplication")
    lines.append(f"- IE_DEDUP: **{ie_dedup if ie_dedup is not None else 'n/a'}**")
    lines.append(f"- IE_DEDUP_STRICT: **{ie_dedup_strict if ie_dedup_strict is not None else 'n/a'}**")
    lines.append(f"- IE_DEDUP_POLICY: **{ie_dedup_policy if ie_dedup_policy is not None else 'n/a'}**")
    lines.append(f"- IE_DEDUP_CACHE_MB: **{ie_dedup_cache_mb if ie_dedup_cache_mb is not None else 'n/a'}**")
    lines.append("- Artifacts (bytes / MB):")
    lines.append(f"  - model.defaults.bin: **{_fmt_bytes(st.get('defaults_size'))}** ({_fmt_mb(st.get('defaults_size'))})")
    lines.append(f"  - model.masks.bin: **{_fmt_bytes(st.get('masks_size'))}** ({_fmt_mb(st.get('masks_size'))})")
    lines.append(f"  - model.exceptions.bin: **{_fmt_bytes(st.get('exceptions_size'))}** ({_fmt_mb(st.get('exceptions_size'))})")
    lines.append(f"  - Total dedup blobs: **{_fmt_bytes(st.get('total_size'))}** ({_fmt_mb(st.get('total_size'))})")
    lines.append(f"- Dedup blobs / model.ie.bin: **{ratio_s}**")
    lines.append(f"- model.ie.bin size: **{model_bin_sz if model_bin_sz is not None else 'n/a'}** ({model_bin_sz_mb})")
    lines.append("- Artifact paths (best effort):")
    lines.append(f"  - defaults: `{st.get('defaults_path') or 'n/a'}` → `{st.get('defaults_realpath') or 'n/a'}`")
    lines.append(f"  - masks: `{st.get('masks_path') or 'n/a'}` → `{st.get('masks_realpath') or 'n/a'}`")
    lines.append(f"  - exceptions: `{st.get('exceptions_path') or 'n/a'}` → `{st.get('exceptions_realpath') or 'n/a'}`")
    return "\n".join(lines) + "\n"


def _render_device(title: str, agg: Agg, bpt: Optional[int], model_size: Optional[int]) -> str:
    bt = _bytes_touched(agg["tokens_sum"], bpt)
    bw = _bandwidth_gbps(bt, agg["wall_sum"])
    coverage = "n/a"
    if bpt and model_size and model_size > 0:
        coverage = f"{(float(bpt) / float(model_size)):.6f}"

    body = f"""## {title} — Summary (latest benchmark)
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
- Working-set coverage (bytes_per_token / model.ie.bin): **{coverage}**
- Effective bandwidth: **{_fmt_float(bw, 2, 'GB/s')}**
"""
    body += "\n" + _render_memory_details(agg)
    return body


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
- IE_DEDUP: **{shared.get('ie_dedup','n/a')}**
- IE_DEDUP_STRICT: **{shared.get('ie_dedup_strict','n/a')}**
- IE_DEDUP_POLICY: **{shared.get('ie_dedup_policy','n/a')}**
- IE_DEDUP_CACHE_MB: **{shared.get('ie_dedup_cache_mb','n/a')}**
"""


def _render_system(model_dir: Optional[str]) -> str:
    cpu = _discover_cpu_model() or "n/a"
    cores = os.cpu_count() or 0
    mem = _discover_mem_total_gb()
    os_name = _discover_os() or "n/a"
    kernel = _discover_kernel() or "n/a"
    git = _discover_git() or "n/a"
    model_bin = _find_model_bin(model_dir)
    model_size = _filesize_follow(model_bin)

    return f"""## System & Model Info
- CPU: **{cpu}**
- Logical cores: **{cores if cores else 'n/a'}**
- RAM (MemTotal): **{_fmt_float(mem, 1, 'GB')}**
- OS: **{os_name}**
- Kernel: **{kernel}**
- Git commit: **{git}**
- Model file: **{model_bin or 'n/a'}**
- Model size: **{_fmt_float(_bytes_to_gb(float(model_size)) if model_size else None, 3, 'GB')}**
"""


def _render_table(cpu_runs: List[Run], gpu_runs: List[Run]) -> str:
    lines = []
    lines.append("## Comparative Runs\n")
    lines.append("| Device | Run | Tokens | Wall (s) | TPS | p50 (ms) | p95 (ms) | RSS peak (MB) | PSS peak (MB) | VMS peak (MB) | minflt | majflt |")
    lines.append("|:------:|----:|-------:|---------:|----:|---------:|---------:|--------------:|--------------:|--------------:|------:|------:|")

    def row(dv: str, r: Run, idx: int) -> str:
        return (
            f"| {dv} | {idx} | "
            f"{int(r.get('tokens_generated', 0))} | "
            f"{_fmt_float(float(r.get('wall_time_s', 0.0)), 3)} | "
            f"{_fmt_float(float(r.get('tps_true', 0.0)), 3)} | "
            f"{_fmt_float(float(r.get('latency_p50_ms', r.get('p50_ms', 0.0))), 3)} | "
            f"{_fmt_float(float(r.get('latency_p95_ms', r.get('p95_ms', 0.0))), 3)} | "
            f"{_fmt_float(float(r.get('rss_peak_mb', 0.0)), 3)} | "
            f"{_fmt_float(float(r.get('pss_peak_mb', 0.0)) if r.get('pss_peak_mb') is not None else None, 3)} | "
            f"{_fmt_float(float(r.get('vms_peak_mb', 0.0)) if r.get('vms_peak_mb') is not None else None, 3)} | "
            f"{int(r.get('minflt', 0))} | {int(r.get('majflt', 0))} |"
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
        env_k = {
            "ie_dedup": "IE_DEDUP",
            "ie_dedup_strict": "IE_DEDUP_STRICT",
            "ie_dedup_policy": "IE_DEDUP_POLICY",
            "ie_dedup_cache_mb": "IE_DEDUP_CACHE_MB",
        }.get(k)
        if env_k and env_k in os.environ:
            v = os.environ.get(env_k)
            if k in ("ie_dedup", "ie_dedup_strict", "ie_dedup_cache_mb"):
                try:
                    return int(v) if v is not None else None
                except Exception:
                    return v
            return v
        return None

    keys = [
        "engine_bin",
        "prompts",
        "threads",
        "precision",
        "batch",
        "prefetch",
        "pretranspose",
        "affinity",
        "max_new",
        "ie_require_model",
        "ie_bytes_per_token",
        "ie_stride_bytes",
        "ie_verify_touch",
        "model_dir",
        "ie_dedup",
        "ie_dedup_strict",
        "ie_dedup_policy",
        "ie_dedup_cache_mb",
    ]
    return {k: g(k) for k in keys}


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update PERFORMANCE.md from strict JSON logs (last 3 runs per device).")
    p.add_argument("--cpu-json", default=None, help="Path to strict CPU JSONL")
    p.add_argument("--gpu-json", default=None, help="Path to strict GPU JSONL")
    p.add_argument("--out", default="docs/PERFORMANCE.md", help="Output Markdown file")

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

    p.add_argument("--ie-dedup", dest="ie_dedup", type=int, default=None)
    p.add_argument("--ie-dedup-strict", dest="ie_dedup_strict", type=int, default=None)
    p.add_argument("--ie-dedup-policy", dest="ie_dedup_policy", default=None)
    p.add_argument("--ie-dedup-cache-mb", dest="ie_dedup_cache_mb", type=int, default=None)

    p.add_argument("--model-dir", dest="model_dir", default=None)
    return p.parse_args()


def _extract_per_prompt_from_run(run: Optional[Run]) -> Optional[List[PromptRec]]:
    if not run or not isinstance(run, dict):
        return None

    candidates = [
        "per_prompt",
        "per_prompt_stats",
        "per_prompt_metrics",
        "prompt_stats",
        "prompt_metrics",
        "prompt_runs",
        "prompt_results",
        "samples",
        "results",
    ]
    for k in candidates:
        v = run.get(k)
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return list(v)

    for k in ("report", "details", "metrics", "bench"):
        v = run.get(k)
        if isinstance(v, dict):
            for kk in candidates:
                vv = v.get(kk)
                if isinstance(vv, list) and vv and all(isinstance(x, dict) for x in vv):
                    return list(vv)

    return None


def _prompt_key(rec: PromptRec, fallback_index_1based: int) -> str:
    for k in ("prompt_idx", "prompt_index", "idx", "index", "prompt_id", "id"):
        if k in rec:
            try:
                return str(int(rec[k]))
            except Exception:
                return str(rec[k])
    return str(fallback_index_1based)


def _short_prompt_text(rec: PromptRec, key: str, limit: int = 64) -> str:
    s = rec.get(key)
    if not isinstance(s, str) or not s:
        return ""
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    s = s.replace("|", "\\|")
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def _per_prompt_map(pp: List[PromptRec]) -> Dict[str, PromptRec]:
    out: Dict[str, PromptRec] = {}
    for i, rec in enumerate(pp, 1):
        out[_prompt_key(rec, i)] = rec
    return out


def _f(rec: Optional[PromptRec], k: str) -> Optional[float]:
    if not rec:
        return None
    v = rec.get(k)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _i(rec: Optional[PromptRec], k: str) -> Optional[int]:
    if not rec:
        return None
    v = rec.get(k)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _b(rec: Optional[PromptRec], k: str) -> Optional[bool]:
    if not rec:
        return None
    if k not in rec:
        return None
    v = rec.get(k)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "ok", "pass"):
            return True
        if s in ("0", "false", "f", "no", "n", "fail"):
            return False
    return None


def _extract_token_check(rec: Optional[PromptRec]) -> Tuple[Optional[bool], Optional[int], Optional[int], Optional[int]]:
    """
    Returns (ok, step, engine_id, hf_id) best-effort.

    Supported schemas:
    - Flat:
      - token_id_check_ok / token_ids_ok / ids_ok
      - token_id_check_first_mismatch_step / first_mismatch_step / mismatch_step
      - token_id_check_engine_id / engine_id / engine_next_id
      - token_id_check_hf_id / hf_id / hf_next_id
    - Nested:
      - token_id_check: { ok: bool, first_mismatch: { step, engine_id, hf_id } }
      - token_check / hf_check / id_check (similar shape)
    """
    if not rec or not isinstance(rec, dict):
        return None, None, None, None

    ok_keys = ("token_id_check_ok", "token_ids_ok", "ids_ok", "id_check_ok", "token_check_ok")
    step_keys = ("token_id_check_first_mismatch_step", "first_mismatch_step", "mismatch_step", "diverge_step")
    eng_keys = ("token_id_check_engine_id", "engine_id", "engine_next_id", "engine_token_id", "engine_next_token_id")
    hf_keys = ("token_id_check_hf_id", "hf_id", "hf_next_id", "hf_token_id", "hf_next_token_id")

    ok = None
    for k in ok_keys:
        ok = _b(rec, k)
        if ok is not None:
            break

    step = None
    for k in step_keys:
        step = _i(rec, k)
        if step is not None:
            break

    eng = None
    for k in eng_keys:
        eng = _i(rec, k)
        if eng is not None:
            break

    hf = None
    for k in hf_keys:
        hf = _i(rec, k)
        if hf is not None:
            break

    if ok is not None or step is not None or eng is not None or hf is not None:
        return ok, step, eng, hf

    for container_key in ("token_id_check", "token_check", "hf_check", "id_check", "verify_ids", "verify_hf_ids"):
        v = rec.get(container_key)
        if isinstance(v, dict):
            ok2 = None
            if "ok" in v:
                vv = v.get("ok")
                ok2 = vv if isinstance(vv, bool) else _b({"x": vv}, "x")
            if ok2 is None:
                for k in ok_keys:
                    if k in v:
                        vv = v.get(k)
                        ok2 = vv if isinstance(vv, bool) else _b({"x": vv}, "x")
                        if ok2 is not None:
                            break

            fm = v.get("first_mismatch")
            if not isinstance(fm, dict):
                fm = v.get("mismatch")
            if not isinstance(fm, dict):
                fm = v.get("first_failure")
            if not isinstance(fm, dict):
                fm = None

            step2 = None
            eng2 = None
            hf2 = None
            if fm:
                for k in ("step", "t", "index"):
                    step2 = _i(fm, k)
                    if step2 is not None:
                        break
                for k in ("engine_id", "engine_next_id", "engine", "engine_token"):
                    eng2 = _i(fm, k)
                    if eng2 is not None:
                        break
                for k in ("hf_id", "hf_next_id", "hf", "hf_token"):
                    hf2 = _i(fm, k)
                    if hf2 is not None:
                        break

            return ok2, step2, eng2, hf2

    return None, None, None, None


def _ok_str(ok: Optional[bool]) -> str:
    if ok is True:
        return "OK"
    if ok is False:
        return "FAIL"
    return "n/a"


def _mismatch_str(ok: Optional[bool], step: Optional[int], eng: Optional[int], hf: Optional[int]) -> str:
    if ok is True:
        return "-"
    if ok is False:
        parts: List[str] = []
        if step is not None:
            parts.append(f"step={step}")
        if eng is not None:
            parts.append(f"eng={eng}")
        if hf is not None:
            parts.append(f"hf={hf}")
        return " ".join(parts) if parts else "mismatch"
    if step is None and eng is None and hf is None:
        return "n/a"
    parts = []
    if step is not None:
        parts.append(f"step={step}")
    if eng is not None:
        parts.append(f"eng={eng}")
    if hf is not None:
        parts.append(f"hf={hf}")
    return " ".join(parts) if parts else "n/a"


def _render_per_prompt_side_by_side(cpu_run: Optional[Run], gpu_run: Optional[Run]) -> str:
    cpu_pp = _extract_per_prompt_from_run(cpu_run)
    gpu_pp = _extract_per_prompt_from_run(gpu_run)

    if not cpu_pp and not gpu_pp:
        return ""

    cpu_map = _per_prompt_map(cpu_pp or [])
    gpu_map = _per_prompt_map(gpu_pp or [])

    keys: List[str] = []
    seen = set()
    for rec in (cpu_pp or []):
        k = _prompt_key(rec, len(keys) + 1)
        if k not in seen:
            keys.append(k)
            seen.add(k)
    for rec in (gpu_pp or []):
        k = _prompt_key(rec, len(keys) + 1)
        if k not in seen:
            keys.append(k)
            seen.add(k)

    def _prompt_label(k: str, c: Optional[PromptRec], g: Optional[PromptRec]) -> str:
        txt = _short_prompt_text(c or {}, "prompt", 56) or _short_prompt_text(g or {}, "prompt", 56)
        if txt:
            return f"{k}: {txt}"
        return k

    lines: List[str] = []
    lines.append("## Per-Prompt CPU vs GPU (latest run)\n")
    lines.append(
        "| Prompt | CPU TPS | GPU TPS | Speedup | CPU wall (s) | GPU wall (s) | CPU tokens | GPU tokens | CPU KV hits | GPU KV hits | CPU ID check | GPU ID check | CPU mismatch | GPU mismatch |"
    )
    lines.append(
        "|:------|--------:|--------:|--------:|-------------:|-------------:|----------:|----------:|-----------:|-----------:|:-----------:|:-----------:|:------------|:------------|"
    )

    for k in keys:
        c = cpu_map.get(k)
        g = gpu_map.get(k)

        cpu_tps = _f(c, "tps_true")
        gpu_tps = _f(g, "tps_true")
        speed = None
        if cpu_tps and cpu_tps > 0 and gpu_tps is not None:
            speed = gpu_tps / cpu_tps

        cpu_wall = _f(c, "wall_time_s")
        gpu_wall = _f(g, "wall_time_s")

        cpu_tok = _i(c, "tokens_generated")
        gpu_tok = _i(g, "tokens_generated")

        cpu_kv = _i(c, "kv_hits")
        gpu_kv = _i(g, "kv_hits")

        c_ok, c_step, c_eng, c_hf = _extract_token_check(c)
        g_ok, g_step, g_eng, g_hf = _extract_token_check(g)

        lines.append(
            "| "
            + _prompt_label(k, c, g)
            + " | "
            + _fmt_float(cpu_tps, 3)
            + " | "
            + _fmt_float(gpu_tps, 3)
            + " | "
            + _fmt_float(speed, 3)
            + " | "
            + _fmt_float(cpu_wall, 3)
            + " | "
            + _fmt_float(gpu_wall, 3)
            + " | "
            + (str(cpu_tok) if cpu_tok is not None else "n/a")
            + " | "
            + (str(gpu_tok) if gpu_tok is not None else "n/a")
            + " | "
            + (str(cpu_kv) if cpu_kv is not None else "n/a")
            + " | "
            + (str(gpu_kv) if gpu_kv is not None else "n/a")
            + " | "
            + _ok_str(c_ok)
            + " | "
            + _ok_str(g_ok)
            + " | "
            + _mismatch_str(c_ok, c_step, c_eng, c_hf)
            + " | "
            + _mismatch_str(g_ok, g_step, g_eng, g_hf)
            + " |"
        )

    lines.append("")
    lines.append(
        "_Note: The ID-check columns render only if per-prompt records include token verification fields (e.g., `token_id_check_ok` or a nested `token_id_check` object). The mismatch column reports the first divergent step and IDs when available._\n"
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse()

    cpu_json = args.cpu_json or ("build/strict_cpu.json" if os.path.isfile("build/strict_cpu.json") else None)
    gpu_json = args.gpu_json or ("build/strict_gpu.json" if os.path.isfile("build/strict_gpu.json") else None)

    cpu_runs: List[Run] = []
    cpu_sum: Optional[Dict[str, Any]] = None
    if cpu_json and os.path.isfile(cpu_json):
        c_objs = _read_json_lines(cpu_json)
        c_all, cpu_sum = _partition_runs_and_summary(c_objs)
        cpu_runs = _take_last3(c_all)

    gpu_runs: List[Run] = []
    gpu_sum: Optional[Dict[str, Any]] = None
    if gpu_json and os.path.isfile(gpu_json):
        g_objs = _read_json_lines(gpu_json)
        g_all, gpu_sum = _partition_runs_and_summary(g_objs)
        gpu_runs = _take_last3(g_all)

    cpu_engine_bin = _pick_first(args.cpu_engine_bin, args.engine_bin)
    gpu_engine_bin = _pick_first(args.gpu_engine_bin, args.engine_bin)

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
        "ie_dedup": args.ie_dedup,
        "ie_dedup_strict": args.ie_dedup_strict,
        "ie_dedup_policy": args.ie_dedup_policy,
        "ie_dedup_cache_mb": args.ie_dedup_cache_mb,
    }

    shared = _merge_shared(cli_shared, cpu_sum, gpu_sum)

    if cpu_engine_bin:
        shared["engine_bin"] = cpu_engine_bin
    elif gpu_engine_bin and not shared.get("engine_bin"):
        shared["engine_bin"] = gpu_engine_bin

    bpt_val = shared.get("ie_bytes_per_token")
    try:
        bpt = int(bpt_val) if bpt_val is not None else None
    except Exception:
        bpt = None

    cpu_agg = _agg(cpu_runs) if cpu_runs else None
    gpu_agg = _agg(gpu_runs) if gpu_runs else None

    model_bin_path = _find_model_bin(shared.get("model_dir"))
    model_bin_size = _filesize_follow(model_bin_path)

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

    latest_cpu_run = cpu_runs[-1] if cpu_runs else None
    latest_gpu_run = gpu_runs[-1] if gpu_runs else None

    out_lines: List[str] = []
    out_lines.append("# Performance Notes\n")
    out_lines.append(f"_Last updated: **{_now_utc_str()}**_\n")
    out_lines.append(f"\n**Best true TPS:** **{best_label} — {_fmt_float(best_tps, 3)}**.\n")

    per_prompt_section = _render_per_prompt_side_by_side(latest_cpu_run, latest_gpu_run)
    if per_prompt_section:
        out_lines.append(per_prompt_section)

    if cpu_agg:
        out_lines.append(_render_device("CPU", cpu_agg, bpt, model_bin_size))
        out_lines.append(_render_dedup_section(shared, model_bin_size))
    if gpu_agg:
        out_lines.append(_render_device("GPU", gpu_agg, bpt, model_bin_size))
        out_lines.append(_render_dedup_section(shared, model_bin_size))

    out_lines.append(_render_shared(shared))
    out_lines.append(_render_system(shared.get("model_dir")))
    out_lines.append(_render_table(cpu_runs, gpu_runs))

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_lines))

    print(f"[update_performance_md] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
