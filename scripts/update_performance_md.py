#!/usr/bin/env python3
"""
Update docs/PERFORMANCE.md (English) from the latest benchmark/profiling artifacts.

Sources & precedence:
1) metrics: summary.json {tps_true|avg_tps_true, latency_p50_ms|p50_ms, latency_p95_ms|p95_ms}
2) if missing, aggregate samples.csv (comment-tolerant):
   - mean TPS from tps_true or tokens_generated/wall_time_s
   - median p50/p95 if present; else average per-token latency as fallback
3) params: params.json → summary.json keys → first commented line in samples.csv (k=v) →
   parse 'cmdline' (e.g., "./build/inference-engine --threads 4 --precision bf16 ...")

Writes: docs/PERFORMANCE.md
"""
import json, csv, os, re
from pathlib import Path
from datetime import datetime

PARAM_KEYS = ("threads","precision","pretranspose","affinity")

def repo_root_from_this_script() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent

def latest_report_dir(reports_dir: Path):
    if not reports_dir.exists():
        return None
    dirs = sorted([p for p in reports_dir.iterdir() if p.is_dir()], reverse=True)
    return dirs[0] if dirs else None

def load_json(p: Path) -> dict:
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def coalesce_metrics(summary: dict) -> dict:
    tps = summary.get("tps_true") or summary.get("avg_tps_true") or summary.get("tps")
    p50 = summary.get("latency_p50_ms") or summary.get("p50_ms") or summary.get("p50")
    p95 = summary.get("latency_p95_ms") or summary.get("p95_ms") or summary.get("p95")
    return {"tps_true": tps, "latency_p50_ms": p50, "latency_p95_ms": p95}

def _strip_comment_lines(lines):
    first_comment = None
    out = []
    for ln in lines:
        if ln.startswith("#"):
            if first_comment is None:
                first_comment = ln.strip("# \n")
            continue
        out.append(ln)
    return out, first_comment

def _dicts_from_csv(csv_path: Path):
    if not csv_path.exists():
        return [], None
    text = csv_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    clean, first_comment = _strip_comment_lines(lines)
    if not clean:
        return [], first_comment
    import io
    rdr = csv.DictReader(io.StringIO("\n".join(clean)))
    return list(rdr), first_comment

def aggregate_from_csv(csv_path: Path):
    rows, first_comment = _dicts_from_csv(csv_path)
    metrics, params = {}, {}

    tps_vals, p50_vals, p95_vals = [], [], []
    avg_latencies = []

    for row in rows:
        # Parse cmdline flags if a 'cmdline' column exists
        if row.get('cmdline'):
            for k,v in re.findall(r"--(threads|precision|pretranspose|affinity)\s+([^\s]+)", row['cmdline']):
                params[k] = v

        # params as columns
        for k in PARAM_KEYS:
            if k in row and row[k]:
                params[k] = row[k]

        # TPS
        if row.get("tps_true"):
            try: tps_vals.append(float(row["tps_true"]))
            except: pass
        else:
            try:
                tg = float(row.get("tokens_generated","") or row.get("tokens",""))
                wt = float(row.get("wall_time_s",""))
                if wt > 0:
                    tps_vals.append(tg / wt)
                if tg > 0 and wt >= 0:
                    avg_latencies.append((wt * 1000.0)/tg)
            except: pass

        # p50
        for k in ("latency_p50_ms","p50_ms","p50"):
            if row.get(k):
                try: p50_vals.append(float(row[k])); break
                except: pass

        # p95
        for k in ("latency_p95_ms","p95_ms","p95"):
            if row.get(k):
                try: p95_vals.append(float(row[k])); break
                except: pass

    if tps_vals: metrics["tps_true"] = sum(tps_vals)/len(tps_vals)
    if p50_vals: metrics["latency_p50_ms"] = sorted(p50_vals)[len(p50_vals)//2]
    if p95_vals: metrics["latency_p95_ms"] = sorted(p95_vals)[len(p95_vals)//2]
    if (metrics.get("latency_p50_ms") is None or metrics.get("latency_p95_ms") is None) and avg_latencies:
        avg_ms = sum(avg_latencies)/len(avg_latencies)
        metrics.setdefault("latency_p50_ms", avg_ms)
        metrics.setdefault("latency_p95_ms", avg_ms)

    # parse params from first comment line if needed: "k=v k=v"
    if first_comment:
        for kv in re.findall(r"(\w+)=([\w\.\-:]+)", first_comment):
            params.setdefault(kv[0], kv[1])

    return metrics, params

def parse_params_from_cmdline(cmd: str) -> dict:
    out = {}
    if not cmd: return out
    # Example: ./build/inference-engine --threads 4 --precision bf16 --pretranspose wxh --affinity scatter
    m = re.findall(r"--(threads|precision|pretranspose|affinity)\s+([^\s]+)", cmd)
    for k,v in m:
        out[k] = v
    return out

def merge_params(primary: dict, *others):
    out = dict(primary)
    for d in others:
        for k in PARAM_KEYS:
            if k not in out or out[k] in ("", None):
                if k in d and d[k]:
                    out[k] = d[k]
    return out

def detect_cpu_features() -> str:
    try:
        data = Path("/proc/cpuinfo").read_text(encoding="utf-8").lower()
        feats = []
        if "avx2" in data: feats.append("avx2")
        if re.search(r"\bfma\b", data): feats.append("fma")
        if "sse4_2" in data or "sse4.2" in data: feats.append("sse4.2")
        return ", ".join(feats) if feats else "unknown"
    except Exception:
        return "unknown"

def update_md(root: Path, metrics: dict, params: dict, report_dir: Path):
    flame = (root / "flamegraph.svg").exists()
    perfd = (root / "perf.data").exists()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def fmt(x, suffix=""):
        if x is None: return "n/a" + (f" {suffix}" if suffix else "")
        try: return f"{float(x):.3f}{(' ' + suffix) if suffix else ''}"
        except: return f"{x}"

    tps = metrics.get("tps_true")
    p50 = metrics.get("latency_p50_ms")
    p95 = metrics.get("latency_p95_ms")

    threads = params.get("threads","n/a")
    precision = params.get("precision","fp32")
    pretranspose = params.get("pretranspose","n/a")
    affinity_flag = params.get("affinity","auto")
    affinity_env = "enabled" if os.environ.get("IE_TP_USE_AFFINITY","") == "1" else "disabled"
    cpu_feats = detect_cpu_features()

    md = f"""# Performance Notes

_Last updated: **{ts}**_

## Summary (latest run)
- Reports directory: `{report_dir if report_dir else "(no reports found)"}`
- TPS (true): **{fmt(tps)}**
- Latency p50: **{fmt(p50, "ms")}**
- Latency p95: **{fmt(p95, "ms")}**

## Run parameters
- Threads: **{threads}**
- Precision: **{precision}**
- Pretranspose: **{pretranspose}**
- Affinity policy (CLI): **{affinity_flag}**
- Affinity env toggle (`IE_TP_USE_AFFINITY`): **{affinity_env}**
- Detected CPU features: **{cpu_feats}**

## Profiling Artifacts
- `flamegraph.svg`: **{"present" if flame else "absent"}**
- `perf.data`: **{"present" if perfd else "absent"}**

## Hot Paths (annotated)
- GEMV (`ie_gemv_f32`): AVX2 microkernel if available; otherwise generic path.
- Activation (`tanh` fast path): clamped polynomial/table approximation.
- Thread pool scheduling: contiguous shard with grainsize control and optional pinning.

## Next optimization actions
- Validate NUMA policy impacts using `scripts/set_numa.sh` (`interleave|node:X|strict`).
- Explore epilogue fusion (bias + activation) in GEMV output.
- Extend blocked-K packing and prefetch distances based on flamegraph evidence.
"""
    out_md = root / "docs" / "PERFORMANCE.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"[ok] PERFORMANCE.md updated at: {out_md}")

def main():
    root = repo_root_from_this_script()
    reports_dir = root / "benchmarks" / "reports"
    rpt = latest_report_dir(reports_dir)
    summary = load_json(rpt / "summary.json") if rpt else {}
    metrics = coalesce_metrics(summary)

    # params: params.json → summary keys → csv → cmdline within summary
    params_json = load_json(rpt / "params.json") if rpt else {}
    params_from_summary = {k: summary.get(k) for k in PARAM_KEYS if summary.get(k) is not None}
    csv_metrics, csv_params = aggregate_from_csv(rpt / "samples.csv") if rpt else ({}, {})
    cmdline_params = parse_params_from_cmdline(summary.get("cmdline","")) if summary else {}

    metrics = {**csv_metrics, **{k:v for k,v in metrics.items() if v is not None}} if csv_metrics else metrics
    params = merge_params(params_json, params_from_summary, csv_params, cmdline_params)
    # Sensible defaults if still missing
    if not params.get('threads'):
        try:
            params['threads'] = str(os.cpu_count())
        except Exception:
            params['threads'] = 'auto'
    if not params.get('pretranspose'):
        params['pretranspose'] = 'none'
    if not params.get('affinity'):
        params['affinity'] = 'auto'

    update_md(root, metrics, params, rpt)

if __name__ == "__main__":
    main()
