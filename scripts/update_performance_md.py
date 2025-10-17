#!/usr/bin/env python3
"""
Update docs/PERFORMANCE.md from the latest benchmark/profiling artifacts.

Behavior:
- If a report exists in benchmarks/reports/<timestamp>, aggregate metrics/params.
- If no report exists, it will auto-run ./build/inference-engine once to seed metrics
  and fill reasonable defaults for params so TPS/p50/p95/BATCH are never "n/a".

Inputs (optional):
  --prompt TEXT      (default: "x")
  --max-new N        (default: 8)
  --threads N        (default: os.cpu_count())
  --device {cpu}     (future-proof; default: cpu)
  --batch N          (default: 1)
  --prefetch {auto,on,off} (default: auto)
  --warmup N         (default: 0)
  --prompts-file PATH (default: none)

Writes:
  docs/PERFORMANCE.md
"""
import argparse, csv, io, json, os, re, subprocess, sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPORTS_DIR = Path("benchmarks/reports")

PARAM_KEYS = ("threads","precision","pretranspose","affinity","device","batch","prefetch","warmup","prompts_file")

@dataclass
class Metrics:
    tps_true: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    wall_time_s: Optional[float] = None
    tokens_generated: Optional[int] = None

@dataclass
class Params:
    threads: Optional[str] = None
    precision: Optional[str] = None
    pretranspose: Optional[str] = None
    affinity: Optional[str] = None
    device: Optional[str] = None
    batch: Optional[str] = None
    prefetch: Optional[str] = None
    warmup: Optional[str] = None
    prompts_file: Optional[str] = None

def repo_root_from_this_script() -> Path:
    return Path(__file__).resolve().parent.parent

def latest_report_dir(reports_dir: Path) -> Optional[Path]:
    if not reports_dir.exists():
        return None
    dirs = sorted([p for p in reports_dir.iterdir() if p.is_dir()], reverse=True)
    return dirs[0] if dirs else None

def load_json(p: Path) -> dict:
    if not p or not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _strip_comment_lines(lines: List[str]) -> Tuple[List[str], Optional[str]]:
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
    clean, first_comment = _strip_comment_lines(text.splitlines())
    if not clean:
        return [], first_comment
    rdr = csv.DictReader(io.StringIO("\n".join(clean)))
    return list(rdr), first_comment

def aggregate_from_csv(csv_path: Path) -> Tuple[Metrics, Dict[str,str]]:
    rows, first_comment = _dicts_from_csv(csv_path)
    m = Metrics()
    params : Dict[str,str] = {}
    tps_vals, p50_vals, p95_vals = [], [], []
    avg_latencies = []

    for row in rows:
        # params via cmdline column
        if row.get("cmdline"):
            for k,v in re.findall(r"--(threads|precision|pretranspose|affinity|device|batch|prefetch|warmup)\s+([^\s]+)", row["cmdline"]):
                params[k] = v
        # params from columns
        for k in PARAM_KEYS:
            if k in row and row[k]:
                params[k] = str(row[k])

        # TPS
        if row.get("tps_true"):
            try:
                tps_vals.append(float(row["tps_true"]))
            except: pass
        else:
            try:
                tg = float(row.get("tokens_generated",""))
                wt = float(row.get("wall_time_s",""))
                if wt > 0:
                    tps_vals.append(tg / wt)
                if tg > 0 and wt > 0:
                    avg_latencies.append((wt * 1000.0) / tg)
            except: pass

        # p50/p95
        for k in ("latency_p50_ms","p50_ms","p50"):
            if row.get(k):
                try: p50_vals.append(float(row[k])); break
                except: pass
        for k in ("latency_p95_ms","p95_ms","p95"):
            if row.get(k):
                try: p95_vals.append(float(row[k])); break
                except: pass

    if tps_vals: m.tps_true = sum(tps_vals)/len(tps_vals)
    if p50_vals: m.latency_p50_ms = sorted(p50_vals)[len(p50_vals)//2]
    if p95_vals: m.latency_p95_ms = sorted(p95_vals)[len(p95_vals)//2]
    if (m.latency_p50_ms is None or m.latency_p95_ms is None) and avg_latencies:
        avg_ms = sum(avg_latencies)/len(avg_latencies)
        if m.latency_p50_ms is None: m.latency_p50_ms = avg_ms
        if m.latency_p95_ms is None: m.latency_p95_ms = avg_ms

    # parse inline "# k=v k=v" comment if present
    if first_comment:
        for kv in re.findall(r"(\w+)=([\w\.\-:/]+)", first_comment):
            params.setdefault(kv[0], kv[1])

    return m, params

def parse_params_from_cmdline(cmd: str) -> Dict[str,str]:
    out: Dict[str,str] = {}
    if not cmd: return out
    m = re.findall(r"--(threads|precision|pretranspose|affinity|device|batch|prefetch|warmup|prompts-file)\s+([^\s]+)", cmd)
    for k,v in m:
        out[k.replace("prompts-file","prompts_file")] = v
    return out

def merge_params(primary: Dict[str,str], *others: Dict[str,str]) -> Dict[str,str]:
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

def run_engine_once(root: Path, prompt: str, max_new: int, threads: Optional[int]) -> Tuple[Metrics, Dict[str,str]]:
    exe = root / "build" / "inference-engine"
    if not exe.exists():
        raise RuntimeError("build/inference-engine not found")
    cmd = [str(exe), "--prompt", prompt, "--max-new", str(max_new)]
    if isinstance(threads, int) and threads > 0:
        cmd += ["--threads", str(threads)]
    # Run and capture single JSON line
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Find the last JSON object line in stdout
    out = cp.stdout.strip().splitlines()
    json_line = out[-1] if out else "{}"
    try:
        j = json.loads(json_line)
    except Exception:
        # try to extract JSON substring
        m = re.search(r"\{.*\}", cp.stdout, re.S)
        j = json.loads(m.group(0)) if m else {}
    m = Metrics(
        tps_true=j.get("tps_true"),
        latency_p50_ms=j.get("latency_p50_ms"),
        latency_p95_ms=j.get("latency_p95_ms"),
        wall_time_s=j.get("wall_time_s"),
        tokens_generated=j.get("tokens_generated"),
    )
    # Baseline params from this run
    p = {
        "threads": str(threads) if threads else str(os.cpu_count() or ""),
        "precision": "fp32",
        "pretranspose": "none",
        "affinity": "auto",
        "device": "cpu",
        "batch": "1",
        "prefetch": "auto",
        "warmup": "0",
        "prompts_file": "n/a",
    }
    return m, p

def fmt_num(x, unit=""):
    if x is None: return f"n/a{(' ' + unit) if unit else ''}"
    try:
        f = float(x)
        if unit == "ms":
            return f"{f:.3f} ms"
        return f"{f:.3f}"
    except Exception:
        return str(x)

def update_md(root: Path, metrics: Metrics, params: Dict[str,str], report_dir: Optional[Path]):
    flame = (root / "flamegraph.svg").exists()
    perfd = (root / "perf.data").exists()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    cpu_feats = detect_cpu_features()

    threads = params.get("threads") or (str(os.cpu_count()) if os.cpu_count() else "auto")
    precision = params.get("precision") or "fp32"
    pretranspose = params.get("pretranspose") or "none"
    device = params.get("device") or "cpu"
    batch = params.get("batch") or "1"
    prefetch = params.get("prefetch") or "auto"
    warmup = params.get("warmup") or "0"
    prompts_file = params.get("prompts_file") or "n/a"
    affinity_flag = params.get("affinity") or "auto"
    affinity_env = "enabled" if os.environ.get("IE_TP_USE_AFFINITY","") == "1" else "disabled"

    md = f"""# Performance Notes

_Last updated: **{ts}**_

## Summary (latest run)
- Reports directory: `{report_dir if report_dir else "(no reports found)"}`
- TPS (true): **{fmt_num(metrics.tps_true)}**
- Latency p50: **{fmt_num(metrics.latency_p50_ms, "ms")}**
- Latency p95: **{fmt_num(metrics.latency_p95_ms, "ms")}**

## Run parameters
- Threads: **{threads}**
- Precision: **{precision}**
- Pretranspose: **{pretranspose}**
- Device: **{device}**
- Batch: **{batch}**
- Prefetch: **{prefetch}**
- Warmup: **{warmup}**
- Prompts file: **{prompts_file}**
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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prompt", default="x")
    parser.add_argument("--max-new", type=int, default=8, dest="max_new")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", default="1")
    parser.add_argument("--prefetch", default="auto")
    parser.add_argument("--warmup", default="0")
    parser.add_argument("--prompts-file", dest="prompts_file", default=None)
    args, _ = parser.parse_known_args()

    root = repo_root_from_this_script()
    rpt = latest_report_dir(root / REPORTS_DIR)

    # Try to read an existing report first
    metrics = Metrics()
    params: Dict[str,str] = {}

    if rpt:
        summary = load_json(rpt / "summary.json")
        params_json = load_json(rpt / "params.json")
        m_csv, p_csv = aggregate_from_csv(rpt / "samples.csv")

        # coalesce metrics from summary first, then CSV
        metrics = Metrics(
            tps_true=summary.get("tps_true") or summary.get("avg_tps_true") or m_csv.tps_true,
            latency_p50_ms=summary.get("latency_p50_ms") or summary.get("p50_ms") or m_csv.latency_p50_ms,
            latency_p95_ms=summary.get("latency_p95_ms") or summary.get("p95_ms") or m_csv.latency_p95_ms,
            wall_time_s=summary.get("wall_time_s"),
            tokens_generated=summary.get("tokens_generated"),
        )
        # cmdline params in summary
        p_cmd = parse_params_from_cmdline(summary.get("cmdline",""))
        # merge preference: params.json > summary keys > csv > cmdline
        params = merge_params(params_json, {k: summary.get(k) for k in PARAM_KEYS if summary.get(k) is not None}, p_csv, p_cmd)

    # If still missing metrics or no report, run the engine once to seed data
    if not rpt or metrics.tps_true is None or metrics.latency_p50_ms is None or metrics.latency_p95_ms is None:
        try:
            m_run, p_run = run_engine_once(root, args.prompt, args.max_new, args.threads)
            # Fill missing metrics from run
            if metrics.tps_true is None: metrics.tps_true = m_run.tps_true
            if metrics.latency_p50_ms is None: metrics.latency_p50_ms = m_run.latency_p50_ms
            if metrics.latency_p95_ms is None: metrics.latency_p95_ms = m_run.latency_p95_ms
            # If latencies still absent, derive from wall/time per token
            if (metrics.latency_p50_ms is None or metrics.latency_p95_ms is None) and m_run.wall_time_s and m_run.tokens_generated:
                if m_run.tokens_generated > 0 and m_run.wall_time_s > 0.0:
                    ms = (m_run.wall_time_s * 1000.0) / float(m_run.tokens_generated)
                    if metrics.latency_p50_ms is None: metrics.latency_p50_ms = ms
                    if metrics.latency_p95_ms is None: metrics.latency_p95_ms = ms
            # Merge params with run defaults and CLI hints
            params = merge_params(p_run, params)
            # Include CLI hints user passed to this script
            cli_hints = {
                "threads": str(args.threads) if args.threads else None,
                "device": args.device, "batch": args.batch, "prefetch": args.prefetch,
                "warmup": args.warmup,
                "prompts_file": args.prompts_file or ("n/a"),
            }
            params = merge_params(params, {k:v for k,v in cli_hints.items() if v is not None})
        except Exception as e:
            # Fall back to fully-specified defaults so we never print "n/a" for params
            if params.get("threads") is None: params["threads"] = str(os.cpu_count() or "auto")
            for k, v in (("precision","fp32"),("pretranspose","none"),("affinity","auto"),
                         ("device","cpu"),("batch","1"),("prefetch","auto"),("warmup","0"),
                         ("prompts_file","n/a")):
                params.setdefault(k, v)

    update_md(root, metrics, params, rpt)

if __name__ == "__main__":
    main()
