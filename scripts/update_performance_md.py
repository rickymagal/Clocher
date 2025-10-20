#!/usr/bin/env python3
import os, re, json, statistics, datetime, glob, pathlib, sys, shutil, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmarks" / "reports"
PERF_MD = ROOT / "docs" / "PERFORMANCE.md"

def latest_dir_with(pattern: str) -> pathlib.Path | None:
    cand = []
    for d in REPORTS.glob("*"):
        if not d.is_dir(): continue
        if list(d.glob(pattern)):
            cand.append(d)
    if not cand: return None
    return sorted(cand)[-1]

def load_runs_jsonl(run_dir: pathlib.Path) -> list[dict]:
    rows = []
    f = run_dir / "runs.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"): continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                pass
    for jf in sorted(run_dir.glob("run_*.json")):
        try:
            rows.append(json.loads(jf.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            pass
    return rows

def pick(j, k, alts=()):
    if isinstance(j, dict) and k in j: return j[k]
    for a in alts:
        if a in j: return j[a]
    return None

def fmt_bytes(n):
    try:
        n = int(n)
    except Exception:
        return "n/a"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.1f} {units[i]}"

def sys_info():
    out = {}
    # CPU / cores
    try:
        out["cpu_model"] = subprocess.check_output(["bash","-lc","lscpu | awk -F: '/Model name/ {gsub(/^ +/,\"\",$2); print $2; exit}'"], text=True).strip()
    except Exception:
        out["cpu_model"] = ""
    try:
        out["cores_logical"] = int(os.cpu_count() or 0)
    except Exception:
        out["cores_logical"] = 0
    # MemTotal
    try:
        mem_kb = 0
        with open("/proc/meminfo","r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1]); break
        out["mem_total_gb"] = round(mem_kb/1024/1024, 1) if mem_kb>0 else 0
    except Exception:
        out["mem_total_gb"] = 0
    # OS / kernel
    try:
        out["kernel"] = subprocess.check_output(["bash","-lc","uname -r"], text=True).strip()
    except Exception:
        out["kernel"] = ""
    try:
        out["distro"] = subprocess.check_output(["bash","-lc",". /etc/os-release && echo \"$NAME $VERSION\""], text=True).strip()
    except Exception:
        out["distro"] = ""
    return out

def model_info(model_dir: pathlib.Path):
    info = {"dir": str(model_dir)}
    json_p = model_dir / "model.ie.json"
    bin_p  = model_dir / "model.ie.bin"
    vocab  = model_dir / "vocab.json"
    info["json_path"] = str(json_p)
    info["bin_path"]  = str(bin_p)
    info["vocab_path"]= str(vocab)
    info["json_size"] = json_p.stat().st_size if json_p.exists() else 0
    info["bin_size"]  = bin_p.stat().st_size if bin_p.exists() else 0
    info["json_mtime"]= datetime.datetime.fromtimestamp(json_p.stat().st_mtime, tz=datetime.timezone.utc).isoformat() if json_p.exists() else ""
    info["bin_mtime"] = datetime.datetime.fromtimestamp(bin_p.stat().st_mtime, tz=datetime.timezone.utc).isoformat() if bin_p.exists() else ""
    # dtype / tensors
    info["dtype"] = ""
    info["tensors_count"] = None
    try:
        j = json.loads(json_p.read_text(encoding="utf-8"))
        info["dtype"] = str(j.get("dtype",""))
        if "tensors" in j and isinstance(j["tensors"], list):
            info["tensors_count"] = len(j["tensors"])
    except Exception:
        pass
    return info

def summarize_bench_from_strict(strict_json_path: pathlib.Path|None) -> tuple[str, list[str]]:
    if strict_json_path and strict_json_path.exists():
        try:
            j = json.loads(strict_json_path.read_text(encoding="utf-8"))
        except Exception:
            j = {}
        lines = []
        lines.append(f"- Source: `strict` (scripts/true_tps_strict.sh)")
        runs = pick(j, "runs", ("n_runs","rounds")) or 0
        tok  = pick(j, "tokens_generated", ("tokens","total_tokens","tokens_total")) or 0
        wall = pick(j, "wall_time_s", ("elapsed_s","wall","walltime_s")) or 0.0
        tps  = pick(j, "tps_true", ("tps","true_tps")) or ( (tok/ wall) if wall else 0.0 )
        lines.append(f"- Runs: **{runs}**")
        lines.append(f"- Tokens (total): **{tok}**")
        lines.append(f"- Wall time (s): **{wall:.6f}**")
        lines.append(f"- TPS (true): **{tps:.3f}**")
        return ("## Summary (latest benchmark)", lines)
    # fallback to older harness dirs
    run_dir = latest_dir_with("runs.jsonl")
    if run_dir:
        rows = load_runs_jsonl(run_dir)
        tps = [pick(r,"tps_true",("tps","true_tps")) for r in rows if isinstance(pick(r,"tps_true",("tps","true_tps")), (int,float))]
        p50 = [pick(r,"latency_p50",("p50","lat_p50")) for r in rows if pick(r,"latency_p50",("p50","lat_p50")) is not None]
        p95 = [pick(r,"latency_p95",("p95","lat_p95")) for r in rows if pick(r,"latency_p95",("p95","lat_p95")) is not None]
        tt  = [pick(r,"total_tokens",("tokens_generated","tokens","tokens_total")) for r in rows if pick(r,"total_tokens",("tokens_generated","tokens","tokens_total")) is not None]
        wl  = [pick(r,"wall_time_s",("elapsed_s","wall","walltime_s")) for r in rows if pick(r,"wall_time_s",("elapsed_s","wall","walltime_s")) is not None]
        lines = []
        lines.append(f"- Reports directory: `{str(run_dir)}`")
        lines.append(f"- TPS (true): **{statistics.fmean(tps):.3f}**" if tps else "- TPS (true): **n/a**")
        lines.append(f"- Latency p50: **{(statistics.fmean(p50)):.6f}**" if p50 else "- Latency p50: **n/a**")
        lines.append(f"- Latency p95: **{(statistics.fmean(p95)):.6f}**" if p95 else "- Latency p95: **n/a**")
        if tt and wl and sum(wl)>0:
            agg = sum(tt)/sum(wl)
            lines.append(f"- TPS (aggregate): **{agg:.3f}**")
        return ("## Summary (latest benchmark)", lines)
    return ("## Summary (latest benchmark)", ["- No benchmark report found under `benchmarks/reports/`"])

def summarize_profile(model_dir: pathlib.Path) -> tuple[str, list[str]]:
    fg = ROOT / "flamegraph.svg"
    perfdata = ROOT / "perf.data"
    # also accept flamegraph dropped in model dir by the script
    if not fg.exists():
        fg = model_dir / "flamegraph.svg"
    if not perfdata.exists():
        perfdata = ROOT / "perf.data"
    lines = []
    lines.append(f"- `flamegraph.svg`: **present**" if fg.exists() else "- `flamegraph.svg`: **missing**")
    lines.append(f"- `perf.data`: **present**" if perfdata.exists() else "- `perf.data`: **missing**")
    hints_header = "## Hot Paths (annotated)"
    hints = [
        "- GEMV (`ie_gemv_f32`): AVX2 microkernel if available; otherwise generic path.",
        "- Activation (`tanh` fast path): clamped polynomial/table approximation.",
        "- Thread pool scheduling: contiguous shard with grainsize control and optional pinning.",
    ]
    next_header = "## Next optimization actions"
    next_actions = [
        "- Validate NUMA policy using `scripts/set_numa.sh` (`interleave|node:X|strict`).",
        "- Explore epilogue fusion (bias + activation) on GEMV output.",
        "- Extend blocked-K packing and tune prefetch distances based on the flamegraph.",
    ]
    return ("## Profiling Artifacts", lines + ["", hints_header] + hints + ["", next_header] + next_actions)

def render(strict_json_path: pathlib.Path|None, model_dir: pathlib.Path) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = []
    out.append("# Performance Notes")
    out.append("")
    out.append(f"_Last updated: **{ts}**_")
    out.append("")

    # Summary
    h, lines = summarize_bench_from_strict(strict_json_path)
    out.append(h); out.extend(lines); out.append("")

    # Run parameters (from env)
    env = os.environ.get
    out.append("## Run parameters")
    out.extend([
        f"- Threads: **{env('THREADS') or ''}**",
        f"- Precision: **{env('PRECISION') or ''}**",
        f"- Pretranspose: **{env('PRETRANSPOSE') or ''}**",
        f"- Batch: **{env('BATCH') or ''}**",
        f"- Prefetch: **{env('PREFETCH') or ''}**",
        f"- Max new tokens: **{env('MAX_NEW') or ''}**",
        f"- Target seconds: **{env('TARGET_SECONDS') or ''}**",
        f"- Prompts file: **{env('PROMPTS') or ''}**",
        f"- Affinity policy (CLI): **{env('AFFINITY') or ''}**",
        f"- IE_BYTES_PER_TOKEN: **{env('IE_BYTES_PER_TOKEN') or ''}**",
        f"- IE_STRIDE_BYTES: **{env('IE_STRIDE_BYTES') or ''}**",
        f"- IE_VERIFY_TOUCH: **{env('IE_VERIFY_TOUCH') or ''}**",
        ""
    ])

    # System & Model Info
    s = sys_info()
    m = model_info(model_dir)
    out.append("## System & Model Info")
    out.extend([
        f"- CPU: **{s.get('cpu_model','')}**",
        f"- Logical cores: **{s.get('cores_logical',0)}**",
        f"- RAM (MemTotal): **{s.get('mem_total_gb',0)} GB**",
        f"- OS: **{s.get('distro','')}**",
        f"- Kernel: **{s.get('kernel','')}**",
        f"- Model dir: `{m['dir']}`",
        f"- model.ie.json: `{m['json_path']}` ({fmt_bytes(m['json_size'])}, mtime {m['json_mtime']})",
        f"- model.ie.bin: `{m['bin_path']}` ({fmt_bytes(m['bin_size'])}, mtime {m['bin_mtime']})",
        f"- vocab.json: `{m['vocab_path']}`",
        f"- Dtype: **{m.get('dtype','')}**; Tensors: **{m.get('tensors_count','n/a')}**",
        ""
    ])

    # Profiling
    h2, lines2 = summarize_profile(model_dir)
    out.append(h2); out.extend(lines2); out.append("")
    return "\n".join(out).strip() + "\n"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-json", type=str, default="")
    args = ap.parse_args()

    strict_path = pathlib.Path(args.strict_json).resolve() if args.strict_json else None
    model_dir = pathlib.Path(os.environ.get("MODEL_DIR") or os.environ.get("MODEL") or "models/gpt-oss-20b").resolve()

    REPORTS.mkdir(parents=True, exist_ok=True)
    text = render(strict_path if (strict_path and strict_path.exists()) else None, model_dir)

    PERF_MD.parent.mkdir(parents=True, exist_ok=True)
    if not PERF_MD.exists():
        PERF_MD.write_text(text, encoding="utf-8")
    else:
        content = PERF_MD.read_text(encoding="utf-8", errors="ignore")
        pattern = r"(?s)^# Performance Notes.*?(?=^\# |\Z)"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, text, content, flags=re.MULTILINE)
        else:
            content = text + "\n\n" + content
        PERF_MD.write_text(content, encoding="utf-8")
    print(f"[ok] wrote: {PERF_MD}")

if __name__ == "__main__":
    sys.exit(main())
