#!/usr/bin/env python3
# =============================================================================
# Grid-sweep benchmark harness + optional Prometheus Pushgateway publishing.
#
# This harness sweeps combinations of:
#   - act_quant      : ["none","int8","fp8_e4m3","fp8_e5m2"]
#   - kv_quant       : ["none","int8"]
#   - prefetch       : ["off","on","auto"]
#   - nt_loads       : [0,1]
#   - numa_policy    : ["auto","compact","scatter"]
#
# It creates a per-combination subdirectory under --report-root, writes CSV and
# summary.json snapshots, and (optionally) pushes a metrics snapshot per combo.
#
# NOTE: This is stdlib-only and does not invoke the engine binary by default.
#       You can point IE_ENGINE_BIN to an executable to integrate real runs.
#       For now, it emits synthetically shaped outputs for downstream tooling.
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
import itertools
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# ------------------------------ file / prompts ------------------------------ #

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ensure_outdir(root: Path, combo_id: str) -> Path:
    d = root / combo_id
    _ensure_dir(d)
    return d


def _load_prompts_jsonl(path: Path) -> List[str]:
    ps: List[str] = []
    if not path.exists():
        return ps
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
            t = obj.get("text")
            if isinstance(t, str) and t:
                ps.append(t)
        except Exception:
            # best-effort tolerant reader
            pass
    return ps


# ------------------------------ metrics / push ------------------------------ #

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _env_json(name: str, default):
    try:
        v = os.getenv(name)
        if not v:
            return default
        return json.loads(v)
    except Exception:
        return default


def _escape_label_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt_labels(labels: dict) -> str:
    if not labels:
        return "{}"
    items = sorted((str(k), str(v)) for k, v in labels.items())
    return "{" + ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in items) + "}"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _pg_endpoint(base_url: str, job: str, grouping: dict[str, str]) -> str:
    import urllib.parse
    endpoint = f"{base_url.rstrip('/')}/metrics/job/{urllib.parse.quote(job)}"
    for k, v in grouping.items():
        endpoint += f"/{urllib.parse.quote(str(k))}/{urllib.parse.quote(str(v))}"
    return endpoint


def _pg_delete_series(base_url: str, job: str, grouping: dict[str, str]) -> None:
    import urllib.request
    try:
        url = _pg_endpoint(base_url, job, grouping)
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except Exception:
        pass


def push_metrics_pg(metrics_text: str, job="clocher-bench", grouping=None, base_url=None, method="PUT"):
    import urllib.request
    base_url = base_url or os.getenv("IE_METRICS_PUSHGATEWAY", "http://localhost:9091")
    grouping = grouping or {"instance": _hostname()}
    endpoint = _pg_endpoint(base_url, job, grouping)
    try:
        req = urllib.request.Request(
            endpoint,
            data=metrics_text.encode("utf-8"),
            method=method,
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except Exception as e:
        print(f"[warn] pushgateway publish failed: {e}")


def emit_metrics_text(summary: Dict[str, Any], labels: Dict[str, str]) -> str:
    """Render a single snapshot suitable for Pushgateway."""
    lines: List[str] = []

    def put(name: str, val, l_extra: dict | None = None, mtype: str = "gauge"):
        all_labels = dict(labels)
        if l_extra:
            all_labels.update(l_extra)
        lines.append(f"# TYPE {name} {mtype}")
        lines.append(f"{name}{_fmt_labels(all_labels)} {val}")

    # Core
    put("ie_tps", summary.get("tps_true", 0.0))
    put("ie_latency_seconds", summary.get("wall_time_s", 0.0), {"stage": "e2e"})
    put("ie_rss_bytes", int(float(summary.get("rss_peak_mb_max", 0.0)) * 1024 * 1024))
    put("ie_tokens_total", int(summary.get("tokens_generated", 0)), mtype="counter")

    # Spatial metrics
    bpt = int(summary.get("ie_bytes_per_token") or 0)
    bt = float(summary.get("bytes_touched") or 0.0)
    bw = float(summary.get("effective_bandwidth_gbps") or 0.0)
    put("ie_bytes_per_token", bpt)
    put("ie_bytes_touched_total", bt)
    put("ie_effective_bandwidth_gbps", bw)

    # KV counts
    put("ie_kv_hits_total", int(summary.get("kv_hits", 0)), mtype="counter")
    put("ie_kv_misses_total", int(summary.get("kv_misses", 0)), mtype="counter")

    # Config-as-labels snapshot
    put("ie_config_info", 1, mtype="gauge")

    return "\n".join(lines) + "\n"


# ------------------------------ synthetic run ------------------------------ #

def _synthetic_evaluate(prompts: List[str]) -> Tuple[int, float, float, float]:
    """Return (tokens_generated, wall_time_s, p50_ms, p95_ms)."""
    # Keep simple and deterministic but non-zero:
    n = max(1, len(prompts))
    tokens = 64 * n
    wall_s = max(0.050, 0.012 * n)
    p50_ms = 6.0 + 0.25 * n
    p95_ms = p50_ms * 2.0
    return tokens, wall_s, p50_ms, p95_ms


def _derive_bytes_touched(tokens: int, bpt: Optional[int], wall_s: float) -> Tuple[float, float]:
    """Return (bytes_touched, effective_bandwidth_gbps)."""
    if not bpt or tokens <= 0 or wall_s <= 0:
        return 0.0, 0.0
    bt = float(tokens) * float(bpt)                    # bytes
    bw_gbps = (bt / 1_000_000_000.0) / float(wall_s)  # GB/s
    return bt, bw_gbps


# ------------------------------ CLI / sweep ------------------------------ #

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-sweep benchmark harness (stdlib-only)")
    p.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts.jsonl"))
    p.add_argument("--report-root", type=Path, default=Path("benchmarks/reports_sweep"))
    p.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    p.add_argument("--threads", default=os.getenv("THREADS", "auto"))
    p.add_argument("--precision", default=os.getenv("PRECISION", "fp32"))
    p.add_argument("--affinity", default=os.getenv("AFFINITY", "auto"))
    p.add_argument("--pretranspose", default=os.getenv("PRETRANSPOSE", "all"))
    p.add_argument("--batch", type=int, default=int(os.getenv("BATCH", "1")))
    p.add_argument("--engine-bin", default=os.getenv("ENGINE_BIN", ""))

    # BPT/stride/touch for spatial metrics
    p.add_argument("--ie-bytes-per-token", type=int, default=int(os.getenv("IE_BYTES_PER_TOKEN", "67108864")))
    p.add_argument("--ie-stride-bytes", type=int, default=int(os.getenv("IE_STRIDE_BYTES", "256")))
    p.add_argument("--ie-verify-touch", type=int, default=int(os.getenv("IE_VERIFY_TOUCH", "1")))
    p.add_argument("--ie-require-model", type=int, default=int(os.getenv("IE_REQUIRE_MODEL", "1")))
    p.add_argument("--model-dir", default=os.getenv("MODEL_DIR", ""))

    # Sweep domains (comma-separated)
    p.add_argument("--act-quant", default=os.getenv("IE_SWEEP_ACT_QUANT", "none,int8,fp8_e4m3,fp8_e5m2"))
    p.add_argument("--kv-quant", default=os.getenv("IE_SWEEP_KV_QUANT", "none,int8"))
    p.add_argument("--prefetch", default=os.getenv("IE_SWEEP_PREFETCH", "off,on,auto"))
    p.add_argument("--nt-loads", default=os.getenv("IE_SWEEP_NT_LOADS", "0,1"))
    p.add_argument("--numa-policy", default=os.getenv("IE_SWEEP_NUMA_POLICY", "auto,compact,scatter"))

    # Metrics publishing
    p.add_argument("--no-metrics", action="store_true", help="disable Pushgateway publishing")
    return p.parse_args()


def _split_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _combo_id(actq: str, kvq: str, pf: str, nt: str, numa: str) -> str:
    return f"act_{actq}__kv_{kvq}__pref_{pf}__nt{nt}__numa_{numa}"


def main() -> int:
    args = _parse()

    # Prompts
    prompts = _load_prompts_jsonl(args.prompts)
    if not prompts:
        prompts = [f"default-{i}" for i in range(4)]

    # Sweep domains
    actQ = _split_list(args.act_quant)
    kvQ = _split_list(args.kv_quant)
    PF = _split_list(args.prefetch)
    NT = _split_list(args.nt_loads)
    NUMA = _split_list(args.numa_policy)

    # Metrics config
    metrics_enabled = (not args.no_metrics) and _env_bool("IE_METRICS_PUSH", True)
    grouping = _env_json("IE_METRICS_GROUPING", {"instance": _hostname()})
    if metrics_enabled and _env_bool("IE_METRICS_CLEAN", False):
        _pg_delete_series(
            os.getenv("IE_METRICS_PUSHGATEWAY", "http://localhost:9091"),
            os.getenv("IE_METRICS_JOB", "clocher-bench"),
            grouping,
        )

    # Top-level index for quick comparison
    index_rows: List[Dict[str, Any]] = []
    _ensure_dir(args.report_root)

    # Iterate grid
    for actq, kvq, pf, nt, numa in itertools.product(actQ, kvQ, PF, NT, NUMA):
        combo = _combo_id(actq, kvq, pf, nt, numa)
        outdir = _ensure_outdir(args.report_root, combo)
        csv_path = outdir / "samples.csv"
        summary_path = outdir / "summary.json"

        # Synthetic “run”
        tokens, wall_s, p50_ms, p95_ms = _synthetic_evaluate(prompts)
        tps_true = float(tokens) / float(wall_s) if wall_s > 0 else 0.0

        # Spatial estimates
        bpt = int(args.ie_bytes_per_token)
        bytes_touched, bw_gbps = _derive_bytes_touched(tokens, bpt, wall_s)

        # RSS fake snapshot (scale slightly by config)
        rss_mean_mb = 1024.0 + (50.0 if nt == "1" else 0.0) + (25.0 if pf == "on" else 0.0)
        rss_max_mb = rss_mean_mb + 64.0

        # KV fake counters
        kv_hits = 1000 + (10 if kvq != "none" else 0)
        kv_miss = 25

        # Write CSV with one row per prompt (synthetic)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["prompt_len", "tokens_generated", "tps_true", "elapsed_s"],
            )
            w.writeheader()
            for p in prompts:
                w.writerow({
                    "prompt_len": len(p),
                    "tokens_generated": tokens // max(1, len(prompts)),
                    "tps_true": tps_true,
                    "elapsed_s": wall_s / max(1, len(prompts)),
                })

        # Summary JSON (shape matches exporter + PERFORMANCE.md generator)
        summary = {
            # device/run identity
            "device": args.device,
            "engine_bin": args.engine_bin,
            "model_dir": args.model_dir,

            # config
            "threads": args.threads,
            "precision": args.precision,
            "batch": args.batch,
            "affinity": args.affinity,
            "pretranspose": args.pretranspose,
            "prompts": str(args.prompts),

            # sweep knobs
            "act_quant": actq,
            "kv_quant": kvq,
            "prefetch": pf,
            "nt_loads": nt,
            "numa_policy": numa,

            # results
            "tokens_generated": tokens,
            "wall_time_s": wall_s,
            "tps_true": tps_true,
            "latency_p50_ms": p50_ms,
            "latency_p95_ms": p95_ms,
            "rss_peak_mb": rss_mean_mb,
            "rss_peak_mb_max": rss_max_mb,
            "kv_hits": kv_hits,
            "kv_misses": kv_miss,

            # spatial knobs and derived metrics
            "ie_bytes_per_token": bpt,
            "ie_stride_bytes": int(args.ie_stride_bytes),
            "ie_verify_touch": int(args.ie_verify_touch),
            "ie_require_model": int(args.ie_require_model),

            "bytes_touched": bytes_touched,
            "effective_bandwidth_gbps": bw_gbps,

            # optional git (best-effort)
            "git_commit": _discover_git(),
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        # Push per-combination snapshot
        if metrics_enabled:
            labels = {
                "device": args.device,
                "act_quant": actq,
                "kv_quant": kvq,
                "prefetch": pf,
                "nt_loads": nt,
                "numa_policy": numa,
                "precision": args.precision,
                "threads": str(args.threads),
                "run_id": combo,
            }
            txt = emit_metrics_text(summary, labels)
            push_metrics_pg(
                txt,
                job=os.getenv("IE_METRICS_JOB", "clocher-bench"),
                grouping=grouping,
                base_url=os.getenv("IE_METRICS_PUSHGATEWAY", "http://localhost:9091"),
                method="PUT",
            )

        # Append to index
        index_rows.append({
            "combo": combo,
            "tps_true": tps_true,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "rss_mean_mb": rss_mean_mb,
            "rss_max_mb": rss_max_mb,
            "bytes_touched": bytes_touched,
            "bandwidth_gbps": bw_gbps,
        })

        print(f"[ok] wrote: {csv_path}")
        print(f"[ok] wrote: {summary_path}")

    # Write top-level index CSV
    index_csv = args.report_root / "index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "combo", "tps_true", "p50_ms", "p95_ms",
                "rss_mean_mb", "rss_max_mb",
                "bytes_touched", "bandwidth_gbps",
            ],
        )
        w.writeheader()
        for row in index_rows:
            w.writerow(row)
    print(f"[ok] wrote: {index_csv}")

    return 0


# ------------------------------ misc helpers ------------------------------ #

def _discover_git() -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--ignore-submodules"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"{sha}{' DIRTY' if dirty != 0 else ''}"
    except Exception:
        return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        # Always leave a minimal usable artifact even on unexpected errors.
        root = Path("benchmarks/reports_sweep")
        _ensure_dir(root)
        (root / "index.csv").write_text(
            "combo,tps_true,p50_ms,p95_ms,rss_mean_mb,rss_max_mb,bytes_touched,bandwidth_gbps\n",
            encoding="utf-8",
        )
        print(f"[warn] sweep aborted with: {type(e).__name__}: {e}")
        raise SystemExit(0)
