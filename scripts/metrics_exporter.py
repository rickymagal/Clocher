#!/usr/bin/env python3
# =============================================================================
# Minimal Prometheus text exporter for engine benchmark results.
#
# Two modes:
#   1) --summary FILE
#        Serve metrics based on a single JSON summary object produced by
#        scripts/true_tps_strict.sh (the final block it prints).
#
#   2) --dir PATH
#        Serve metrics computed from the most recent *.json file in PATH where
#        each file is a single-run JSON line (also printed by true_tps_strict.sh).
#        If both --dir and --summary are specified, --summary wins.
#
# This version also surfaces memory-related metrics and configuration knobs:
# - bytes touched, effective bandwidth, bytes-per-token (BPT)
# - activation quantization (act_quant), kv quantization (kv_quant)
# - prefetch controls, non-temporal loads, NUMA policy
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Any, Optional


# ------------------------------ FS helpers ------------------------------ #

def _find_latest_json(path: Path) -> Optional[Path]:
    files = sorted(
        path.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _load_one_json_file(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return json.loads(txt)
    except Exception:
        return {}


# --------------------------- Prom text helpers --------------------------- #

def _gauge_line(name: str, value: Any, labels: Dict[str, str] | None = None) -> str:
    if value is None:
        return ""
    if labels:
        lab = ",".join([f'{k}="{v}"' for (k, v) in labels.items()])
        return f"{name}{{{lab}}} {value}\n"
    return f"{name} {value}\n"


def _help_type(name: str, help_text: str) -> str:
    return f"# HELP {name} {help_text}\n# TYPE {name} gauge\n"


# ------------------------------ Collector ------------------------------ #

def _label_str(x: Any) -> str:
    # Ensure label values are strings and safe
    s = "" if x is None else str(x)
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def collect_metrics_from_summary(obj: Dict[str, Any]) -> str:
    """Convert a summary JSON object to Prometheus text."""
    host = socket.gethostname()

    # Core labels (stable)
    base_labels = {
        "host": host,
        "device": _label_str(obj.get("device")),            # "cpu" | "gpu" | etc.
        "model_dir": _label_str(obj.get("model_dir")),
        "precision": _label_str(obj.get("precision")),
        "affinity": _label_str(obj.get("affinity")),
        "pretranspose": _label_str(obj.get("pretranspose")),
        "batch": _label_str(obj.get("batch")),
        "threads": _label_str(obj.get("threads")),
    }

    # Config/knobs as labels-only info gauges
    cfg_labels = {
        "host": host,
        "act_quant": _label_str(obj.get("act_quant")),                # "none"|"int8"|"fp8_e4m3"|...
        "kv_quant": _label_str(obj.get("kv_quant")),                  # "none"|"int8"|...
        "prefetch": _label_str(obj.get("prefetch")),                  # "off"|"on"|"auto"|N
        "nt_loads": _label_str(obj.get("nt_loads")),                  # 0|1
        "numa_policy": _label_str(obj.get("numa_policy")),            # "auto"|"compact"|"scatter"|...
        "ie_bytes_per_token": _label_str(obj.get("ie_bytes_per_token")),
        "ie_stride_bytes": _label_str(obj.get("ie_stride_bytes")),
        "ie_require_model": _label_str(obj.get("ie_require_model")),
        "ie_verify_touch": _label_str(obj.get("ie_verify_touch")),
        "prompts": _label_str(obj.get("prompts")),
        "engine_bin": _label_str(obj.get("engine_bin")),
        "git_commit": _label_str(obj.get("git_commit")),
    }

    # Numeric metrics
    tokens_generated = obj.get("tokens_generated") or obj.get("tokens_sum") or 0
    wall_time_s = obj.get("wall_time_s") or obj.get("wall_sum") or 0.0
    tps_true = obj.get("tps_true") or 0.0
    p50_ms = obj.get("latency_p50_ms") or obj.get("p50_mean") or 0.0
    p95_ms = obj.get("latency_p95_ms") or obj.get("p95_mean") or 0.0
    rss_mean_mb = obj.get("rss_peak_mb") or obj.get("rss_mean") or 0.0
    rss_max_mb = obj.get("rss_peak_mb_max") or obj.get("rss_max") or 0.0
    kv_hits = obj.get("kv_hits") or 0
    kv_misses = obj.get("kv_misses") or 0

    # Memory/IO related (optional in summary)
    bytes_per_token = obj.get("ie_bytes_per_token")
    bytes_touched = obj.get("bytes_touched")  # may be bytes or GB depending on source
    if isinstance(bytes_touched, str):
        try:
            bytes_touched = float(bytes_touched)
        except Exception:
            bytes_touched = None
    eff_bw_gbps = obj.get("effective_bandwidth_gbps")

    # If not provided, attempt a naive derivation
    if bytes_touched is None and bytes_per_token and tokens_generated:
        try:
            bytes_touched = float(bytes_per_token) * float(tokens_generated)
        except Exception:
            bytes_touched = None

    out = []

    # Core totals
    out.append(_help_type("ie_tokens_generated_total", "Total tokens generated across runs"))
    out.append(_gauge_line("ie_tokens_generated_total", tokens_generated, base_labels))

    out.append(_help_type("ie_wall_time_seconds", "Total wall time across runs (s)"))
    out.append(_gauge_line("ie_wall_time_seconds", wall_time_s, base_labels))

    out.append(_help_type("ie_tps_true", "True tokens-per-second across runs"))
    out.append(_gauge_line("ie_tps_true", tps_true, base_labels))

    # Latency
    out.append(_help_type("ie_latency_p50_ms", "Mean p50 latency (ms)"))
    out.append(_gauge_line("ie_latency_p50_ms", p50_ms, base_labels))

    out.append(_help_type("ie_latency_p95_ms", "Mean p95 latency (ms)"))
    out.append(_gauge_line("ie_latency_p95_ms", p95_ms, base_labels))

    # Memory
    out.append(_help_type("ie_rss_peak_mb_mean", "Mean peak RSS (MB) across runs"))
    out.append(_gauge_line("ie_rss_peak_mb_mean", rss_mean_mb, base_labels))

    out.append(_help_type("ie_rss_peak_mb_max", "Max peak RSS (MB) across runs"))
    out.append(_gauge_line("ie_rss_peak_mb_max", rss_max_mb, base_labels))

    # KV cache
    out.append(_help_type("ie_kv_hits_total", "Total KV cache hits during runs"))
    out.append(_gauge_line("ie_kv_hits_total", kv_hits, base_labels))

    out.append(_help_type("ie_kv_misses_total", "Total KV cache misses during runs"))
    out.append(_gauge_line("ie_kv_misses_total", kv_misses, base_labels))

    # Spatial metrics
    out.append(_help_type("ie_bytes_per_token", "Model bytes processed per generated token (bytes/token)"))
    out.append(_gauge_line("ie_bytes_per_token", bytes_per_token or 0, base_labels))

    out.append(_help_type("ie_bytes_touched_total", "Total bytes touched across runs"))
    out.append(_gauge_line("ie_bytes_touched_total", bytes_touched or 0, base_labels))

    out.append(_help_type("ie_effective_bandwidth_gbps", "Effective aggregate bandwidth (GB/s)"))
    out.append(_gauge_line("ie_effective_bandwidth_gbps", eff_bw_gbps or 0, base_labels))

    # Build/config info as labels
    out.append("# HELP ie_config_info Build/runtime knobs as labels\n# TYPE ie_config_info gauge\n")
    out.append(_gauge_line("ie_config_info", 1, cfg_labels))

    return "".join(out)


# ------------------------------- HTTP server ------------------------------- #

class Handler(BaseHTTPRequestHandler):
    """HTTP handler serving /metrics from either a summary file or dir."""

    summary_path: Optional[Path] = None
    dir_path: Optional[Path] = None

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found\n")
                return

            obj: Dict[str, Any] = {}
            if self.summary_path is not None:
                obj = _load_one_json_file(self.summary_path)
            elif self.dir_path is not None:
                latest = _find_latest_json(self.dir_path)
                if latest is not None:
                    obj = _load_one_json_file(latest)

            txt = collect_metrics_from_summary(obj if obj else {})
            payload = txt.encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            err = f"# exporter_error{{msg=\"{type(e).__name__}\"}} 1\n"
            payload = err.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def make_server(addr: str, port: int, summary: Optional[Path], directory: Optional[Path]) -> HTTPServer:
    """Create an HTTPServer serving metrics from the given source."""
    class _Factory(Handler):
        summary_path = summary
        dir_path = directory
    return HTTPServer((addr, port), _Factory)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=9109, help="Port (default: 9109)")
    ap.add_argument("--summary", type=str, help="Path to final summary JSON file")
    ap.add_argument("--dir", type=str, help="Directory of per-run JSON files")
    args = ap.parse_args()

    summary = Path(args.summary) if args.summary else None
    directory = Path(args.dir) if args.dir else None
    if not summary and not directory:
        raise SystemExit("ERROR: specify --summary FILE or --dir DIR")

    if summary and not summary.is_file():
        raise SystemExit(f"ERROR: summary file not found: {summary}")
    if directory and not directory.is_dir():
        raise SystemExit(f"ERROR: directory not found: {directory}")

    srv = make_server(args.addr, args.port, summary, directory)
    print(f"[metrics_exporter] serving on http://{args.addr}:{args.port}/metrics", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
