#!/usr/bin/env python3
"""
Minimal Prometheus exporter for Clocher metrics (stdlib-only).

It serves /metrics based on the latest benchmarks/reports/<TS>/summary.json.
Intended to be scraped by Prometheus; a docker-compose is provided.
"""

import argparse
import glob
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Dict, Any


def latest_summary_path() -> str:
    """
    Return the path to the latest summary.json under benchmarks/reports, or "".

    Returns:
        Path string or empty string when nothing is found.
    """
    roots = sorted(glob.glob("benchmarks/reports/*"))
    if not roots:
        return ""
    latest = roots[-1]
    path = os.path.join(latest, "summary.json")
    return path if os.path.exists(path) else ""


def load_metrics() -> Optional[Dict[str, Any]]:
    """
    Load and parse the latest summary.json as a dictionary.

    Returns:
        Parsed JSON dictionary or None on error/missing file.
    """
    p = latest_summary_path()
    if not p:
        return None
    try:
        with open(p, "r") as f:
            js = json.load(f)
        return js
    except Exception:
        return None


def format_prometheus(js: Dict[str, Any]) -> str:
    """
    Convert a summary.json dictionary into Prometheus exposition format.

    Args:
        js: Summary dictionary.

    Returns:
        A text string in Prometheus exposition format (ending with newline).
    """
    lines = []
    tps = js.get("tps_true", 0.0)
    p50 = js.get("latency_p50_ms", 0.0)
    p95 = js.get("latency_p95_ms", 0.0)
    rss = js.get("rss_peak_mb", 0)
    kvh = js.get("kv_hits", 0)
    kvm = js.get("kv_misses", 0)
    dev = js.get("device", "cpu")
    prec = js.get("precision", "fp32")
    pretr = js.get("pretranspose", "none")

    lines.append("# HELP clocher_tps_true Tokens per second (true wall).")
    lines.append("# TYPE clocher_tps_true gauge")
    lines.append(f'clocher_tps_true{{device="{dev}",precision="{prec}",pretranspose="{pretr}"}} {tps}')

    lines.append("# HELP clocher_latency_p50_ms Median token latency in ms.")
    lines.append("# TYPE clocher_latency_p50_ms gauge")
    lines.append(f'clocher_latency_p50_ms{{device="{dev}",precision="{prec}",pretranspose="{pretr}"}} {p50}')

    lines.append("# HELP clocher_latency_p95_ms P95 token latency in ms.")
    lines.append("# TYPE clocher_latency_p95_ms gauge")
    lines.append(f'clocher_latency_p95_ms{{device="{dev}",precision="{prec}",pretranspose="{pretr}"}} {p95}')

    lines.append("# HELP clocher_rss_peak_mb Peak RSS in MB.")
    lines.append("# TYPE clocher_rss_peak_mb gauge")
    lines.append(f'clocher_rss_peak_mb {rss}')

    lines.append("# HELP clocher_kv_hits KV cache hits.")
    lines.append("# TYPE clocher_kv_hits counter")
    lines.append(f'clocher_kv_hits {kvh}')

    lines.append("# HELP clocher_kv_misses KV cache misses.")
    lines.append("# TYPE clocher_kv_misses counter")
    lines.append(f'clocher_kv_misses {kvm}')

    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    """
    HTTP request handler that serves /metrics from the latest summary.json.
    """

    def do_GET(self) -> None:
        """
        Handle GET requests; returns Prometheus metrics or 404 for other paths.
        """
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        js = load_metrics()
        body = format_prometheus(js) if js else "# no data\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))


def main() -> None:
    """
    Start the HTTP server that exposes /metrics.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[exporter] listening on :{args.port} (GET /metrics)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
