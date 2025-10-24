# =============================================================================
# File: scripts/metrics_exporter.py
# =============================================================================
#!/usr/bin/env python3
"""
Minimal Prometheus text exporter for engine benchmark results.

Two modes:
  1) --summary FILE
     Serve metrics based on a single JSON summary object produced by
     scripts/true_tps_strict.sh (the final block it prints).

  2) --dir PATH
     Serve metrics computed from the most recent *.json file in PATH where each
     file is a single-run JSON line (also printed by true_tps_strict.sh).
     If both --dir and --summary are specified, --summary wins.

The server exposes /metrics on the given port and refreshes metrics on demand.

Examples:
  python3 scripts/metrics_exporter.py --summary /tmp/final_summary.json
  python3 scripts/metrics_exporter.py --dir /tmp/runs --port 9109
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def _find_latest_json(path: Path) -> Optional[Path]:
  """Return the newest *.json file under path, or None."""
  files = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
  return files[0] if files else None


def _load_one_json_file(path: Path) -> Dict[str, Any]:
  """Load a JSON object from file, return {} on error."""
  try:
    with open(path, "r", encoding="utf-8") as f:
      txt = f.read().strip()
      # handle either JSON object or a one-line object with trailing newline
      return json.loads(txt)
  except Exception:
    return {}


def _gauge_line(name: str, value: Any, labels: Dict[str, str] | None = None) -> str:
  """Render a Prometheus gauge sample line."""
  if value is None:
    return ""
  if labels:
    lab = ",".join([f'{k}="{v}"' for (k, v) in labels.items()])
    return f"{name}{{{lab}}} {value}\n"
  return f"{name} {value}\n"


def _help_type(name: str, help_text: str) -> str:
  return f"# HELP {name} {help_text}\n# TYPE {name} gauge\n"


def collect_metrics_from_summary(obj: Dict[str, Any]) -> str:
  """Convert a summary JSON object to Prometheus text."""
  host = socket.gethostname()
  labels = {
    "host": host,
    "model_dir": str(obj.get("model_dir", "")),
    "precision": str(obj.get("precision", "")),
    "affinity": str(obj.get("affinity", "")),
    "pretranspose": str(obj.get("pretranspose", "")),
    "batch": str(obj.get("batch", "")),
  }

  out = []
  out.append(_help_type("ie_tokens_generated_total", "Total tokens generated across runs"))
  out.append(_gauge_line("ie_tokens_generated_total", obj.get("tokens_generated", 0), labels))

  out.append(_help_type("ie_wall_time_seconds", "Total wall time across runs (s)"))
  out.append(_gauge_line("ie_wall_time_seconds", obj.get("wall_time_s", 0), labels))

  out.append(_help_type("ie_tps_true", "True tokens-per-second across runs"))
  out.append(_gauge_line("ie_tps_true", obj.get("tps_true", 0), labels))

  out.append(_help_type("ie_latency_p50_ms", "Mean p50 latency (ms)"))
  out.append(_gauge_line("ie_latency_p50_ms", obj.get("latency_p50_ms", 0), labels))

  out.append(_help_type("ie_latency_p95_ms", "Mean p95 latency (ms)"))
  out.append(_gauge_line("ie_latency_p95_ms", obj.get("latency_p95_ms", 0), labels))

  out.append(_help_type("ie_rss_peak_mb_mean", "Mean peak RSS (MB) across runs"))
  out.append(_gauge_line("ie_rss_peak_mb_mean", obj.get("rss_peak_mb", 0), labels))

  out.append(_help_type("ie_rss_peak_mb_max", "Max peak RSS (MB) across runs"))
  out.append(_gauge_line("ie_rss_peak_mb_max", obj.get("rss_peak_mb_max", 0), labels))

  out.append(_help_type("ie_kv_hits_total", "Total KV cache hits during runs"))
  out.append(_gauge_line("ie_kv_hits_total", obj.get("kv_hits", 0), labels))

  out.append(_help_type("ie_kv_misses_total", "Total KV cache misses during runs"))
  out.append(_gauge_line("ie_kv_misses_total", obj.get("kv_misses", 0), labels))

  # Environment-driven “build info”
  build_labels = {
    "host": host,
    "threads": str(obj.get("threads", "")),
    "max_new": str(obj.get("max_new", "")),
    "prompts": str(obj.get("prompts", "")),
    "ie_bytes_per_token": str(obj.get("ie_bytes_per_token", "")),
    "ie_stride_bytes": str(obj.get("ie_stride_bytes", "")),
    "ie_require_model": str(obj.get("ie_require_model", "")),
    "ie_verify_touch": str(obj.get("ie_verify_touch", "")),
  }
  out.append("# HELP ie_build_info Build/runtime knobs as labels\n# TYPE ie_build_info gauge\n")
  out.append(_gauge_line("ie_build_info", 1, build_labels))

  return "".join(out)


class Handler(BaseHTTPRequestHandler):
  """HTTP handler serving /metrics from either a summary file or dir."""

  # These are set by the server factory below.
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
