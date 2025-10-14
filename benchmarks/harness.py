#!/usr/bin/env python3
"""
Stdlib-only benchmark harness for the Inference Engine (baseline).

- Reads benchmarks/prompts.jsonl (ignores blank lines and lines starting with '#').
- Runs the compiled CLI for each prompt and captures a JSON line of metrics.
- Writes CSV and JSON summary under benchmarks/reports/<UTC timestamp>/.

This script is dependency-free and portable across Python 3.8+.
"""

from __future__ import annotations
import csv
import json
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "inference-engine"
PROMPTS = ROOT / "benchmarks" / "prompts.jsonl"
REPORTS_DIR = ROOT / "benchmarks" / "reports"


def _load_prompts(path: Path) -> list[str]:
    """
    Load prompts from a JSONL file, skipping blank/comment lines.

    Each non-empty line must be a JSON object with a "text" field.
    """
    if not path.exists():
        raise FileNotFoundError(f"prompts file not found: {path}")
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON line in {path}: {line!r}") from e
            if "text" not in obj or not isinstance(obj["text"], str):
                raise ValueError(f"Missing 'text' field in JSON line: {line!r}")
            prompts.append(obj["text"])
    if not prompts:
        raise ValueError(f"No valid prompts found in {path}")
    return prompts


def main() -> int:
    # Preconditions
    if not BIN.exists():
        print(f"[error] binary not found: {BIN} (run: make build)", file=sys.stderr)
        return 1

    try:
        prompts = _load_prompts(PROMPTS)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    # Output directory (timezone-aware UTC)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "samples.csv"
    summary_path = out_dir / "summary.json"

    rows: list[dict] = []

    # Run once per prompt (baseline)
    for p in prompts:
        t0 = time.time()
        try:
            cp = subprocess.run(
                [str(BIN), p],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print("[error] engine run failed", file=sys.stderr)
            print(f"command: {e.cmd}", file=sys.stderr)
            print(f"stdout: {e.stdout}", file=sys.stderr)
            print(f"stderr: {e.stderr}", file=sys.stderr)
            return 1
        t1 = time.time()

        line = cp.stdout.strip()
        try:
            m = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[error] engine did not output JSON: {line!r}", file=sys.stderr)
            return 1

        m["elapsed_s"] = t1 - t0
        m["prompt_len"] = len(p)
        rows.append(m)

    # Write CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Summary
    avg_tps = sum(r.get("tps_true", 0.0) for r in rows) / len(rows)
    total_tokens = sum(int(r.get("tokens_generated", 0)) for r in rows)
    summary = {"avg_tps_true": avg_tps, "total_tokens": total_tokens, "samples": len(rows)}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    print(f"[ok] wrote: {csv_path}")
    print(f"[ok] wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
