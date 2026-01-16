#!/usr/bin/env python3
"""
Strict harness wrapper for Clocher inference-engine.

This script intentionally does *not* require any engine-side flags like --runs or --report-json.
It runs the existing strict harness (scripts/true_tps_strict.sh) by setting the expected env vars,
captures its JSONL output, and writes:

  - the full JSONL log (per-run + summary) to --out-jsonl
  - a summary.json (the final JSON object) to benchmarks/reports/<run_id>/summary.json

It also prints the summary.json path (one line) to keep compatibility with simple CI checks.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, Optional


def _repo_root_from_here() -> pathlib.Path:
    # Prefer CWD if it looks like the repo root; otherwise fall back to script location parents.
    cwd = pathlib.Path.cwd()
    if (cwd / "scripts" / "true_tps_strict.sh").is_file():
        return cwd
    here = pathlib.Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "scripts" / "true_tps_strict.sh").is_file():
            return p
    return cwd


def _last_json_object_from_jsonl(text: str) -> Dict[str, Any]:
    last: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            last = obj
    if last is None:
        raise RuntimeError("strict harness produced no JSON objects on stdout")
    return last


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--engine-bin", default=os.environ.get("ENGINE_BIN", "./build/inference-engine"))
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), choices=["cpu", "cuda", "ze"])
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "./models/gpt-oss-20b"))
    ap.add_argument("--prompts", dest="prompts_file", default=os.environ.get("PROMPTS", "./benchmarks/prompts_10.txt"))

    ap.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "1")))
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "1")))
    ap.add_argument("--threads", default=os.environ.get("THREADS", "0"))
    ap.add_argument("--precision", default=os.environ.get("PRECISION", "fp32"))
    ap.add_argument("--batch", default=os.environ.get("BATCH", "1"))
    ap.add_argument("--prefetch", default=os.environ.get("PREFETCH", "auto"))
    ap.add_argument("--pretranspose", default=os.environ.get("PRETRANSPOSE", "all"))
    ap.add_argument("--affinity", default=os.environ.get("AFFINITY", "auto"))
    ap.add_argument("--max-new", type=int, default=int(os.environ.get("MAX_NEW", "128")))

    ap.add_argument("--ie-bytes-per-token", default=os.environ.get("IE_BYTES_PER_TOKEN", "67108864"))
    ap.add_argument("--ie-stride-bytes", default=os.environ.get("IE_STRIDE_BYTES", "256"))
    ap.add_argument("--ie-verify-touch", default=os.environ.get("IE_VERIFY_TOUCH", "1"))

    ap.add_argument("--ie-dedup", default=os.environ.get("IE_DEDUP", "1"))
    ap.add_argument("--ie-dedup-policy", default=os.environ.get("IE_DEDUP_POLICY", ""))
    ap.add_argument("--ie-dedup-cache-mb", default=os.environ.get("IE_DEDUP_CACHE_MB", ""))
    ap.add_argument("--ie-dedup-hot-bytes", default=os.environ.get("IE_DEDUP_HOT_BYTES", ""))
    ap.add_argument("--ie-dedup-hot-list", default=os.environ.get("IE_DEDUP_HOT_LIST", ""))

    ap.add_argument("--strict-sh", default=os.environ.get("STRICT_SH", "scripts/true_tps_strict.sh"))
    ap.add_argument("--out-jsonl", default=os.environ.get("OUT_JSONL", "build/strict_out.jsonl"))
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))

    args = ap.parse_args()

    repo = _repo_root_from_here()
    strict_sh = (repo / args.strict_sh).resolve()
    if not strict_sh.is_file():
        raise SystemExit(f"error: strict script not found: {strict_sh}")

    engine_bin = pathlib.Path(args.engine_bin).expanduser()
    if not engine_bin.is_absolute():
        engine_bin = (repo / engine_bin).resolve()
    if not engine_bin.is_file():
        raise SystemExit(f"error: engine binary not found: {engine_bin}")

    model_dir = pathlib.Path(args.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = (repo / model_dir).resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"error: model dir not found: {model_dir}")

    prompts = pathlib.Path(args.prompts_file).expanduser()
    if not prompts.is_absolute():
        prompts = (repo / prompts).resolve()
    if not prompts.is_file():
        raise SystemExit(f"error: prompts file not found: {prompts}")

    run_id = args.run_id.strip() or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = pathlib.Path(args.out_jsonl).expanduser()
    if not out_jsonl.is_absolute():
        out_jsonl = (repo / out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    report_dir = repo / "benchmarks" / "reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "summary.json"

    env = os.environ.copy()
    env.update({
        "ENGINE_BIN": str(engine_bin),
        "DEVICE": args.device,
        "MODEL_DIR": str(model_dir),
        "PROMPTS": str(prompts),
        "RUNS": str(int(args.runs)),
        "ROUNDS": str(int(args.rounds)),
        "THREADS": str(args.threads),
        "PRECISION": str(args.precision),
        "BATCH": str(args.batch),
        "PREFETCH": str(args.prefetch),
        "PRETRANSPOSE": str(args.pretranspose),
        "AFFINITY": str(args.affinity),
        "MAX_NEW": str(int(args.max_new)),
        "IE_BYTES_PER_TOKEN": str(args.ie_bytes_per_token),
        "IE_STRIDE_BYTES": str(args.ie_stride_bytes),
        "IE_VERIFY_TOUCH": str(args.ie_verify_touch),
        "IE_DEDUP": str(args.ie_dedup),
    })

    if args.ie_dedup_policy:
        env["IE_DEDUP_POLICY"] = args.ie_dedup_policy
    if args.ie_dedup_cache_mb:
        env["IE_DEDUP_CACHE_MB"] = args.ie_dedup_cache_mb
    if args.ie_dedup_hot_bytes:
        env["IE_DEDUP_HOT_BYTES"] = args.ie_dedup_hot_bytes
    if args.ie_dedup_hot_list:
        env["IE_DEDUP_HOT_LIST"] = args.ie_dedup_hot_list

    # Run strict harness and capture stdout (JSONL).
    proc = subprocess.run(
        [str(strict_sh)],
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    out_jsonl.write_text(proc.stdout)
    summary = _last_json_object_from_jsonl(proc.stdout)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # One line for CI / quick checks.
    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
