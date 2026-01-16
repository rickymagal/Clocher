#!/usr/bin/env python3
"""
run_harness_report.py

Run the strict harness (scripts/true_tps_strict.sh) and convert its JSONL output
into a normalized per-prompt report artifact.

It also supports a pure conversion mode:
  - stdin -> stdout
  - IN -> OUT
  - --in / --out
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root_from_here() -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    if (cwd / "scripts" / "true_tps_strict.sh").is_file():
        return cwd
    here = pathlib.Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "scripts" / "true_tps_strict.sh").is_file():
            return p
    return cwd


def _read_all_text(path: Optional[str]) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_all_text(path: Optional[str], text: str) -> None:
    if path is None or path == "-":
        sys.stdout.write(text)
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _parse_jsonl_objects(text: str) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    if objs:
        return objs
    # Fallback: try a single JSON object.
    text_stripped = text.lstrip()
    if text_stripped.startswith("{"):
        try:
            obj = json.loads(text_stripped)
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            pass
    raise ValueError("no JSON objects found in input")


def _select_runs_and_meta(objs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    runs = [o for o in objs if isinstance(o.get("prompts"), list)]
    meta = {}
    for o in reversed(objs):
        if not isinstance(o.get("prompts"), list):
            meta = o
            break
    if not meta and runs:
        meta = runs[-1]
    return runs, meta


def _pick_run_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "timestamp",
        "threads",
        "precision",
        "model_dir",
        "prompts",
        "runs",
        "rounds",
        "device",
        "engine_bin",
        "batch",
        "max_new",
        "affinity",
        "pretranspose",
        "prefetch",
        "ie_require_model",
        "ie_bytes_per_token",
        "ie_stride_bytes",
        "ie_verify_touch",
        "ie_dedup",
        "ie_dedup_strict",
        "ie_dedup_policy",
        "ie_dedup_cache_mb",
        "model_ie_bin_bytes",
        "dedup_defaults_bytes",
        "dedup_masks_bytes",
        "dedup_exceptions_bytes",
        "dedup_total_bytes",
        "expected_tokens_file",
    ]
    meta: Dict[str, Any] = {}
    for k in keys:
        if k in obj:
            meta[k] = obj[k]
    return meta


def _flatten_prompt_rounds(strict_obj: Dict[str, Any], run_idx: Optional[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prompts = strict_obj.get("prompts", [])
    if not isinstance(prompts, list):
        return out

    for p in prompts:
        if not isinstance(p, dict):
            continue

        prompt_index = p.get("prompt_index")
        prompt_id = p.get("prompt_id")
        prompt_text = p.get("prompt")
        rounds = p.get("rounds", [])

        if not isinstance(rounds, list) or len(rounds) == 0:
            rec = {
                "run": run_idx,
                "prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "round": None,
                "generated_token_count": p.get("tokens_generated"),
                "generated_token_ids": p.get("tokens"),
                "window_time_s": p.get("window_time_s"),
                "prefill_time_s": p.get("prefill_time_s"),
                "decode_time_s": p.get("decode_time_s"),
                "tps_true": p.get("tps_true"),
                "tps_window": p.get("tps_window"),
                "expected_present": p.get("expected_present"),
                "expected_ok": p.get("expected_all_ok"),
            }
            out.append(rec)
            continue

        for r in rounds:
            if not isinstance(r, dict):
                continue
            rec = {
                "run": run_idx,
                "prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "round": r.get("round"),
                "generated_token_count": r.get("tokens_generated"),
                "generated_token_ids": r.get("tokens"),
                "window_time_s": r.get("window_time_s"),
                "prefill_time_s": r.get("prefill_time_s"),
                "decode_time_s": r.get("decode_time_s"),
                "expected_present": r.get("expected_present"),
                "expected_ok": r.get("expected_ok"),
                "mismatch_index": r.get("mismatch_index"),
                "expected_at": r.get("expected_at"),
                "actual_at": r.get("actual_at"),
            }
            out.append(rec)

    return out


def build_report_from_runs(runs: List[Dict[str, Any]], meta_src: Dict[str, Any]) -> Dict[str, Any]:
    summary_keys = [
        "tokens_generated",
        "wall_time_s",
        "prefill_time_s",
        "decode_time_s",
        "tps_true",
        "tps_window",
        "latency_p50_ms",
        "latency_p95_ms",
        "rss_peak_mb",
        "kv_hits",
        "kv_misses",
        "prompts_count",
        "rounds",
        "minflt",
        "majflt",
        "swap_in_mb",
        "swap_out_mb",
        "pss_peak_mb",
        "vms_peak_mb",
        "rss_floor_mb",
        "rss_delta_mb",
    ]
    summary: Dict[str, Any] = {}
    if runs:
        last = runs[-1]
        for k in summary_keys:
            if k in last:
                summary[k] = last[k]

    records: List[Dict[str, Any]] = []
    for idx, run_obj in enumerate(runs):
        records.extend(_flatten_prompt_rounds(run_obj, idx))

    meta = _pick_run_meta(meta_src)
    if runs:
        run_meta = _pick_run_meta(runs[-1])
        for k, v in run_meta.items():
            if k not in meta:
                meta[k] = v

    return {
        "meta": meta,
        "summary": summary,
        "records": records,
    }


def _emit_jsonl(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(json.dumps({"type": "meta", **(report.get("meta") or {})}, ensure_ascii=False))
    lines.append(json.dumps({"type": "summary", **(report.get("summary") or {})}, ensure_ascii=False))
    for rec in report.get("records") or []:
        lines.append(json.dumps({"type": "record", **rec}, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def _emit_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


def _run_strict_harness(args: argparse.Namespace, repo: pathlib.Path) -> str:
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
        "IE_DEDUP_STRICT": str(args.ie_dedup_strict),
        "IE_DEDUP_POLICY": str(args.ie_dedup_policy),
        "IE_DEDUP_CACHE_MB": str(args.ie_dedup_cache_mb),
    })
    if args.expected_tokens:
        env["EXPECTED_TOKENS"] = args.expected_tokens
    if args.report_tokens_max is not None:
        env["REPORT_TOKENS_MAX"] = str(int(args.report_tokens_max))

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
    return proc.stdout


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strict harness and emit a per-prompt report.")
    p.add_argument("--engine-bin", default=os.environ.get("ENGINE_BIN", "build/inference-engine"))
    p.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), choices=["cpu", "cuda", "ze"])
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/gpt-oss-20b"))
    p.add_argument("--prompts", dest="prompts_file", default=os.environ.get("PROMPTS", "benchmarks/prompts_10.txt"))
    p.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "1")))
    p.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "1")))
    p.add_argument("--threads", default=os.environ.get("THREADS", "0"))
    p.add_argument("--precision", default=os.environ.get("PRECISION", "fp32"))
    p.add_argument("--batch", default=os.environ.get("BATCH", "1"))
    p.add_argument("--prefetch", default=os.environ.get("PREFETCH", "auto"))
    p.add_argument("--pretranspose", default=os.environ.get("PRETRANSPOSE", "all"))
    p.add_argument("--affinity", default=os.environ.get("AFFINITY", "auto"))
    p.add_argument("--max-new", type=int, default=int(os.environ.get("MAX_NEW", "128")))
    p.add_argument("--ie-bytes-per-token", default=os.environ.get("IE_BYTES_PER_TOKEN", "67108864"))
    p.add_argument("--ie-stride-bytes", default=os.environ.get("IE_STRIDE_BYTES", "256"))
    p.add_argument("--ie-verify-touch", default=os.environ.get("IE_VERIFY_TOUCH", "1"))
    p.add_argument("--ie-dedup", default=os.environ.get("IE_DEDUP", "1"))
    p.add_argument("--ie-dedup-strict", default=os.environ.get("IE_DEDUP_STRICT", "0"))
    p.add_argument("--ie-dedup-policy", default=os.environ.get("IE_DEDUP_POLICY", "lossless"))
    p.add_argument("--ie-dedup-cache-mb", default=os.environ.get("IE_DEDUP_CACHE_MB", "0"))
    p.add_argument("--expected-tokens", default=os.environ.get("EXPECTED_TOKENS", ""))
    p.add_argument("--report-tokens-max", type=int, default=None)

    p.add_argument("--strict-sh", default=os.environ.get("STRICT_SH", "scripts/true_tps_strict.sh"))
    p.add_argument("--out", dest="out_path", default=None, help="Output path (default: stdout).")
    p.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Output format (default: jsonl).")

    p.add_argument("--in", dest="in_path", default=None, help="Input path (default: stdin). Use '-' for stdin.")
    p.add_argument("pos_in", nargs="?", default=None, help="Positional input path (optional).")
    p.add_argument("pos_out", nargs="?", default=None, help="Positional output path (optional).")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    repo = _repo_root_from_here()

    in_path = args.in_path
    out_path = args.out_path

    if in_path is None and args.pos_in is not None:
        in_path = args.pos_in
    if out_path is None and args.pos_out is not None:
        out_path = args.pos_out

    if in_path is not None:
        raw = _read_all_text(in_path)
    else:
        raw = _run_strict_harness(args, repo)

    objs = _parse_jsonl_objects(raw)
    runs, meta = _select_runs_and_meta(objs)
    report = build_report_from_runs(runs, meta)

    if args.format == "json":
        text = _emit_json(report)
    else:
        text = _emit_jsonl(report)

    _write_all_text(out_path, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
