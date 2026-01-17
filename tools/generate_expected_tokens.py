#!/usr/bin/env python3
"""
generate_expected_tokens.py

Generate an expected_tokens.txt file from the current engine output.

Each line:
  <prompt_id><space><token0,token1,token2,...>

prompt_id is taken from the engine JSON (FNV-1a over prompt bytes).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional


def _repo_root_from_here() -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    if (cwd / "build" / "inference-engine").is_file():
        return cwd
    here = pathlib.Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "build" / "inference-engine").is_file():
            return p
    return cwd


def _run_engine(args: argparse.Namespace, repo: pathlib.Path) -> Dict[str, Any]:
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

    prompts = pathlib.Path(args.prompts).expanduser()
    if not prompts.is_absolute():
        prompts = (repo / prompts).resolve()
    if not prompts.is_file():
        raise SystemExit(f"error: prompts file not found: {prompts}")

    cmd = [
        str(engine_bin),
        "--device",
        args.device,
        "--model-dir",
        str(model_dir),
        "--prompts-file",
        str(prompts),
        "--threads",
        str(args.threads),
        "--precision",
        args.precision,
        "--batch",
        str(args.batch),
        "--prefetch",
        args.prefetch,
        "--pretranspose",
        args.pretranspose,
        "--affinity",
        args.affinity,
        "--max-new",
        str(args.max_new),
        "--rounds",
        str(args.rounds),
        "--warmup",
        str(args.warmup),
    ]

    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.greedy:
        cmd += ["--greedy"]

    env = os.environ.copy()
    env["IE_DEDUP"] = "1"
    if args.ie_dedup_strict:
        env["IE_DEDUP_STRICT"] = "1"
    if args.ie_sample_kind:
        env["IE_SAMPLE_KIND"] = args.ie_sample_kind
    if args.ie_seed_env:
        env["IE_SEED"] = str(args.ie_seed_env)

    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    text = proc.stdout.strip()
    if not text:
        raise SystemExit("error: engine produced no stdout")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: failed to parse engine JSON: {e}")


def _extract_expected_lines(obj: Dict[str, Any]) -> List[str]:
    prompts = obj.get("prompts", [])
    if not isinstance(prompts, list):
        raise SystemExit("error: engine JSON missing prompts list")

    lines: List[str] = []
    for p in prompts:
        if not isinstance(p, dict):
            continue
        prompt_id = p.get("prompt_id")
        if not isinstance(prompt_id, str):
            raise SystemExit("error: prompt_id missing in engine JSON")

        rounds = p.get("rounds", [])
        tokens: Optional[List[int]] = None
        if isinstance(rounds, list) and rounds:
            r0 = rounds[0]
            if isinstance(r0, dict):
                tok = r0.get("tokens")
                if isinstance(tok, list):
                    tokens = [int(x) for x in tok]
        if tokens is None:
            tok = p.get("tokens")
            if isinstance(tok, list):
                tokens = [int(x) for x in tok]

        if tokens is None or not tokens:
            raise SystemExit(f"error: missing tokens for prompt_id {prompt_id}")

        token_csv = ",".join(str(x) for x in tokens)
        lines.append(f"{prompt_id} {token_csv}")

    return lines


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate expected_tokens.txt from engine output.")
    p.add_argument("--engine-bin", default=os.environ.get("ENGINE_BIN", "build/inference-engine"))
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/gpt-oss-20b"))
    p.add_argument("--prompts", default=os.environ.get("PROMPTS", "benchmarks/prompts_10.txt"))
    p.add_argument("--out", default=os.environ.get("EXPECTED_TOKENS", "benchmarks/expected_tokens.txt"))
    p.add_argument("--tokenizer", default=os.environ.get("TOKENIZER", ""))
    p.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    p.add_argument("--threads", default=os.environ.get("THREADS", "0"))
    p.add_argument("--precision", default=os.environ.get("PRECISION", "fp32"))
    p.add_argument("--batch", default=os.environ.get("BATCH", "1"))
    p.add_argument("--prefetch", default=os.environ.get("PREFETCH", "auto"))
    p.add_argument("--pretranspose", default=os.environ.get("PRETRANSPOSE", "all"))
    p.add_argument("--affinity", default=os.environ.get("AFFINITY", "auto"))
    p.add_argument("--max-new", type=int, default=int(os.environ.get("MAX_NEW", "128")))
    p.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "1")))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP_TOKENS", "0")))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--ie-dedup-strict", action="store_true")
    p.add_argument("--ie-sample-kind", default=os.environ.get("IE_SAMPLE_KIND", ""))
    p.add_argument("--ie-seed-env", default=os.environ.get("IE_SEED", ""))
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    repo = _repo_root_from_here()

    obj = _run_engine(args, repo)
    lines = _extract_expected_lines(obj)

    out_path = pathlib.Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
