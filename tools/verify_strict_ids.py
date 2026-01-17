#!/usr/bin/env python3
"""
verify_strict_ids.py

Validate HF ID-check results embedded in strict JSONL output (build/strict_*.json).

Exit code:
  0 = all ID checks passed
  1 = failures found or no ID-check data present
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def _load_jsonl(path: str) -> List[Tuple[int, Dict[str, Any]]]:
    out: List[Tuple[int, Dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"line {lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"line {lineno}: JSON root is not an object")
            out.append((lineno, obj))
    return out


def _iter_per_prompt(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    pp = obj.get("per_prompt")
    if not isinstance(pp, list):
        return []
    return [p for p in pp if isinstance(p, dict)]


def _fail(msg: str) -> None:
    print(f"[verify] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strict_jsonl", help="Path to strict JSONL output (build/strict_cpu.json)")
    ap.add_argument("--max-failures", type=int, default=50, help="Cap failures shown")
    args = ap.parse_args()

    if not os.path.isfile(args.strict_jsonl):
        _fail(f"ERROR: file not found: {args.strict_jsonl}")
        return 1

    try:
        rows = _load_jsonl(args.strict_jsonl)
    except Exception as e:
        _fail(str(e))
        return 1

    failures: List[str] = []
    checks_seen = 0

    for lineno, obj in rows:
        for rec in _iter_per_prompt(obj):
            ok = rec.get("token_id_check_ok")
            prompt_ok = rec.get("token_id_check_prompt_ok")
            if ok is None and prompt_ok is None:
                continue
            checks_seen += 1
            if ok is not True or prompt_ok is False:
                pi = rec.get("prompt_index")
                step = rec.get("token_id_check_first_mismatch_step")
                eng = rec.get("token_id_check_engine_id")
                hf = rec.get("token_id_check_hf_id")
                reason = rec.get("token_id_check_reason")
                failures.append(
                    f"line {lineno} prompt_index={pi} ok={ok} prompt_ok={prompt_ok} "
                    f"mismatch_step={step} engine_id={eng} hf_id={hf} reason={reason}"
                )
                if len(failures) >= args.max_failures:
                    break
        if len(failures) >= args.max_failures:
            break

    if checks_seen == 0:
        _fail("FAIL: no token_id_check fields found in strict JSONL")
        return 1

    if failures:
        _fail(f"FAIL: {len(failures)} prompts failed ID check")
        for msg in failures[: args.max_failures]:
            print(f"  - {msg}")
        return 1

    print(f"[verify] OK: {checks_seen} prompt ID checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
