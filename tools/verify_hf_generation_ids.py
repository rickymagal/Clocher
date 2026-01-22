#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify engine-generated token IDs against a Hugging Face reference at the ID level.

This is designed to work *before* decode/vocab alignment is fixed.

How it works:
- The engine produces a JSONL "trace" with (at minimum):
  1) One "prompt" record per prompt:
     {"kind":"prompt","prompt_index":0,"prompt_token_ids":[...], "prompt":"..."}
  2) One "gen_step" record per generation step:
     {"kind":"gen_step","prompt_index":0,"step":0,"next_id":123}
     Optionally include:
     {"topk_ids":[...], "topk_logits":[...]}  (aligned arrays)

- For each prompt, we run the HF model in greedy mode step-by-step starting from
  the exact prompt_token_ids logged by the engine, and compare:
  - next-token ID at every step
  - optional top-k IDs, and (optionally) logits within a tolerance

Outputs:
- A JSON report (one object) with per-prompt results and an overall OK flag.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: This verifier requires 'torch' and 'transformers' installed.", file=sys.stderr)
    print(f"ERROR detail: {e}", file=sys.stderr)
    sys.exit(2)


@dataclass
class Step:
    step: int
    next_id: int
    topk_ids: Optional[List[int]] = None
    topk_logits: Optional[List[float]] = None


@dataclass
class PromptTrace:
    prompt_index: int
    prompt_text: Optional[str]
    prompt_token_ids: List[int]
    steps: List[Step]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return out


def _load_trace(path: str) -> List[PromptTrace]:
    objs = _read_jsonl(path)
    prompts: Dict[int, PromptTrace] = {}

    for o in objs:
        kind = str(o.get("kind", ""))
        if kind == "prompt":
            pi = int(o["prompt_index"])
            ptxt = o.get("prompt")
            ids = o.get("prompt_token_ids")
            if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
                raise ValueError(f"Trace prompt record missing/invalid prompt_token_ids (prompt_index={pi})")
            prompts[pi] = PromptTrace(
                prompt_index=pi,
                prompt_text=str(ptxt) if isinstance(ptxt, str) else None,
                prompt_token_ids=[int(x) for x in ids],
                steps=[],
            )
        elif kind == "gen_step":
            pi = int(o["prompt_index"])
            st = int(o["step"])
            nid = int(o["next_id"])
            topk_ids = o.get("topk_ids")
            topk_logits = o.get("topk_logits")

            tk_ids: Optional[List[int]] = None
            tk_logits: Optional[List[float]] = None

            if isinstance(topk_ids, list) and all(isinstance(x, int) for x in topk_ids):
                tk_ids = [int(x) for x in topk_ids]
            if isinstance(topk_logits, list) and all(isinstance(x, (int, float)) for x in topk_logits):
                tk_logits = [float(x) for x in topk_logits]

            if pi not in prompts:
                # Allow traces that emit steps before the prompt record by creating a stub.
                prompts[pi] = PromptTrace(prompt_index=pi, prompt_text=None, prompt_token_ids=[], steps=[])

            prompts[pi].steps.append(Step(step=st, next_id=nid, topk_ids=tk_ids, topk_logits=tk_logits))

    out = list(prompts.values())
    out.sort(key=lambda p: p.prompt_index)

    for p in out:
        p.steps.sort(key=lambda s: s.step)
        if not p.prompt_token_ids:
            raise ValueError(f"Missing prompt_token_ids for prompt_index={p.prompt_index} (no 'prompt' record?)")

    return out


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s == "fp16" or s == "float16":
        return torch.float16
    if s == "bf16" or s == "bfloat16":
        return torch.bfloat16
    if s == "fp32" or s == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype '{s}' (use fp16|bf16|fp32)")


def _device_from_str(s: str) -> torch.device:
    s = s.strip().lower()
    if s == "cuda":
        return torch.device("cuda")
    if s == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{s}' (use cuda|cpu)")


def _topk_from_logits(logits: torch.Tensor, k: int) -> Tuple[List[int], List[float]]:
    # logits: [vocab]
    vals, idx = torch.topk(logits, k)
    ids = [int(x) for x in idx.tolist()]
    lgs = [float(x) for x in vals.tolist()]
    return ids, lgs


def _compare_topk(
    engine_ids: Optional[List[int]],
    engine_logits: Optional[List[float]],
    hf_ids: List[int],
    hf_logits: List[float],
    tol: Optional[float],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if engine_ids is None:
        return True, None  # nothing to compare
    if engine_ids != hf_ids:
        return False, {"reason": "topk_ids_mismatch", "engine_topk_ids": engine_ids, "hf_topk_ids": hf_ids}
    if tol is None or engine_logits is None:
        return True, None
    if len(engine_logits) != len(hf_logits):
        return False, {"reason": "topk_logits_len_mismatch", "engine_topk_logits": engine_logits, "hf_topk_logits": hf_logits}
    diffs = [abs(float(a) - float(b)) for a, b in zip(engine_logits, hf_logits)]
    mx = max(diffs) if diffs else 0.0
    if mx > float(tol):
        return False, {"reason": "topk_logits_tol_exceeded", "max_abs_diff": mx, "tol": float(tol)}
    return True, None


def _check_prompt_tokens(
    tokenizer: Any,
    prompt_text: str,
    prompt_ids: List[int],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    hf_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if hf_ids == prompt_ids:
        return True, None
    # Report first mismatch and lengths for quick debugging.
    min_len = min(len(hf_ids), len(prompt_ids))
    mismatch_at = None
    for i in range(min_len):
        if int(hf_ids[i]) != int(prompt_ids[i]):
            mismatch_at = i
            break
    return False, {
        "reason": "prompt_token_ids_mismatch",
        "engine_len": len(prompt_ids),
        "hf_len": len(hf_ids),
        "mismatch_index": mismatch_at,
        "engine_id": int(prompt_ids[mismatch_at]) if mismatch_at is not None else None,
        "hf_id": int(hf_ids[mismatch_at]) if mismatch_at is not None else None,
    }


def _verify_prompt(
    model: Any,
    prompt_ids: List[int],
    steps: List[Step],
    topk: int,
    tol: Optional[float],
    device: torch.device,
) -> Dict[str, Any]:
    # Greedy next-token check, step-by-step.
    # We keep a running list of ids so we don't depend on decode.
    gen: List[int] = []
    first_mismatch: Optional[Dict[str, Any]] = None

    with torch.no_grad():
        for s in steps:
            # Build input ids = prompt + generated_so_far
            ids = prompt_ids + gen
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_ids=inp)
            logits = out.logits[0, -1, :]  # [vocab]

            hf_next = int(torch.argmax(logits).item())
            if hf_next != int(s.next_id):
                first_mismatch = {
                    "step": int(s.step),
                    "engine_next_id": int(s.next_id),
                    "hf_next_id": hf_next,
                }
                break

            # Optional top-k comparison (IDs and optionally logits)
            if topk > 0 and (s.topk_ids is not None or s.topk_logits is not None):
                hf_ids, hf_lgs = _topk_from_logits(logits, topk)
                ok, detail = _compare_topk(s.topk_ids, s.topk_logits, hf_ids, hf_lgs, tol)
                if not ok:
                    first_mismatch = {
                        "step": int(s.step),
                        "engine_next_id": int(s.next_id),
                        "hf_next_id": hf_next,
                        "topk_detail": detail,
                    }
                    break

            gen.append(hf_next)

    return {
        "ok": first_mismatch is None,
        "steps_checked": len(gen) if first_mismatch is None else int(first_mismatch["step"]),
        "first_mismatch": first_mismatch,
    }


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify engine token IDs against HF greedy generation.")
    p.add_argument("--trace-jsonl", required=True, help="Engine trace JSONL (prompt + gen_step records).")
    p.add_argument("--hf-dir", required=True, help="Path to HF model directory (config + weights).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda|cpu")
    p.add_argument("--dtype", default="fp16" if torch.cuda.is_available() else "fp32", help="fp16|bf16|fp32")
    p.add_argument("--max-prompts", type=int, default=0, help="If >0, verify only first N prompts.")
    p.add_argument("--topk", type=int, default=0, help="If >0, compare top-k IDs/logits when present in trace.")
    p.add_argument("--logits-tol", type=float, default=None, help="If set, max abs diff allowed for top-k logits.")
    p.add_argument("--check-prompt", action="store_true", help="Verify prompt tokenization against HF tokenizer.")
    p.add_argument("--tokenizer-only", action="store_true",
                   help="Only verify prompt tokenization; skip HF model generation.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    return p.parse_args()


def main() -> int:
    args = _parse()
    device = _device_from_str(args.device)
    dtype = _dtype_from_str(args.dtype)

    traces = _load_trace(args.trace_jsonl)
    if args.max_prompts and args.max_prompts > 0:
        traces = traces[: int(args.max_prompts)]

    # Tokenizer not strictly required for ID-level checks, but loading it ensures
    # the HF folder is complete and config is readable.
    tok = AutoTokenizer.from_pretrained(args.hf_dir, use_fast=True)

    model = None
    if not args.tokenizer_only:
        # Load model
        # - On CUDA, default to half precision to reduce VRAM unless the user requests otherwise.
        # - On CPU, float32 is safest (bf16 may also work on some CPUs).
        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if device.type == "cuda":
            # Let transformers place layers automatically across GPUs if available.
            load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.hf_dir, **load_kwargs)
        model.eval()

        # If we are on CPU without device_map, explicitly move.
        if device.type == "cpu":
            model.to(device)

    results: List[Dict[str, Any]] = []
    overall_ok = True
    first_fail: Optional[Dict[str, Any]] = None

    for pt in traces:
        prompt_ok = True
        prompt_mismatch: Optional[Dict[str, Any]] = None
        if args.check_prompt:
            if not pt.prompt_text:
                prompt_ok = False
                prompt_mismatch = {"reason": "prompt_text_missing"}
            else:
                prompt_ok, prompt_mismatch = _check_prompt_tokens(tok, pt.prompt_text, pt.prompt_token_ids)

        if args.tokenizer_only:
            r = {
                "ok": True,
                "steps_checked": 0,
                "first_mismatch": None,
                "skipped_generation_check": True,
            }
        else:
            r = _verify_prompt(
                model=model,
                prompt_ids=pt.prompt_token_ids,
                steps=pt.steps,
                topk=int(args.topk),
                tol=args.logits_tol,
                device=device,
            )
        rec = {
            "prompt_index": pt.prompt_index,
            "prompt_len": len(pt.prompt_token_ids),
            "gen_steps": len(pt.steps),
            "prompt_ok": prompt_ok,
            "prompt_mismatch": prompt_mismatch,
            **r,
        }
        results.append(rec)
        if (not rec["ok"] or not prompt_ok) and overall_ok:
            overall_ok = False
            if not prompt_ok:
                first_fail = {"prompt_index": pt.prompt_index, **(prompt_mismatch or {})}
            else:
                first_fail = {"prompt_index": pt.prompt_index, **(rec.get("first_mismatch") or {})}

    out = {
        "ok": overall_ok,
        "prompts_verified": len(results),
        "first_failure": first_fail,
        "per_prompt": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"[verify_hf_generation_ids] ok={overall_ok} prompts={len(results)} out={args.out}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
