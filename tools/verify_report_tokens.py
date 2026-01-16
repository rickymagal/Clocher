#!/usr/bin/env python3
"""
verify_report_tokens.py

Verifies per-prompt JSONL benchmark reports for tokenizer / decode parity.

It can run in two modes:
  1) Structural-only checks (default if HF tokenizer is unavailable)
  2) HF parity checks (requires: pip install transformers tokenizers)

Typical usage:
  python3 tools/verify_report_tokens.py build/reports/cpu/report.jsonl --hf-tokenizer models/gpt-oss-20b/hf

Exit code:
  0 = all checks passed
  1 = failures found (or HF requested but unavailable)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_utf8(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", errors="strict"))


def is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def as_int_list(x: Any) -> Optional[List[int]]:
    if x is None:
        return None
    if not isinstance(x, list):
        return None
    out: List[int] = []
    for i, v in enumerate(x):
        if not is_int(v):
            return None
        out.append(int(v))
    return out


def json_loads_line(line: str, lineno: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(line)
    except Exception as e:
        return None, f"line {lineno}: invalid JSON: {e}"
    if not isinstance(obj, dict):
        return None, f"line {lineno}: JSON root is not an object"
    return obj, None


@dataclass
class Failure:
    lineno: int
    prompt_id: Optional[Any]
    run: Optional[Any]
    round_idx: Optional[Any]
    field: str
    message: str
    details: Optional[Dict[str, Any]] = None


def get_field(obj: Dict[str, Any], key: str) -> Any:
    return obj.get(key, None)


def norm_prompt_bytes(prompt: str) -> bytes:
    # Keep bytes exact. If prompt is already a Python str, encode to UTF-8 strictly.
    return prompt.encode("utf-8", errors="strict")


def try_import_transformers() -> Tuple[bool, Optional[str]]:
    try:
        import transformers  # noqa: F401
        import tokenizers  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def load_hf_tokenizer(tokenizer_path_or_name: str, revision: Optional[str]) -> Any:
    from transformers import AutoTokenizer

    kwargs: Dict[str, Any] = {"use_fast": True}
    if revision:
        kwargs["revision"] = revision

    tok = AutoTokenizer.from_pretrained(tokenizer_path_or_name, **kwargs)

    # Some tokenizers default to adding special tokens. We want exact raw behavior.
    # We enforce add_special_tokens=False when encoding.
    return tok


def hf_encode(tok: Any, prompt: str) -> List[int]:
    enc = tok(prompt, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    ids = enc.get("input_ids", None)
    if ids is None:
        raise RuntimeError("HF tokenizer did not return input_ids")
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
        # Batched; should not happen for single string, but handle anyway.
        ids = ids[0]
    if not isinstance(ids, list):
        raise RuntimeError("HF tokenizer input_ids is not a list")
    out: List[int] = []
    for v in ids:
        if not is_int(v):
            raise RuntimeError("HF tokenizer returned non-int token id")
        out.append(int(v))
    return out


def hf_decode(tok: Any, ids: List[int]) -> str:
    # clean_up_tokenization_spaces=False to keep parity; skip_special_tokens=False because we did not add any.
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def best_effort_get_meta(obj: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    prompt_id = obj.get("prompt_id", obj.get("prompt_index", obj.get("prompt_idx", None)))
    run = obj.get("run", obj.get("run_idx", None))
    round_idx = obj.get("round", obj.get("round_idx", None))
    return prompt_id, run, round_idx


def get_generated_ids(obj: Dict[str, Any]) -> Optional[List[int]]:
    gen = as_int_list(obj.get("generated_token_ids", None))
    if gen is not None:
        return gen
    return as_int_list(obj.get("tokens", None))


def validate_record_structure(
    obj: Dict[str, Any],
    lineno: int,
    failures: List[Failure],
    require_prompt_tokens: bool,
    require_generated_tokens: bool,
) -> bool:
    ok = True
    prompt_id, run, round_idx = best_effort_get_meta(obj)

    prompt = get_field(obj, "prompt")
    if not isinstance(prompt, str):
        failures.append(Failure(lineno, prompt_id, run, round_idx, "prompt", "missing or not a string"))
        ok = False

    p_ids = as_int_list(get_field(obj, "prompt_token_ids"))
    if p_ids is None and require_prompt_tokens:
        failures.append(Failure(lineno, prompt_id, run, round_idx, "prompt_token_ids", "missing or not a list[int]"))
        ok = False

    g_ids = get_generated_ids(obj)
    if g_ids is None and require_generated_tokens:
        failures.append(Failure(lineno, prompt_id, run, round_idx, "generated_token_ids", "missing token ids"))
        ok = False

    # Optional count checks (if present, must match)
    p_cnt = get_field(obj, "prompt_token_count")
    if p_cnt is not None:
        if not is_int(p_cnt):
            failures.append(Failure(lineno, prompt_id, run, round_idx, "prompt_token_count", "present but not int"))
            ok = False
        elif p_ids is not None and int(p_cnt) != len(p_ids):
            failures.append(
                Failure(
                    lineno,
                    prompt_id,
                    run,
                    round_idx,
                    "prompt_token_count",
                    "count does not match len(prompt_token_ids)",
                    {"prompt_token_count": int(p_cnt), "len(prompt_token_ids)": len(p_ids)},
                )
            )
            ok = False

    g_cnt = get_field(obj, "generated_token_count")
    if g_cnt is not None:
        if not is_int(g_cnt):
            failures.append(Failure(lineno, prompt_id, run, round_idx, "generated_token_count", "present but not int"))
            ok = False
        elif g_ids is not None and int(g_cnt) != len(g_ids):
            failures.append(
                Failure(
                    lineno,
                    prompt_id,
                    run,
                    round_idx,
                    "generated_token_count",
                    "count does not match len(generated_token_ids)",
                    {"generated_token_count": int(g_cnt), "len(generated_token_ids)": len(g_ids)},
                )
            )
            ok = False

    # Token id sanity
    for field_name, ids in (("prompt_token_ids", p_ids), ("generated_token_ids", g_ids)):
        if ids is None:
            continue
        for i, tid in enumerate(ids):
            if tid < 0:
                failures.append(
                    Failure(
                        lineno,
                        prompt_id,
                        run,
                        round_idx,
                        field_name,
                        "negative token id encountered",
                        {"index": i, "token_id": tid},
                    )
                )
                ok = False

    # Optional decoded_text fields sanity
    for tfield in ("decoded_text", "decoded_text_full", "decoded_text_gen"):
        val = obj.get(tfield, None)
        if val is not None and not isinstance(val, str):
            failures.append(Failure(lineno, prompt_id, run, round_idx, tfield, "present but not a string"))
            ok = False

    return ok


def compare_lists(a: List[int], b: List[int], max_items: int) -> Dict[str, Any]:
    # Returns small diff summary.
    same_prefix = 0
    for x, y in zip(a, b):
        if x == y:
            same_prefix += 1
        else:
            break
    same_suffix = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x == y:
            same_suffix += 1
        else:
            break
    return {
        "len_a": len(a),
        "len_b": len(b),
        "same_prefix": same_prefix,
        "same_suffix": same_suffix,
        "a_head": a[:max_items],
        "b_head": b[:max_items],
        "a_tail": a[-max_items:] if len(a) > max_items else a,
        "b_tail": b[-max_items:] if len(b) > max_items else b,
    }


def check_hash_fields(obj: Dict[str, Any], lineno: int, failures: List[Failure]) -> None:
    prompt_id, run, round_idx = best_effort_get_meta(obj)

    prompt = obj.get("prompt", None)
    if isinstance(prompt, str):
        want = obj.get("prompt_sha256", None)
        if isinstance(want, str) and len(want) == 64:
            got = sha256_utf8(prompt)
            if got != want:
                failures.append(
                    Failure(
                        lineno,
                        prompt_id,
                        run,
                        round_idx,
                        "prompt_sha256",
                        "hash mismatch",
                        {"expected": want, "got": got},
                    )
                )

    def ids_sha(ids: Optional[List[int]]) -> str:
        # Stable byte encoding: little-endian int32
        if ids is None:
            return ""
        buf = io.BytesIO()
        for v in ids:
            buf.write(int(v).to_bytes(4, byteorder="little", signed=True))
        return sha256_bytes(buf.getvalue())

    p_ids = as_int_list(obj.get("prompt_token_ids", None))
    g_ids = get_generated_ids(obj)

    want = obj.get("prompt_token_ids_sha256", None)
    if isinstance(want, str) and len(want) == 64 and p_ids is not None:
        got = ids_sha(p_ids)
        if got != want:
            failures.append(
                Failure(
                    lineno,
                    prompt_id,
                    run,
                    round_idx,
                    "prompt_token_ids_sha256",
                    "hash mismatch",
                    {"expected": want, "got": got},
                )
            )

    want = obj.get("generated_token_ids_sha256", None)
    if isinstance(want, str) and len(want) == 64 and g_ids is not None:
        got = ids_sha(g_ids)
        if got != want:
            failures.append(
                Failure(
                    lineno,
                    prompt_id,
                    run,
                    round_idx,
                    "generated_token_ids_sha256",
                    "hash mismatch",
                    {"expected": want, "got": got},
                )
            )

    for tfield, hfield in (("decoded_text", "decoded_text_sha256"), ("decoded_text_full", "decoded_text_full_sha256")):
        text = obj.get(tfield, None)
        want_h = obj.get(hfield, None)
        if isinstance(text, str) and isinstance(want_h, str) and len(want_h) == 64:
            got_h = sha256_utf8(text)
            if got_h != want_h:
                failures.append(
                    Failure(
                        lineno,
                        prompt_id,
                        run,
                        round_idx,
                        hfield,
                        "hash mismatch",
                        {"expected": want_h, "got": got_h},
                    )
                )


def hf_parity_checks(
    tok: Any,
    obj: Dict[str, Any],
    lineno: int,
    failures: List[Failure],
    max_items: int,
    text_mode: str,
) -> None:
    prompt_id, run, round_idx = best_effort_get_meta(obj)

    prompt = obj.get("prompt", None)
    if not isinstance(prompt, str):
        return

    p_ids = as_int_list(obj.get("prompt_token_ids", None))
    g_ids = get_generated_ids(obj)
    if p_ids is None or g_ids is None:
        return

    # Tokenization parity: prompt -> ids
    try:
        hf_p_ids = hf_encode(tok, prompt)
    except Exception as e:
        failures.append(Failure(lineno, prompt_id, run, round_idx, "hf_encode", f"HF encode failed: {e}"))
        return

    if hf_p_ids != p_ids:
        failures.append(
            Failure(
                lineno,
                prompt_id,
                run,
                round_idx,
                "prompt_token_ids",
                "HF tokenization mismatch",
                {"diff": compare_lists(p_ids, hf_p_ids, max_items)},
            )
        )

    # Decode parity checks. We support a few layouts:
    #  - decoded_text_full: full text (prompt+generated)
    #  - decoded_text_gen: generated-only decoded text
    #  - decoded_text: either full or generated-only (we infer via text_mode)
    decoded_full = obj.get("decoded_text_full", None)
    decoded_gen = obj.get("decoded_text_gen", None)
    decoded_generic = obj.get("decoded_text", None)

    ids_full = p_ids + g_ids

    try:
        hf_full = hf_decode(tok, ids_full)
        hf_gen = hf_decode(tok, g_ids)
    except Exception as e:
        failures.append(Failure(lineno, prompt_id, run, round_idx, "hf_decode", f"HF decode failed: {e}"))
        return

    def check_text(field: str, expected: str, actual: Any) -> None:
        if actual is None:
            return
        if not isinstance(actual, str):
            return
        if actual != expected:
            # Provide a compact diff window
            a = actual
            b = expected
            n = 160
            details = {
                "actual_head": a[:n],
                "expected_head": b[:n],
                "actual_tail": a[-n:] if len(a) > n else a,
                "expected_tail": b[-n:] if len(b) > n else b,
                "len_actual": len(a),
                "len_expected": len(b),
            }
            failures.append(Failure(lineno, prompt_id, run, round_idx, field, "HF decode mismatch", details))

    check_text("decoded_text_full", hf_full, decoded_full)
    check_text("decoded_text_gen", hf_gen, decoded_gen)

    # Generic field: user selects interpretation.
    if isinstance(decoded_generic, str):
        if text_mode == "full":
            check_text("decoded_text", hf_full, decoded_generic)
        elif text_mode == "gen":
            check_text("decoded_text", hf_gen, decoded_generic)
        else:
            # auto: accept either exact match; otherwise flag with both expected variants.
            if decoded_generic != hf_full and decoded_generic != hf_gen:
                details = {
                    "expected_full_head": hf_full[:160],
                    "expected_gen_head": hf_gen[:160],
                    "actual_head": decoded_generic[:160],
                    "expected_full_tail": hf_full[-160:] if len(hf_full) > 160 else hf_full,
                    "expected_gen_tail": hf_gen[-160:] if len(hf_gen) > 160 else hf_gen,
                    "actual_tail": decoded_generic[-160:] if len(decoded_generic) > 160 else decoded_generic,
                }
                failures.append(Failure(lineno, prompt_id, run, round_idx, "decoded_text", "HF decode mismatch", details))


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_jsonl", help="Path to report.jsonl")
    ap.add_argument("--out-json", default="", help="Write machine-readable summary JSON here")
    ap.add_argument("--max-failures", type=int, default=50, help="Cap failures collected")
    ap.add_argument("--max-diff-items", type=int, default=24, help="Max token ids shown in diffs")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any failures (default: true)")
    ap.add_argument("--no-strict", dest="strict", action="store_false", help="Always exit 0")
    ap.set_defaults(strict=True)

    ap.add_argument("--hf-tokenizer", default="", help="HF tokenizer path or repo name (enables HF parity)")
    ap.add_argument("--hf-revision", default="", help="HF revision tag/sha")
    ap.add_argument(
        "--text-mode",
        choices=["auto", "full", "gen"],
        default="auto",
        help="How to interpret decoded_text when present",
    )
    ap.add_argument(
        "--require-expected",
        action="store_true",
        help="Fail if expected_present is false or missing on any record",
    )

    args = ap.parse_args()

    report_path = args.report_jsonl
    if not os.path.isfile(report_path):
        print(f"ERROR: report not found: {report_path}", file=sys.stderr)
        return 1 if args.strict else 0

    failures: List[Failure] = []
    total = 0
    json_errors = 0
    structure_failed = 0

    hf_enabled = bool(args.hf_tokenizer)
    hf_ok = False
    hf_tok = None
    hf_import_err = None

    if hf_enabled:
        hf_ok, hf_import_err = try_import_transformers()
        if not hf_ok:
            failures.append(
                Failure(
                    lineno=0,
                    prompt_id=None,
                    run=None,
                    round_idx=None,
                    field="hf",
                    message=f"HF requested but transformers/tokenizers unavailable: {hf_import_err}",
                )
            )
        else:
            try:
                hf_tok = load_hf_tokenizer(args.hf_tokenizer, args.hf_revision or None)
            except Exception as e:
                failures.append(
                    Failure(
                        lineno=0,
                        prompt_id=None,
                        run=None,
                        round_idx=None,
                        field="hf",
                        message=f"failed to load HF tokenizer: {e}",
                    )
                )
                hf_tok = None

    with open(report_path, "r", encoding="utf-8", errors="strict") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            obj, err = json_loads_line(line, lineno)
            if err is not None:
                json_errors += 1
                failures.append(Failure(lineno, None, None, None, "json", err))
                if len(failures) >= args.max_failures:
                    break
                continue
            assert obj is not None
            if obj.get("type", "record") != "record":
                continue

            ok_struct = validate_record_structure(
                obj,
                lineno,
                failures,
                require_prompt_tokens=hf_enabled,
                require_generated_tokens=True,
            )
            if not ok_struct:
                structure_failed += 1
                if len(failures) >= args.max_failures:
                    break
                continue

            check_hash_fields(obj, lineno, failures)
            if len(failures) >= args.max_failures:
                break

            if args.require_expected:
                exp_present = obj.get("expected_present", None)
                exp_ok = obj.get("expected_ok", None)
                if exp_present is not True:
                    failures.append(
                        Failure(lineno, *best_effort_get_meta(obj), "expected_present", "missing/false expected_present")
                    )
                elif exp_ok is not True:
                    failures.append(
                        Failure(lineno, *best_effort_get_meta(obj), "expected_ok", "expected tokens mismatch")
                    )
                if len(failures) >= args.max_failures:
                    break

            if hf_tok is not None:
                hf_parity_checks(
                    hf_tok,
                    obj,
                    lineno,
                    failures,
                    max_items=args.max_diff_items,
                    text_mode=args.text_mode,
                )
                if len(failures) >= args.max_failures:
                    break

    passed = (len(failures) == 0)
    summary = {
        "report": os.path.abspath(report_path),
        "records_total": total,
        "json_errors": json_errors,
        "structure_failed": structure_failed,
        "hf_enabled": hf_enabled,
        "hf_loaded": hf_tok is not None,
        "failures_count": len(failures),
        "failures": [
            {
                "lineno": x.lineno,
                "prompt_id": x.prompt_id,
                "run": x.run,
                "round": x.round_idx,
                "field": x.field,
                "message": x.message,
                "details": x.details or {},
            }
            for x in failures
        ],
        "ok": passed,
    }

    if args.out_json:
        write_json(args.out_json, summary)

    # Human summary
    if passed:
        print(f"[verify] OK: {total} records checked")
        return 0

    print(f"[verify] FAIL: {len(failures)} failures across {total} records")
    for i, fail in enumerate(failures[: min(len(failures), 20)]):
        meta = []
        if fail.prompt_id is not None:
            meta.append(f"prompt_id={fail.prompt_id}")
        if fail.run is not None:
            meta.append(f"run={fail.run}")
        if fail.round_idx is not None:
            meta.append(f"round={fail.round_idx}")
        meta_s = (" " + " ".join(meta)) if meta else ""
        print(f"  - line {fail.lineno}{meta_s} [{fail.field}] {fail.message}")
        if fail.details:
            try:
                print("    " + json.dumps(fail.details, sort_keys=True)[:800])
            except Exception:
                pass

    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
