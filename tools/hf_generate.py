#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple


def _eprint(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt is not None:
        return prompt
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


def _resolve_device(device: str) -> str:
    d = (device or "").strip().lower()
    if d in ("cpu", "cuda", "mps"):
        return d
    return "cpu"


def _load_model_and_tokenizer(model_spec: str, device: str):
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        print(json.dumps({"error": f"missing deps: {e}", "tokens": []}))
        raise SystemExit(2)

    model_path = model_spec
    local = os.path.isdir(model_path)

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local,
        torch_dtype=None,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but torch.cuda.is_available() is false")
        model = model.to("cuda")
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("mps requested but torch.backends.mps.is_available() is false")
        model = model.to("mps")
    else:
        model = model.to("cpu")

    model.eval()
    return model, tok, torch


def _generate(
    model,
    tok,
    torch_mod,
    prompt: str,
    max_new: int,
    temperature: float,
) -> Tuple[str, list]:
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    dev = next(model.parameters()).device
    input_ids = input_ids.to(dev)
    attn = inputs.get("attention_mask")
    if attn is not None:
        attn = attn.to(dev)

    do_sample = temperature is not None and float(temperature) > 0.0

    with torch_mod.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=int(max_new),
            do_sample=bool(do_sample),
            temperature=float(temperature) if do_sample else 1.0,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    out_ids = out[0].tolist()
    prompt_len = input_ids.shape[-1]
    gen_ids = out_ids[prompt_len:]

    text = tok.decode(gen_ids, skip_special_tokens=True)
    return text, gen_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    args = ap.parse_args()

    prompt = _read_prompt(args.prompt, args.prompt_file)
    device = _resolve_device(args.device)

    try:
        model, tok, torch_mod = _load_model_and_tokenizer(args.model, device)
        text, tokens = _generate(
            model=model,
            tok=tok,
            torch_mod=torch_mod,
            prompt=prompt,
            max_new=max(0, int(args.max_new)),
            temperature=float(args.temperature),
        )
        print(json.dumps({"text": text, "tokens": tokens}))
        return 0
    except SystemExit:
        raise
    except Exception as e:
        print(json.dumps({"error": str(e), "tokens": []}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
