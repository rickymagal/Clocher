#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from typing import List, Tuple

MOE_BIAS_SUFFIXES = (
    ".mlp.experts.gate_up_proj_bias",
    ".mlp.experts.down_proj_bias",
)

ROUTER_BIAS_SUFFIX = ".mlp.router.bias"


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp, path)


def strip_tensors(ie: dict, remove_router_bias: bool) -> Tuple[dict, List[str]]:
    tensors = ie.get("tensors", [])
    if not isinstance(tensors, list):
        raise RuntimeError("model.ie.json: expected 'tensors' to be a list")

    kept = []
    removed: List[str] = []

    for t in tensors:
        name = t.get("name", "")
        if not isinstance(name, str):
            kept.append(t)
            continue

        drop = False
        if name.endswith(MOE_BIAS_SUFFIXES):
            drop = True
        if remove_router_bias and name.endswith(ROUTER_BIAS_SUFFIX):
            drop = True

        if drop:
            removed.append(name)
        else:
            kept.append(t)

    ie2 = dict(ie)
    ie2["tensors"] = kept
    return ie2, removed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Remove optional MoE bias tensors from model.ie.json so infer_gptoss.c does not reject size mismatches."
    )
    ap.add_argument("--model-dir", required=True, help="Model directory containing model.ie.json")
    ap.add_argument("--dry-run", action="store_true", help="Do not modify files, only report what would be removed")
    ap.add_argument("--remove-router-bias", action="store_true", help="Also remove model.layers.*.mlp.router.bias")
    args = ap.parse_args()

    model_dir = args.model_dir
    ie_path = os.path.join(model_dir, "model.ie.json")

    if not os.path.isfile(ie_path):
        raise FileNotFoundError(f"missing: {ie_path}")

    ie = _load_json(ie_path)
    ie2, removed = strip_tensors(ie, remove_router_bias=args.remove_router_bias)

    print(f"model.ie.json: tensors_before={len(ie.get('tensors', []))} tensors_after={len(ie2.get('tensors', []))}")
    print(f"removed={len(removed)}")
    for n in removed[:20]:
        print(f"  - {n}")
    if len(removed) > 20:
        print(f"  ... ({len(removed) - 20} more)")

    if args.dry_run:
        print("dry-run: no files modified")
        return 0

    bak_path = ie_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(ie_path, bak_path)
        print(f"backup_written={bak_path}")
    else:
        print(f"backup_exists={bak_path}")

    _write_json(ie_path, ie2)
    print(f"updated={ie_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
