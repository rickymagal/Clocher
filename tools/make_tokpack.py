#!/usr/bin/env python3
"""
Build a runtime tokenizer pack ("TKPK" v1) from a tiktoken encoding.

Pack format (little-endian):
  u8  magic[4] = "TKPK"
  u32 version  = 1
  u32 n_mergeable
  u32 n_special
  u32 pat_len
  u8  pat_bytes[pat_len]         (UTF-8)

  Repeat n_mergeable times:
    u16 len
    u8  bytes[len]               (raw bytes key)
    u32 token_id

  Repeat n_special times:
    u16 len
    u8  utf8[len]                (UTF-8 special string)
    u32 token_id
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import Dict, Tuple, Any


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def get_encoding_obj(name: str):
    try:
        import tiktoken  # type: ignore
    except Exception as e:
        die(f"failed to import tiktoken: {e}")

    try:
        enc = tiktoken.get_encoding(name)
    except Exception as e:
        die(f"tiktoken.get_encoding({name!r}) failed: {e}")
    return enc


def extract_from_encoding(enc: Any) -> Tuple[Dict[bytes, int], Dict[str, int], str]:
    """
    Tries multiple internal layouts across tiktoken versions.

    Returns:
      mergeable_ranks: Dict[bytes, int]
      special_tokens:  Dict[str, int]
      pat_str:         str
    """
    mergeable = getattr(enc, "_mergeable_ranks", None)
    special = getattr(enc, "_special_tokens", None)
    pat = getattr(enc, "_pat_str", None)

    if mergeable is None or special is None or pat is None:
        core = getattr(enc, "_core_bpe", None)
        if core is not None:
            if mergeable is None:
                mergeable = getattr(core, "mergeable_ranks", None)
            if special is None:
                special = getattr(core, "special_tokens", None)
            if pat is None:
                pat = getattr(core, "pat_str", None)

    if mergeable is None or special is None or pat is None:
        die(
            "could not extract encoding internals "
            "(mergeable_ranks/special_tokens/pat_str missing)."
        )

    if not isinstance(mergeable, dict):
        die(f"mergeable_ranks is not a dict (type={type(mergeable)})")
    if not isinstance(special, dict):
        die(f"special_tokens is not a dict (type={type(special)})")
    if not isinstance(pat, str):
        die(f"pat_str is not a str (type={type(pat)})")

    mergeable_ranks: Dict[bytes, int] = {bytes(k): int(v) for k, v in mergeable.items()}
    special_tokens: Dict[str, int] = {str(k): int(v) for k, v in special.items()}
    pat_str: str = pat
    return mergeable_ranks, special_tokens, pat_str


def write_u16(f, x: int) -> None:
    if x < 0 or x > 0xFFFF:
        die(f"u16 out of range: {x}")
    f.write(struct.pack("<H", x))


def write_u32(f, x: int) -> None:
    if x < 0 or x > 0xFFFFFFFF:
        die(f"u32 out of range: {x}")
    f.write(struct.pack("<I", x))


def build_tokpack(encoding_name: str, out_path: str) -> None:
    enc = get_encoding_obj(encoding_name)
    mergeable_ranks, special_tokens, pat_str = extract_from_encoding(enc)

    pat_bytes = pat_str.encode("utf-8")

    merge_items = sorted(mergeable_ranks.items(), key=lambda kv: (kv[1], kv[0]))
    spec_items = sorted(special_tokens.items(), key=lambda kv: (kv[1], kv[0]))

    tmp_path = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(tmp_path, "wb") as f:
        f.write(b"TKPK")
        write_u32(f, 1)  # version
        write_u32(f, len(merge_items))
        write_u32(f, len(spec_items))
        write_u32(f, len(pat_bytes))
        f.write(pat_bytes)

        for b, tid in merge_items:
            if len(b) > 0xFFFF:
                die(f"mergeable bytes too long: len={len(b)} token_id={tid}")
            write_u16(f, len(b))
            f.write(b)
            write_u32(f, tid)

        for s, tid in spec_items:
            sb = s.encode("utf-8")
            if len(sb) > 0xFFFF:
                die(f"special string too long: len={len(sb)} token_id={tid}")
            write_u16(f, len(sb))
            f.write(sb)
            write_u32(f, tid)

    os.replace(tmp_path, out_path)

    max_id_merge = max(mergeable_ranks.values()) if mergeable_ranks else -1
    max_id_spec = max(special_tokens.values()) if special_tokens else -1
    max_id = max(max_id_merge, max_id_spec)

    print(f"ok: wrote {out_path}")
    print(f"  encoding       : {encoding_name}")
    print(f"  mergeable_ranks: {len(merge_items)}")
    print(f"  special_tokens : {len(spec_items)}")
    print(f"  pat_len        : {len(pat_bytes)}")
    print(f"  max_token_id   : {max_id}")

    for k in ("<|start|>", "<|end|>", "<|message|>"):
        if k in special_tokens:
            print(f"  special {k:12s}: {special_tokens[k]}")
        else:
            print(f"  special {k:12s}: (missing)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="o200k_harmony", help="tiktoken encoding name")
    ap.add_argument("--out", required=True, help="output .tokpack path")
    args = ap.parse_args()
    build_tokpack(args.encoding, args.out)


if __name__ == "__main__":
    main()
