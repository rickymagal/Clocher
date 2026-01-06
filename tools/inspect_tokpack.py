#!/usr/bin/env python3
"""
Inspect a TKPK v1 pack and print high-level stats + special tokens.
"""

from __future__ import annotations

import argparse
import struct
import sys


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def rd_u16(b: bytes, off: int) -> tuple[int, int]:
    if off + 2 > len(b):
        die("truncated u16")
    return struct.unpack_from("<H", b, off)[0], off + 2


def rd_u32(b: bytes, off: int) -> tuple[int, int]:
    if off + 4 > len(b):
        die("truncated u32")
    return struct.unpack_from("<I", b, off)[0], off + 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--print-specials", action="store_true")
    args = ap.parse_args()

    data = open(args.path, "rb").read()
    if len(data) < 20:
        die("file too small")
    if data[:4] != b"TKPK":
        die("bad magic")

    off = 4
    version, off = rd_u32(data, off)
    if version != 1:
        die(f"unsupported version: {version}")

    n_merge, off = rd_u32(data, off)
    n_spec, off = rd_u32(data, off)
    pat_len, off = rd_u32(data, off)

    if off + pat_len > len(data):
        die("truncated pattern")
    pat = data[off : off + pat_len].decode("utf-8", errors="replace")
    off += pat_len

    for _ in range(n_merge):
        l, off = rd_u16(data, off)
        off += l
        _, off = rd_u32(data, off)

    specials = []
    for _ in range(n_spec):
        l, off = rd_u16(data, off)
        s = data[off : off + l].decode("utf-8", errors="replace")
        off += l
        tid, off = rd_u32(data, off)
        specials.append((s, tid))

    print(f"path        : {args.path}")
    print(f"version     : {version}")
    print(f"n_mergeable : {n_merge}")
    print(f"n_special   : {n_spec}")
    print(f"pat_len     : {pat_len}")
    print(f"pat_prefix  : {pat[:80]!r}")

    if args.print_specials:
        specials.sort(key=lambda kv: kv[1])
        for s, tid in specials:
            print(f"{tid:8d}  {s!r}")

    m = {s: tid for s, tid in specials}
    for k in ("<|start|>", "<|end|>", "<|message|>"):
        print(f"special {k:12s}: {m.get(k, None)}")


if __name__ == "__main__":
    main()
