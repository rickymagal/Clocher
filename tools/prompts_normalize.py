#!/usr/bin/env python3
"""
Normalize a newline-delimited prompts file.

Goal:
- Make a clean prompts file where each prompt is one logical line.
- Strip BOM, normalize line endings, optionally drop empty lines, optionally dedupe.
- Preserve the prompt text verbatim (other than chosen normalization steps).

Typical use:
  python3 tools/prompts_normalize.py --in prompts.txt --out prompts.norm.txt

Notes:
- This script does NOT escape content into JSON by default; it keeps one prompt per line.
- If your prompts contain literal newlines, store them as \n sequences inside a single line.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        data = f.read()
    # Strip UTF-8 BOM if present.
    if data.startswith("\ufeff"):
        data = data.lstrip("\ufeff")
    # Normalize CRLF/CR into LF, then split.
    data = data.replace("\r\n", "\n").replace("\r", "\n")
    return data.split("\n")


def _normalize_lines(
    lines: Iterable[str],
    strip: bool,
    drop_empty: bool,
    dedupe: bool,
    dedupe_keep_order: bool,
) -> List[str]:
    out: List[str] = []
    seen = set()

    for line in lines:
        s = line
        if strip:
            s = s.strip()
        if drop_empty and s == "":
            continue

        if dedupe:
            if s in seen:
                continue
            seen.add(s)

        out.append(s)

    # If not keeping order during dedupe, sort for stability.
    if dedupe and not dedupe_keep_order:
        out = sorted(out)

    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input prompts file (one prompt per line).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output prompts file.")
    ap.add_argument("--strip", action="store_true", help="Strip leading/trailing whitespace from each line.")
    ap.add_argument("--keep-empty", action="store_true", help="Keep empty lines as empty prompts.")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicate prompts.")
    ap.add_argument(
        "--dedupe-keep-order",
        action="store_true",
        help="When deduping, preserve first-seen order (default is sorted for stability).",
    )

    args = ap.parse_args(argv)

    lines = _read_lines(args.in_path)

    norm = _normalize_lines(
        lines,
        strip=bool(args.strip),
        drop_empty=(not bool(args.keep_empty)),
        dedupe=bool(args.dedupe),
        dedupe_keep_order=bool(args.dedupe_keep_order),
    )

    # Ensure trailing newline for POSIX friendliness.
    with open(args.out_path, "w", encoding="utf-8", newline="\n") as f:
        for s in norm:
            f.write(s)
            f.write("\n")

    in_count = len(lines)
    out_count = len(norm)
    dropped = in_count - out_count
    sys.stderr.write(f"[prompts_normalize] in_lines={in_count} out_prompts={out_count} dropped={dropped}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
