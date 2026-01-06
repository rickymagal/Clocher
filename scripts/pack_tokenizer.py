#!/usr/bin/env python3
# ============================================================================
# File: scripts/pack_tokenizer.py
# ============================================================================
"""
@file pack_tokenizer.py
@brief Packs a HuggingFace-style tokenizer.json into a compact binary format
       (IETOK1) usable by the GPT-OSS inference engine.

Usage:
  python3 scripts/pack_tokenizer.py --model-dir models/gpt-oss-20b

Output:
  models/gpt-oss-20b/tokenizer.ie.bin
"""

import os
import sys
import json
import struct
import argparse
import zlib

MAGIC = b"IETOK1"
VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Model directory containing tokenizer.json")
    args = ap.parse_args()
    md = args.model_dir

    tok_json = os.path.join(md, "tokenizer.json")
    if not os.path.exists(tok_json):
        sys.exit(f"error: {tok_json} not found")

    with open(tok_json, "r", encoding="utf-8") as f:
        j = json.load(f)

    vocab = j.get("model", {}).get("vocab", {})
    merges = j.get("model", {}).get("merges", [])
    added = []

    sp_map = os.path.join(md, "special_tokens_map.json")
    if os.path.exists(sp_map):
        try:
            with open(sp_map, "r", encoding="utf-8") as f:
                sm = json.load(f)
            for k, v in sm.items():
                if isinstance(v, str):
                    added.append((v, None))
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, str):
                            added.append((x, None))
        except Exception:
            pass

    vocab_items = sorted(vocab.items(), key=lambda kv: kv[1])
    n_vocab = len(vocab_items)
    n_merges = len(merges)
    n_added = len(added)

    out_path = os.path.join(md, "tokenizer.ie.bin")
    with open(out_path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IIIII", VERSION, n_vocab, n_merges, n_added, 0))

        for tok, tid in vocab_items:
            tok_b = tok.encode("utf-8")
            out.write(struct.pack("<I", len(tok_b)))
            out.write(tok_b)
            out.write(struct.pack("<I", tid))

        for m in merges:
            if isinstance(m, list) and len(m) == 2:
                a, b = m
            elif isinstance(m, str):
                sp = m.split()
                if len(sp) != 2:
                    continue
                a, b = sp
            else:
                continue
            ab = f"{a} {b}".encode("utf-8")
            out.write(struct.pack("<I", len(ab)))
            out.write(ab)

        for tok, _ in added:
            tok_b = tok.encode("utf-8")
            out.write(struct.pack("<I", len(tok_b)))
            out.write(tok_b)

        # trailer checksum
        out.write(struct.pack("<I", zlib.crc32(MAGIC)))

    print(f"ok: wrote {out_path}")
    print(f"  vocab: {n_vocab} merges: {n_merges} added: {n_added}")


if __name__ == "__main__":
    main()
