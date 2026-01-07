#!/usr/bin/env python3
# ============================================================================
# File: scripts/pack_tokenizer.py
# ============================================================================
"""
Packs a HuggingFace-style tokenizer.json into the engine's packed tokenizer
format (IETOK1) expected by engine/src/io/tokenizer_gptoss.c.

This packer writes:
  magic:      6 bytes  "IETOK1"
  version:    u16le    1
  vocab_size: u32le
  merges_cnt: u32le
  off_vocab:  u32le
  off_merges: u32le
  off_special:u32le
  reserved:   u32le

Then at off_vocab:
  repeated vocab_size times:
    u32le len
    len bytes (utf-8 token text)

Then at off_merges:
  repeated merges_cnt times:
    u32le len_a
    len_a bytes
    u32le len_b
    len_b bytes

No special-tokens section is currently written; off_special == end of file.
"""

import argparse
import json
import os
import struct
import sys
from typing import Dict, List, Tuple


MAGIC = b"IETOK1"
VERSION_U16 = 1


def _die(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    raise SystemExit(1)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_vocab_and_merges(j: dict) -> Tuple[Dict[str, int], List]:
    model = j.get("model") or {}
    vocab = model.get("vocab") or j.get("vocab") or {}
    merges = model.get("merges") or j.get("merges") or []
    if not isinstance(vocab, dict):
        _die("error: tokenizer vocab is not an object")
    if not isinstance(merges, list):
        _die("error: tokenizer merges is not an array")
    return vocab, merges


def _extract_added_tokens(j: dict) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    arr = j.get("added_tokens") or []
    if not isinstance(arr, list):
        return out
    for t in arr:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        content = t.get("content")
        if isinstance(tid, int) and isinstance(content, str):
            out.append((tid, content))
    return out


def _normalize_merges(merges: List) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in merges:
        if isinstance(m, list) and len(m) == 2 and all(isinstance(x, str) for x in m):
            out.append((m[0], m[1]))
            continue
        if isinstance(m, str):
            sp = m.split()
            if len(sp) == 2:
                out.append((sp[0], sp[1]))
            continue
    return out


def _build_id_to_tok(vocab: Dict[str, int], added: List[Tuple[int, str]]) -> Tuple[List[str], int, int]:
    id_to_tok: Dict[int, str] = {}
    max_id = -1

    for tok, tid in vocab.items():
        if not isinstance(tok, str):
            continue
        if not isinstance(tid, int) or tid < 0:
            continue
        id_to_tok[tid] = tok
        if tid > max_id:
            max_id = tid

    added_count = 0
    for tid, content in added:
        if tid < 0:
            continue
        id_to_tok[tid] = content
        if tid > max_id:
            max_id = tid
        added_count += 1

    if max_id < 0:
        _die("error: could not find any token ids in tokenizer.json")

    vocab_size = max_id + 1
    table: List[str] = ["" for _ in range(vocab_size)]
    missing: List[int] = []
    for i in range(vocab_size):
        s = id_to_tok.get(i)
        if s is None:
            missing.append(i)
        else:
            table[i] = s

    if missing:
        preview = ", ".join(str(x) for x in missing[:20])
        _die(f"error: tokenizer has gaps in ids; missing {len(missing)} ids (first: {preview})")

    base_vocab_count = len(vocab)
    return table, base_vocab_count, added_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Model directory containing tokenizer.json")
    args = ap.parse_args()
    md = args.model_dir

    tok_json = os.path.join(md, "tokenizer.json")
    if not os.path.exists(tok_json):
        _die(f"error: tokenizer.json not found at {tok_json}")

    j = _load_json(tok_json)
    vocab, merges_raw = _extract_vocab_and_merges(j)
    added = _extract_added_tokens(j)

    id_table, base_vocab_count, added_count = _build_id_to_tok(vocab, added)
    merges = _normalize_merges(merges_raw)

    vocab_size = len(id_table)
    merges_count = len(merges)

    header_len = 6 + 2 + (6 * 4)

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<H", VERSION_U16)

    # Placeholder u32 fields
    blob += struct.pack("<IIIIII", 0, 0, 0, 0, 0, 0)

    if len(blob) != header_len:
        _die("error: internal header size mismatch")

    off_vocab = header_len

    # Vocab section
    for s in id_table:
        b = s.encode("utf-8")
        blob += struct.pack("<I", len(b))
        blob += b

    off_merges = len(blob)

    # Merges section: write (a, b) as two LP-strings per entry
    for a, b in merges:
        ab = a.encode("utf-8")
        bb = b.encode("utf-8")
        blob += struct.pack("<I", len(ab))
        blob += ab
        blob += struct.pack("<I", len(bb))
        blob += bb

    off_special = len(blob)

    # Fill header fields
    struct.pack_into(
        "<IIIIII",
        blob,
        6 + 2,
        vocab_size,
        merges_count,
        off_vocab,
        off_merges,
        off_special,
        0,
    )

    out_path = os.path.join(md, "tokenizer.ie.bin")
    with open(out_path, "wb") as f:
        f.write(blob)

    print(f"ok: wrote {out_path}")
    print(f"  vocab_size: {vocab_size} (base_vocab: {base_vocab_count}, added_tokens: {added_count})")
    print(f"  merges: {merges_count}")
    print(f"  offsets: vocab={off_vocab} merges={off_merges} special={off_special}")


if __name__ == "__main__":
    main()
