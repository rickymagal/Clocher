#!/usr/bin/env python3
import argparse
import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional


MAGIC = b"IETOK1\x00\x00"
VERSION = 1


def _read_json(path: Path) -> dict:
    with path.open("rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return json.loads(raw.decode("utf-8"))


def _infer_special_ids(model_dir: Path) -> Dict[str, Optional[int]]:
    out = {"bos_id": None, "eos_id": None, "unk_id": None, "pad_id": None}
    cfg1 = model_dir / "special_tokens_map.json"
    cfg2 = model_dir / "tokenizer_config.json"

    def try_load(p: Path) -> Optional[dict]:
        try:
            return _read_json(p)
        except Exception:
            return None

    smap = try_load(cfg1) or {}
    tcfg = try_load(cfg2) or {}

    # These files often store token *strings*, not ids. We just record strings in meta,
    # and later map to ids after vocab is built.
    out["_bos_token"] = tcfg.get("bos_token") or smap.get("bos_token")
    out["_eos_token"] = tcfg.get("eos_token") or smap.get("eos_token")
    out["_unk_token"] = tcfg.get("unk_token") or smap.get("unk_token")
    out["_pad_token"] = tcfg.get("pad_token") or smap.get("pad_token")

    def norm_tok(x):
        if x is None:
            return None
        if isinstance(x, dict) and "content" in x:
            return x["content"]
        if isinstance(x, str):
            return x
        return None

    out["_bos_token"] = norm_tok(out["_bos_token"])
    out["_eos_token"] = norm_tok(out["_eos_token"])
    out["_unk_token"] = norm_tok(out["_unk_token"])
    out["_pad_token"] = norm_tok(out["_pad_token"])
    return out


def _extract_bpe(tokenizer: dict) -> Tuple[Dict[str, int], List[Tuple[str, str]], List[dict]]:
    model = tokenizer.get("model")
    if not isinstance(model, dict):
        raise SystemExit("ERROR: tokenizer.json missing top-level 'model' object")

    mtype = model.get("type")
    if mtype not in ("BPE", "bpe"):
        raise SystemExit(f"ERROR: unsupported tokenizer model type: {mtype!r} (expected 'BPE')")

    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict):
        raise SystemExit("ERROR: tokenizer.json model.vocab missing or not an object")
    if not isinstance(merges, list):
        raise SystemExit("ERROR: tokenizer.json model.merges missing or not a list")

    added_tokens = tokenizer.get("added_tokens")
    if added_tokens is None:
        added_tokens = []
    if not isinstance(added_tokens, list):
        raise SystemExit("ERROR: tokenizer.json added_tokens is not a list")

    vocab_map: Dict[str, int] = {}
    for k, v in vocab.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, int):
            continue
        vocab_map[k] = v

    merge_pairs: List[Tuple[str, str]] = []
    for m in merges:
        if not isinstance(m, str):
            continue
        parts = m.split()
        if len(parts) != 2:
            # Some tokenizers store merges as a single string with spaces in tokens.
            # If this happens, you need a different packer for that tokenizer.
            raise SystemExit(f"ERROR: unexpected merge entry {m!r} (expected 'A B')")
        merge_pairs.append((parts[0], parts[1]))

    return vocab_map, merge_pairs, added_tokens


def _build_id_to_token(vocab: Dict[str, int], added_tokens: List[dict]) -> List[str]:
    max_id = -1
    for tok, tid in vocab.items():
        if tid > max_id:
            max_id = tid

    for at in added_tokens:
        if not isinstance(at, dict):
            continue
        tid = at.get("id")
        if isinstance(tid, int) and tid > max_id:
            max_id = tid

    if max_id < 0:
        raise SystemExit("ERROR: empty vocab")

    id_to_tok = [""] * (max_id + 1)
    used = [False] * (max_id + 1)

    for tok, tid in vocab.items():
        if 0 <= tid <= max_id and not used[tid]:
            id_to_tok[tid] = tok
            used[tid] = True

    for at in added_tokens:
        if not isinstance(at, dict):
            continue
        tok = at.get("content")
        tid = at.get("id")
        if isinstance(tok, str) and isinstance(tid, int) and 0 <= tid <= max_id:
            id_to_tok[tid] = tok
            used[tid] = True

    # Validate no holes (engine expects dense ids)
    holes = [i for i, t in enumerate(id_to_tok) if t == ""]
    if holes:
        raise SystemExit(f"ERROR: tokenizer ids are not dense; first hole at id={holes[0]} "
                         f"(total holes={len(holes)})")

    return id_to_tok


def _map_special_ids(id_to_tok: List[str], special_info: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    tok_to_id = {t: i for i, t in enumerate(id_to_tok)}
    out = {"bos_id": None, "eos_id": None, "unk_id": None, "pad_id": None}

    for name, key in (("bos_id", "_bos_token"),
                      ("eos_id", "_eos_token"),
                      ("unk_id", "_unk_token"),
                      ("pad_id", "_pad_token")):
        t = special_info.get(key)
        if isinstance(t, str):
            out[name] = tok_to_id.get(t)

    return out


def _write_packed(out_path: Path,
                  id_to_tok: List[str],
                  merges: List[Tuple[str, str]],
                  special_ids: Dict[str, Optional[int]]) -> None:
    vocab_size = len(id_to_tok)
    merges_count = len(merges)

    # Layout:
    # header:
    #   magic[8], version u32
    #   vocab_size u32, merges_count u32
    #   bos_id i32, eos_id i32, unk_id i32, pad_id i32
    # section vocab:
    #   for each token id in [0..vocab_size):
    #     tok_len u32, tok_bytes
    # section merges:
    #   for each merge:
    #     a_len u32, a_bytes
    #     b_len u32, b_bytes
    #
    # (Yes, merges stored as strings; the engine can map them to ids at load-time.)
    def i32(x: Optional[int]) -> int:
        return int(x) if isinstance(x, int) else -1

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<II", vocab_size, merges_count))
        f.write(struct.pack("<iiii",
                            i32(special_ids.get("bos_id")),
                            i32(special_ids.get("eos_id")),
                            i32(special_ids.get("unk_id")),
                            i32(special_ids.get("pad_id"))))

        for tok in id_to_tok:
            b = tok.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)

        for a, b in merges:
            ab = a.encode("utf-8")
            bb = b.encode("utf-8")
            f.write(struct.pack("<I", len(ab)))
            f.write(ab)
            f.write(struct.pack("<I", len(bb)))
            f.write(bb)

        f.flush()
        os.fsync(f.fileno())

    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack HF tokenizer.json into Clocher tokenizer.ie.bin")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to HuggingFace tokenizer.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output path for tokenizer.ie.bin")
    ap.add_argument("--model-dir", dest="model_dir", default=None,
                    help="Model directory (optional) to infer special token ids from tokenizer_config.json/special_tokens_map.json")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    model_dir = Path(args.model_dir) if args.model_dir else in_path.parent

    tok = _read_json(in_path)
    vocab, merges, added_tokens = _extract_bpe(tok)
    id_to_tok = _build_id_to_token(vocab, added_tokens)
    sp_info = _infer_special_ids(model_dir)
    sp_ids = _map_special_ids(id_to_tok, sp_info)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_packed(out_path, id_to_tok, merges, sp_ids)

    meta = {
        "format": "IETOK1",
        "version": VERSION,
        "vocab_size": len(id_to_tok),
        "merges_count": len(merges),
        "special_tokens": {
            "bos_id": sp_ids.get("bos_id"),
            "eos_id": sp_ids.get("eos_id"),
            "unk_id": sp_ids.get("unk_id"),
            "pad_id": sp_ids.get("pad_id"),
            "bos_token": sp_info.get("_bos_token"),
            "eos_token": sp_info.get("_eos_token"),
            "unk_token": sp_info.get("_unk_token"),
            "pad_token": sp_info.get("_pad_token"),
        },
        "input_tokenizer_json": str(in_path),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"OK: wrote {out_path}")
    print(f"OK: wrote {meta_path}")
    print(f"vocab_size={meta['vocab_size']} merges_count={meta['merges_count']} "
          f"bos_id={sp_ids.get('bos_id')} eos_id={sp_ids.get('eos_id')} "
          f"unk_id={sp_ids.get('unk_id')} pad_id={sp_ids.get('pad_id')}")


if __name__ == "__main__":
    main()
