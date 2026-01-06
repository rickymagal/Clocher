#!/usr/bin/env python3
import tiktoken, argparse, struct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Base encoding for GPT-OSS-20B
    enc = tiktoken.get_encoding("o200k_harmony")


    # Add the model’s chat special tokens
    specials = {
        "<|start|>": 200006,
        "<|end|>": 200007,
        "<|message|>": 200008,
    }
    for k, v in specials.items():
        enc._special_tokens[k] = v

    ranks = enc._mergeable_ranks
    specials_rev = {v: k for k, v in enc._special_tokens.items()}

    with open(args.out, "wb") as f:
        # header: number of mergeable ranks
        f.write(struct.pack("<Q", len(ranks)))
        for s, i in sorted(ranks.items(), key=lambda kv: kv[1]):
            if isinstance(s, (bytes, bytearray)):
                s_bytes = s
            elif isinstance(s, str):
                s_bytes = s.encode("utf-8")
            elif isinstance(s, int):
                s_bytes = str(s).encode("utf-8")
            else:
                raise TypeError(f"unexpected key type {type(s)}")

            f.write(struct.pack("<I", i))
            f.write(struct.pack("<I", len(s_bytes)))
            f.write(s_bytes)

        # specials
        f.write(struct.pack("<Q", len(specials_rev)))
        for i, s in specials_rev.items():
            s_bytes = s.encode("utf-8")
            f.write(struct.pack("<I", i))
            f.write(struct.pack("<I", len(s_bytes)))
            f.write(s_bytes)

    all_ids = list(ranks.values()) + list(specials_rev.keys())
    print("ok: wrote", args.out)
    print("  vocab_size :", len(all_ids))
    print("  specials   :", len(specials_rev))
    print("  max_token_id:", max(all_ids))

if __name__ == "__main__":
    main()
