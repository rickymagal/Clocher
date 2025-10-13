# Design (baseline)

- Process: single binary `inference-engine`.
- CLI prints one JSON line per run (metrics consumed by harness).
- Engine API: create → generate → metrics → destroy.
- Tokenizer/weights: stubbed for baseline; replaced with real loader in Step 2–3.
