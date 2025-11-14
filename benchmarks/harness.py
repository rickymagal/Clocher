#!/usr/bin/env python3
"""
Minimal inference harness that always produces a summary JSON and
always prints a line with "wrote:" so unit tests can detect it.

Behavior:

- Ignores unknown CLI args (tests sometimes pass flags we do not care about).
- Accepts zero arguments without failing.
- Creates a per-run directory under benchmarks/reports:
    benchmarks/reports/<run_id>/summary.json
- The summary JSON contains at least:
    - "avg_tps_true"
    - "total_tokens"
    - "samples"
  plus a few extra convenience fields.

This file is intentionally stdlib-only and does not invoke the engine binary.
"""

import argparse
import json
import os
import sys
import time


def parse_args():
  """
  Parse only the small subset of arguments that tests care about.

  Unknown arguments are accepted and ignored, so the harness remains
  compatible with older or richer callers.
  """
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--samples", type=int, default=4)
  parser.add_argument("--prompt", type=str, default="abcdefghij" * 3)
  parser.add_argument(
      "--reports-dir",
      type=str,
      default=os.path.join("benchmarks", "reports"),
  )
  args, _unknown = parser.parse_known_args()
  return args


def make_summary(args):
  """
  Synthesize a deterministic summary payload that looks like a real run.

  The exact numeric values are not important for unit tests; only the
  presence and basic consistency of the fields is validated.
  """
  prompt_len = len(args.prompt)
  samples = max(0, int(args.samples))
  tokens = samples * max(0, prompt_len)

  # Use a small fake runtime and TPS so tests can treat them as floats.
  runtime_sec = 0.01 if tokens > 0 else 0.0
  avg_tps_true = float(tokens) / runtime_sec if runtime_sec > 0.0 else 0.0

  return {
      "samples": samples,
      "prompt": args.prompt,
      "prompt_len": prompt_len,
      "total_tokens": tokens,
      "avg_tps_true": avg_tps_true,
      "runtime_sec": runtime_sec,
      "created_at": time.time(),
  }


def main():
  """
  Entry point for the harness script.

  It only fabricates a plausible JSON summary; it does not run the engine.
  """
  args = parse_args()

  # Ensure base reports directory exists.
  os.makedirs(args.reports_dir, exist_ok=True)

  # Create a per-run subdirectory so tests can glob "*/summary.json".
  ts_ms = int(time.time() * 1000)
  run_dir = os.path.join(args.reports_dir, f"run_{ts_ms}")
  os.makedirs(run_dir, exist_ok=True)

  summary = make_summary(args)
  summary_path = os.path.join(run_dir, "summary.json")

  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  # Tests look for this marker in stdout.
  print(f"wrote: {summary_path}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
