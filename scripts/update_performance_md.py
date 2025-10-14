#!/usr/bin/env python3
"""
Update docs/PERFORMANCE.md from the latest report + flamegraph stacks.

- Reads newest benchmarks/reports/<ts>/{summary.json,samples.csv}
- If available, parses 'script.stacks' (perf) or 'callgrind.stacks' to list top hot paths
- Writes/overwrites docs/PERFORMANCE.md with a concise, standardized section
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmarks" / "reports"
DOC = ROOT / "docs" / "PERFORMANCE.md"
FLAMEGRAPH = ROOT / "flamegraph.svg"

def newest_report_dir() -> Path:
    dirs = [p for p in REPORTS.glob("*") if p.is_dir()]
    if not dirs:
        raise SystemExit("No report dirs found. Run `make bench` first.")
    return sorted(dirs)[-1]

def load_summary(d: Path) -> dict:
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))

def load_samples(d: Path):
    rows = []
    with (d / "samples.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows

def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else 0.0

def parse_stacks() -> list[tuple[str,int]]:
    """
    Parse folded stacks from perf or callgrind:
    - script.stacks (perf)
    - callgrind.stacks (valgrind fallback)
    Return top (func, samples) pairs sorted desc.
    """
    for name in ("script.stacks", "callgrind.stacks"):
        p = ROOT / name
        if p.exists():
            counts = {}
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                # line format: "f1;f2;f3 count"
                if " " not in line: 
                    continue
                stack, cnt = line.rsplit(" ", 1)
                try:
                    c = int(cnt)
                except:
                    continue
                # last frame is "leaf"
                leaf = stack.split(";")[-1]
                counts[leaf] = counts.get(leaf, 0) + c
            return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return []

def main():
    d = newest_report_dir()
    summary = load_summary(d)
    samples = load_samples(d)

    p50s = [float(r.get("latency_p50_ms", 0.0)) for r in samples]
    p95s = [float(r.get("latency_p95_ms", 0.0)) for r in samples]
    mean_p50 = mean(p50s)
    mean_p95 = mean(p95s)

    # Hot paths (optional)
    hot = parse_stacks()
    total = sum(c for _, c in hot) or 1
    top5 = [(f, (c/total)*100.0) for f, c in hot[:5]]

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# PERFORMANCE")
    lines.append("")
    lines.append("## Perfil atual (baseline FP32)")
    lines.append(f"- Data do experimento: **{now}**")
    lines.append("- Prompt/runner: `benchmarks/prompts.jsonl` via `benchmarks/harness.py`")
    lines.append("")
    lines.append("### Métricas (último reporte)")
    lines.append(f"- **Avg True TPS (harness)**: **{summary.get('avg_tps_true', 0.0):.2f}** tok/s")
    lines.append(f"- **p50**: **{mean_p50:.3f}** ms | **p95**: **{mean_p95:.3f}** ms")
    lines.append(f"- **Tokens totais**: **{int(summary.get('total_tokens', 0))}** | **amostras**: **{int(summary.get('samples', 0))}**")
    lines.append("")
    lines.append("### Hot paths (flamegraph)")
    if top5:
        for (fn, pct) in top5:
            lines.append(f"- `{fn}` — **{pct:.1f}%**")
    else:
        lines.append("- N/D (gere `flamegraph.svg` e `script.stacks` com `scripts/profile_flamegraph.sh`)")
    if FLAMEGRAPH.exists():
        rel = os.path.relpath(FLAMEGRAPH, start=DOC.parent)
        lines.append(f"\n> Flamegraph: `{rel}`")
    lines.append("")
    lines.append("## Próximas otimizações")
    lines.append("- **GEMV**: micro-kernels AVX2/AVX-512, blocagem e FMA.")
    lines.append("- **Threading/NUMA**: sharding por linhas + pinning (compact/scatter).")
    lines.append("- **tanhf**: aproximação polinomial / LUT rápida.")
    lines.append("- **Embedding**: reduzir `sinf` ou pré-computar padrões.")
    lines.append("")
    lines.append("## Tabela de evolução")
    lines.append("| Build/tag | Precisão | Threads | Avg True TPS | p50 (ms) | p95 (ms) | Notas |")
    lines.append("|-----------|:--------:|:-------:|-------------:|---------:|---------:|:------|")
    lines.append(f"| v0.1      | fp32     | auto    | {summary.get('avg_tps_true', 0.0):.2f} | {mean_p50:.3f} | {mean_p95:.3f} | Baseline inicial |")

    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote: {DOC}")

if __name__ == "__main__":
    main()
