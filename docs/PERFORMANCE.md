# PERFORMANCE

## Perfil atual (baseline FP32)
- Data do experimento: **2025-10-14 15:29 UTC**
- Prompt/runner: `benchmarks/prompts.jsonl` via `benchmarks/harness.py`

### Métricas (último reporte)
- **Avg True TPS (harness)**: **6262.59** tok/s
- **p50**: **0.171** ms | **p95**: **0.202** ms
- **Tokens totais**: **160** | **amostras**: **10**

### Hot paths (flamegraph)
- `decode_step` — **78.8%**
- `ie_engine_create` — **21.2%**

> Flamegraph: `../flamegraph.svg`

## Próximas otimizações
- **GEMV**: micro-kernels AVX2/AVX-512, blocagem e FMA.
- **Threading/NUMA**: sharding por linhas + pinning (compact/scatter).
- **tanhf**: aproximação polinomial / LUT rápida.
- **Embedding**: reduzir `sinf` ou pré-computar padrões.

## Tabela de evolução
| Build/tag | Precisão | Threads | Avg True TPS | p50 (ms) | p95 (ms) | Notas |
|-----------|:--------:|:-------:|-------------:|---------:|---------:|:------|
| v0.1      | fp32     | auto    | 6262.59 | 0.171 | 0.202 | Baseline inicial |