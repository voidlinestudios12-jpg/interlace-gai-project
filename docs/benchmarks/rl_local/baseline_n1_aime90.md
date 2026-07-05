# Baseline pass@1 a N=1 — aime_eval_90 (local, RTX 3060)

- **Modelo:** deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- **Arnés:** nova/eval/run_baseline_local.py (vLLM enforce_eager, bf16)
- **Generación:** temp 0.6, top_p 0.95, max_tokens 16384
- **Fecha:** 2026-07-05

| Semilla | Aciertos | N | pass@1 | Truncados |
|---|---|---|---|---|
| 101 | 21 | 90 | 23.33% | 47 |
| 102 | 18 | 90 | 20.00% | 50 |
| 103 | 17 | 90 | 18.89% | 47 |
| 104 | 18 | 90 | 20.00% | 52 |
| 105 | 18 | 90 | 20.00% | 45 |

**Media ± desviación (5 semillas): 20.44 ± 1.68 pp**

Referencia oficial de la fase RL: comparar SOLO contra números de este
mismo arnés y entorno. Truncados cuentan como fallo.
