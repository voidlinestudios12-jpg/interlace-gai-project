# Nova Fase 3 — Evaluación Verificador ORM

**Modelo base:** DeepSeek-R1-Distill-Qwen-1.5B  
**Eval set:** AIME 2023+2024+2025 (90 problemas)  
**Sweep N:** [8, 16, 32, 64]

## Resultados por selector

| Selector | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|
| mayoria | 33.3% (30/90) ±9.7pp | 35.6% (32/90) ±9.9pp | 35.6% (32/90) ±9.9pp | 36.7% (33/90) ±10.0pp |
| autocerteza | 23.3% (21/90) ±8.7pp | 22.2% (20/90) ±8.6pp | 18.9% (17/90) ±8.1pp | 20.0% (18/90) ±8.3pp |
| verificador_prm | 36.7% (33/90) ±10.0pp | 37.8% (34/90) ±10.0pp | 43.3% (39/90) ±10.2pp | 37.8% (34/90) ±10.0pp |
| verificador_prm_pesado | 41.1% (37/90) ±10.2pp | 45.6% (41/90) ±10.3pp | 52.2% (47/90) ±10.3pp | 52.2% (47/90) ±10.3pp |
| oracle | 47.8% (43/90) ±10.3pp | 54.4% (49/90) ±10.3pp | 60.0% (54/90) ±10.1pp | 60.0% (54/90) ±10.1pp |

## Interpretación
- **oracle**: límite superior teórico (si alguna solución entre N es correcta).
- **verificador_prm**: selecciona la solución con mayor P(correcta) según ORM.
- **verificador_prm_pesado**: voto ponderado por P(correcta) del ORM.
- **mayoria**: voto mayoritario (baseline fuerte).
- **autocerteza**: solución con mayor log-probabilidad media (baseline ligero).
