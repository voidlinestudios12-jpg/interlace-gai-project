# Fase 3 — Paso 0: N alto en AIME (30 problemas, N_max=128)

Fecha: 2026-06-19 01:01
Generacion: temp 0.6 / top_p 0.95 / max_tokens 32768.
Muestras truncadas (sin EOS al agotar tokens): 588 (de 3840).

## Mayoria vs Oracle (pass@N) por N

| N | mayoria | oracle (pass@N) | hueco de seleccion |
|---|---|---|---|
| 8 | 40.0% (12/30) | 60.0% (18/30) | +20.0 pts |
| 16 | 46.7% (14/30) | 63.3% (19/30) | +16.7 pts |
| 32 | 50.0% (15/30) | 73.3% (22/30) | +23.3 pts |
| 64 | 50.0% (15/30) | 80.0% (24/30) | +30.0 pts |
| 128 | 53.3% (16/30) | 83.3% (25/30) | +30.0 pts |

- **mayoria**: lo que se elegiria en produccion (voto mayoritario).
- **oracle**: techo de cualquier selector (si alguna muestra acierta). El gold solo mide el techo.
- **hueco**: lo maximo que un verificador perfecto podria rescatar a cada N.
