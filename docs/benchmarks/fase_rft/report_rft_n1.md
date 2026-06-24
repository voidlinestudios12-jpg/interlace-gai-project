# Nova — RFT: modelo PURO a N=1 (pass@1)

**Métrica oficial:** pass@1 a N=1 (una muestra/problema), media±desv. sobre semillas.
El verificador ORM se usó SOLO para construir el dataset dorado, no en inferencia.

| Benchmark | BASE (N=1) | RFT (N=1) | Δ | esperado base→rft |
|---|---|---|---|---|
| aime | 21.6% ± 2.7 (P=90,K=64) | — | — | — |
| gsm8k | — | — | — | — |
| gpqa | — | — | — | — |

## Veredicto
- gsm8k: no medido (sin acceso o sin datos).
- gpqa: no medido (sin acceso o sin datos).

**CONCLUSIÓN: no cumple criterio; documentar y revertir si procede.**