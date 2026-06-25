# Fase RFT — Modelo puro medido a N=1

**Métrica.** A partir de esta fase, la métrica de referencia del modelo es **pass@1 a N=1**:
una sola muestra por problema, modelo puro (sin best-of-N). El verificador entrenado se usa como
herramienta para seleccionar datos de entrenamiento, no en inferencia.

**Baseline (a batir).** AIME (90 problemas, 2023+2024+2025): **pass@1 N=1 ≈ 21.6%**.

## Qué se probó

**Rejection-sampling fine-tuning (RFT / STaR):** a partir de las soluciones CORRECTAS que el propio
modelo genera sobre problemas de entrenamiento (MATH + NuminaMath, descontaminados de todo el set de
evaluación), se construye un dataset "dorado" eligiendo la mejor solución por problema con el
verificador, y se entrena un adaptador LoRA sobre la base **congelada** (lr bajo, pocas épocas,
ancla KL para no olvidar). Se probaron dos configuraciones: una conservadora y otra con más empuje
y datos pesados hacia los problemas difíciles (correcta-en-minoría).

## Resultados (pass@1 a N=1)

| Benchmark | Base | RFT (conservador) | RFT (con más empuje) |
|---|---|---|---|
| AIME (90) | 21.6% | 21.1% | 18.9% |
| GSM8K (200) | 81.1% | 83.5% | 78.8% |
| GPQA (198) | ~34% | 33.7% | 31.8% |

## Conclusión y siguiente paso

- A nivel de **dificultad media (GSM8K)** el RFT conservador aporta una mejora pequeña (+2.4 pp).
- A nivel **AIME (dificultad alta)** el RFT **no mejora** el modelo puro a N=1.
- Lectura: el RFT refuerza caminos de razonamiento que el modelo **ya produce**; por eso ayuda en lo
  que casi resuelve, pero no enseña caminos nuevos en los problemas más difíciles.

**Siguiente palanca prevista: RL con recompensa de corrección verificable (GRPO / RLVR)** — premia
acertar contra el gold y enseña razonamientos nuevos, más adecuado para mover el N=1 en AIME.
La base congelada (Nova-v0) se mantiene como modelo de referencia.
