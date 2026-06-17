# Baselines — DeepSeek-R1-Distill-Qwen-1.5B

Evaluación del modelo base `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (float16, motor vLLM, GPU A100).
Generación con la configuración oficial de DeepSeek-R1: `temperature=0.6`, `top_p=0.95`, una muestra por problema.

| Benchmark | N | Aciertos | Porcentaje | Truncados |
|--------------|------|----------|------------|-----------|
| GSM8K | 1000 | 857 | 85.70 % | 1 |
| AIME 2024 | 30 | 7 | 23.33 % | 1 |
| GPQA-Diamond | 198 | 74 | 37.37 % | 1 |

## Archivos

- `results_<benchmark>.jsonl` — detalle por problema: pregunta, respuesta completa del modelo, predicción, respuesta correcta, acierto (sí/no) y truncado (sí/no).
- `report_<benchmark>.md` — informe legible con las métricas y el desglose de cada problema.

## Notas

- **Truncados:** problemas en los que el modelo de 1.5B entró en un bucle de repetición y agotó el tope de tokens. Se rehicieron con un tope ampliado (32 768 → 65 536); el truncado restante en cada benchmark es un bucle infinito irrecuperable a cualquier tope (limitación del modelo, no de la evaluación).
- **AIME** se evalúa con una sola muestra por problema sobre 30 problemas, por lo que su porcentaje tiene una varianza alta (la referencia oficial ~28,9 % es un promedio de 64 muestras).

---

## Fase 1 — cómputo en inferencia (TTC)

Sin entrenar nada: para cada problema se generan **N** soluciones (mismo modelo, temp 0.6 / top_p 0.95)
y se elige la respuesta final con un selector (mayoría / auto-certeza / verificador). Mide cómo sube
la precisión al aumentar N. Detalle, curvas y datos en [`fase1_ttc/`](fase1_ttc/).

**Mejora (N=1 vs mejor N), mismo arnés:**

| Benchmark | N=1 (base) | Mejor con N alto | Mejora |
|--------------|:-:|:-:|:-:|
| AIME 2024 (30) | 23,3 % | **63,3 %** (N=32) | **+40,0** |
| GPQA-Diamond (198) | 33,8 % | **43,4 %** (N=32) | +9,6 |
| GSM8K (250) | 87,2 % | **92,8 %** (N=4) | +5,6 |

- El mejor selector a N intermedio es **auto-certeza** (en AIME: 37 % vs 27 % en N=4; 60 % vs 53 % en N=16).
- Gráficas: `fase1_ttc/precision_vs_n_{aime,gpqa,gsm8k}.png`. Tablas completas: `fase1_ttc/RESUMEN_ttc.md`.
- **Conclusión:** el cómputo en inferencia amplifica al modelo base donde hay margen (AIME, GPQA) sin tocar los pesos.

---

## Fase 2 — SFT de siembra: INTENTADA y REVERTIDA

Se entrenó un adaptador QLoRA (Light-R1, 1.000 trazas) → **empeoró** el modelo (AIME 63%→17%, GSM8K 93%→86%).
Por el principio "nunca romper la base" → **revertido. Nova = v0.** Causa y evidencia en
[`leccion_fase2/`](leccion_fase2/): choque de formato `<think>` (la plantilla lo añade y las trazas también)
+ lr/épocas agresivos → el modelo perdió su razonamiento profundo. El adaptador quedó archivado (no se usa).

**Nova-v0 (Fase 1) sigue siendo el resultado sólido del proyecto.**
