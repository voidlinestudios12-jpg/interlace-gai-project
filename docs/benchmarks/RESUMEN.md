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
