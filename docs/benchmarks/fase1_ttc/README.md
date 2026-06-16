# Fase 1 — Cómputo en inferencia (TTC)

Resultados de la Fase 1 (motor de cómputo en inferencia: muestrear N soluciones y
elegir la mejor con un selector), **sin entrenar nada**. Modelo base
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (vLLM, GPU A100, temp 0.6 / top_p 0.95).

## Contenido de esta carpeta

- `RESUMEN_ttc.md` — tablas de **precisión vs N** por benchmark y por selector.
- `precision_vs_n_{aime,gpqa,gsm8k}.png` — las mismas curvas en gráfica (3 selectores cada una).
- `COMPLETO_{aime,gpqa,gsm8k}.md` / `.jsonl` — **una entrada por problema**: la pregunta,
  la respuesta correcta, la **distribución de las N respuestas finales** del modelo, y
  la respuesta elegida por cada selector (mayoría / auto-certeza / verificador) con correcto/incorrecto.

## Qué se guarda y qué NO (honestidad)

Para cada muestra se guarda **la respuesta final extraída** (el número o la letra) y
**el voto de cada selector**. **NO se guarda el razonamiento largo completo** (la cadena
de pensamiento palabra por palabra de cada muestra), **por tamaño**: con N=32 serían
varios GB. Si hace falta auditar el razonamiento completo, hay que relanzar el motor
guardando el texto íntegro (`nova/inference/run_ttc_modal.py`).

Las muestras crudas (respuesta extraída + certeza + truncado por muestra) viven en el
volumen persistente de Modal `nova-data` como `ttc_samples_{benchmark}.jsonl`.

## Resultado (resumen)

| Benchmark | Baseline N=1 | Mejor con N alto | Mejora |
|-----------|:-:|:-:|:-:|
| AIME 2024 | 23,3% | 63,3% (N=32) | +40,0 |
| GPQA-Diamond | 33,8% | 43,4% (N=32) | +9,6 |
| GSM8K | 87,2% | 92,8% (N=4) | +5,6 |

El cómputo en inferencia amplifica al modelo base de forma clara donde hay margen
(AIME, GPQA), sin tocar los pesos.
