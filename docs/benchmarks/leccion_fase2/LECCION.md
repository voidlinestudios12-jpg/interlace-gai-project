# Lección — por qué falló el SFT de la Fase 2 (Nova-v1) y se revirtió

**Resultado (medido con el motor de la Fase 1, mismo arnés):** el SFT **empeoró** el modelo.

| Benchmark (mejor selector) | Nova-v0 (base) | Nova-v1 (SFT) |
|---|---|---|
| AIME 2024 (N=32) | **63,3 %** | 16,7 % |
| GSM8K (N=4, 250) | **92,8 %** | 86,0 % |

Por el principio nº1 de la spec (*"si una métrica baja claramente → revertir, nunca avanzar rompiendo"*) → **revertido. Nova = v0.** El adaptador quedó archivado en el volumen Modal como `adapters/ARCHIVADO_nova-v1-sft-FALLIDO` (no se usa).

## Diagnóstico (qué pasó, con evidencia)

1. **NO fue el truncamiento de datos.** De 1.000 trazas de entrenamiento, **0** superaban el límite de 8.192 tokens → ninguna perdió su `\boxed{}` final.
2. **La plantilla de chat añade `<think>` ella sola.** El prompt termina en:
   `...put your final answer within \boxed{}.<｜Assistant｜><think>\n`
   Y las trazas de Light-R1 **también** empiezan con `<think>`. Ese choque de formato (más un `learning_rate` alto de 2e-4 y 3 épocas) **corrompió el razonamiento**.
3. **Efecto observado en Nova-v1** (ver `v1_aime_respuesta.txt` y `v1_gsm8k_respuesta.txt`):
   - Genera **soluciones directas y cortas** (AIME: 1.339 tokens; GSM8K: 177) **sin el bloque exploratorio `<think>...</think>`** (`<think>`=0, `</think>`=0 en la salida).
   - Perdió el **razonamiento profundo nativo** del modelo R1, que es justo lo que hacía fuerte a v0 en problemas difíciles.
   - Curioso: en problemas FÁCILES sigue acertando (resolvió bien el AIME #1 → 33, y GSM8K), pero en los DIFÍCILES se hunde → de ahí el desplome de AIME.

## Lección para cualquier SFT futuro (no se aplica ahora)

- **Cuidar el formato `<think>` del modelo R1:** quitar el `<think>` inicial de las trazas (la plantilla ya lo añade) o desactivar esa inyección — evitar el `<think>` doble.
- **Suave:** `learning_rate` mucho más bajo (~5e-5) y **1 época**, no 3.
- **Medir el FORMATO, no solo la pérdida:** una pérdida que baja NO garantiza mejora. Hay que verificar que el modelo sigue produciendo `<think>...</think>` correctos **y** re-medir los benchmarks antes de fiarse.
- **Plan B de la spec:** si el SFT no rinde, apoyarse en lo que ya funciona (motor de cómputo en inferencia de la Fase 1, que da AIME 63 %).

## Archivos
- `v1_aime_respuesta.txt`, `v1_gsm8k_respuesta.txt` — respuestas completas de Nova-v1 (evidencia).
