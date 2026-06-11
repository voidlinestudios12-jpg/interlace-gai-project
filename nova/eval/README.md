# nova/eval — Benchmark del modelo base

[`run_benchmark.py`](run_benchmark.py) mide `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
(float16, GPU) en **GSM8K**, **AIME 2024** o **GPQA-Diamond**, sin truncar las
cadenas de razonamiento y con guardado incremental reanudable. No entrena nada.

## Uso en Kaggle

1. Notebook nuevo → Settings: **Accelerator: GPU** (T4 vale) e **Internet: ON**.
2. Pega el contenido **entero** de `run_benchmark.py` en una celda.
3. Edita las dos variables del principio:
   ```python
   BENCHMARK = "gsm8k"   # "gsm8k" | "aime" | "gpqa"
   N = 1000              # para una prueba rápida, ponlo en 3
   ```
4. **Save & Run All (Commit)**. Las salidas quedan en `/kaggle/working`
   (pestaña *Output* de la versión).

### GPQA (dataset gated)

GPQA-Diamond requiere un token de HuggingFace con acceso aceptado:

1. Acepta las condiciones en <https://huggingface.co/datasets/Idavidrein/gpqa>.
2. Crea un token de lectura en <https://huggingface.co/settings/tokens>.
3. En Kaggle: *Add-ons → Secrets* → crea el secreto `HF_TOKEN` y actívalo para
   el notebook (en local: variable de entorno `HF_TOKEN`).

Si falta el token o no tiene acceso, el script lo dice claramente y se detiene.

## Salidas

- `results_{benchmark}.jsonl` — una línea por problema con `i`, `pregunta`,
  `respuesta` (completa), `prediccion`, `correcta`, `acierto`, `truncado`.
  Se escribe con flush inmediato; si el archivo ya existe al arrancar, el
  script **reanuda** desde la última línea válida (no repite problemas).
- `report_{benchmark}.md` — informe con la configuración, la precisión, el
  número de truncados y el detalle completo de cada problema.

Para reanudar en Kaggle tras un corte de sesión: copia el `results_*.jsonl`
del run anterior a `/kaggle/working` antes de ejecutar y seguirá desde ahí.

## Detalles

- Prompt según la recomendación oficial de DeepSeek: sin system prompt, con
  temperature 0.6 y top_p 0.95.
- Anti-truncamiento: `max_new_tokens=32768`; si el modelo no emite EOS, se
  continúa la generación hasta un tope total de 49152 tokens. Si aun así no
  termina, el resultado se marca `truncado`.
- Los datos se descargan por HTTP directo (sin la librería `datasets`, que da
  problemas con pyarrow en Kaggle).

## Reutilizar la extracción de respuestas

Las funciones de corrección no dependen de torch y son importables:

```python
from run_benchmark import extraer_boxed, extraer_num, extraer_letra, comparar_num

extraer_num(r"... so the answer is \boxed{7,200}.")  # -> "7200"
extraer_letra(r"... the correct option is \boxed{C}")  # -> "C"
```

## Uso en local

```bash
pip install torch transformers requests pandas
python run_benchmark.py   # requiere GPU CUDA
```
