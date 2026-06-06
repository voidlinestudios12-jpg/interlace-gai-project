# Interlace · GAI Project

Proyecto de investigación para construir dos modelos de razonamiento sobre una base común (DeepSeek-R1-Distill-Qwen-1.5B):

- Nova — modelo de máxima capacidad (prioriza la calidad sin restricción de coste).
- Kairo — modelo eficiente (máxima capacidad por unidad de coste).

Estado: PRECONSTRUCCIÓN. Los modelos aún no están entrenados; este repositorio contiene el diseño, la documentación y, progresivamente, el código.

## Estructura
- docs/ — documentación (model cards de Nova y Kairo, decisiones, benchmarks).
- shared/ — carga del modelo base y utilidades comunes.
- nova/ — código de Nova (larc, cortex, veritas, forge, praxis).
- kairo/ — código de Kairo (gear, thrift).
- notebooks/ — notebooks de Kaggle/Colab.
- data/ — datasets y trazas (no versionados; viven en Drive/Kaggle).
- checkpoints/ — pesos de modelos (no versionados; viven en Drive/Kaggle).
- results/ — logs, métricas y gráficas.

## Entorno
- Desarrollo y entrenamiento en Kaggle Notebooks (GPU, principal) y Google Colab (respaldo).
- Stack: Python, PyTorch, HuggingFace Transformers, PEFT (LoRA/QLoRA).
