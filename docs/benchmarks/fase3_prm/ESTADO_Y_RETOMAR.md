# Fase 3 — Estado y cómo RETOMAR (handoff entre ordenadores)

Fecha: 2026-06-21. Escrito al cambiar de ordenador. **Léeme primero para continuar.**

## Dónde vive todo (nube)
- **Código:** este repo GitHub (privado). Incluye el pipeline del verificador en `nova/forge/preparar_datos_verificador.py` (pasos `preparar`/`generar`/`filtrar`, fuentes `math` y `numina`, modo `--spawn`), entreno `nova/forge/sft_verificador.py`, medición `nova/inference/run_verif_eval_modal.py`.
- **Datos/artefactos:** volumen **Modal `nova-data`** (cuenta `voidlinestudios12`). Es nube: accesible desde cualquier PC con el token de Modal. **No se pierde nada al cambiar de ordenador.**
- **Pesos futuros:** Hugging Face privado (cuenta `InterlaceAI`).

## Estado de la Fase 3 (verificador ORM)
- **v1 LISTO:** `verif_dataset_v1.jsonl` = 3200 soluciones / 400 problemas MATH (niveles 3-5). 79% correctas, 141 problemas "mixtos".
- **Enriquecimiento NuminaMath (en curso):**
  - `verif_problemas_numina.jsonl` = **600 problemas** olympiads+aops_forum (idx 1000-1599), descontaminados contra AIME 23/24/25 + MATH500 + GSM8K. Pool disponible: 57k.
  - `verif_dataset_numina.jsonl` = generación **184/600 hecha** (idx 1000-1183) = 1472 soluciones, **86 mixtos / 45 "correcta-en-minoría" (tipo-AIME)**. **Faltan 416** (idx 1184-1599).
- **Hallazgo clave:** DeepSeek-distill es **bimodal** en MATH (resuelve casi siempre o casi nunca); la franja "mixta" ≈37% suba el nivel que suba. NuminaMath (competición) **triplica** los problemas con la correcta en minoría (v1≈8,5% → NuminaMath≈25%), que son los que importan para AIME. Por eso se eligió NuminaMath.

## ⛔ BLOQUEO actual: límite de gasto de Modal
`modal run` falla con **"workspace billing cycle spend limit reached"**. Es el **límite de gasto (spending limit) del ciclo** (se agotó el crédito gratis ~$30/mes). **Añadir saldo NO basta** si el spending limit sigue bajo.
- **Arreglo:** en `modal.com → Settings → Billing/Usage` **subir el "spending limit"** del ciclo (o esperar al reinicio mensual — la fecha sale en el dashboard).

## Cómo CONTINUAR (en el ordenador principal)
1. `git pull` (rama `main`). Reconectar: `python -m pip install -U modal huggingface_hub`; `modal token new`; `hf auth login`. En **Windows**: fijar `PYTHONUTF8=1` (`setx PYTHONUTF8 1`) o la CLI de Modal peta con los glyphs ✓.
2. Resolver el límite de Modal (arriba).
3. **Reanudar la generación** (reanudable; salta los 184 ya hechos). USAR `--detach --spawn` para que NO se cancele al desconectar/apagar:
   ```
   modal run --detach nova/forge/preparar_datos_verificador.py --paso generar --spawn \
     --k 8 --max-tokens 12288 --problemas /data/verif_problemas_numina.jsonl \
     --dataset /data/verif_dataset_numina.jsonl --limite 0
   ```
   Verificar progreso real: `modal volume get nova-data verif_dataset_numina.jsonl .` y contar idx distintos (debe crecer hacia 600).
4. **Construir v2:** `modal run nova/forge/preparar_datos_verificador.py --paso filtrar --hard /data/verif_dataset_numina.jsonl` → `verif_dataset_v2.jsonl` = v1 + mixtos de NuminaMath.
5. **Entrenar el ORM** (PARAR a que Alex revise antes): `nova/forge/sft_verificador.py`. TODO: su `DATASET_FILE` está fijo a v1; apuntarlo a `verif_dataset_v2.jsonl` (parametrizar o cambiar la constante). El script ya hace split por problema + pesos de clase + intento de subir a HF `nova-verificador-v1`.
6. **Medir:** `run_verif_eval_modal.py --paso generar` y `--paso medir` (AIME 2023+2024+2025; selectores mayoria vs verificador_prm vs oracle).

## Alternativa Kaggle (si Modal sigue capado)
`nova/eval/run_benchmark.py` ya es self-contained para Kaggle (carga el modelo con transformers, extractores `extraer_num`/`comparar_num`). Plan para generar datos del verificador en Kaggle T4:
1. Subir `verif_problemas_numina.jsonl` a un dataset de Kaggle o a HF (privado).
2. Notebook: cargar DeepSeek-R1-Distill-Qwen-1.5B (vLLM o transformers), generar K=8 por problema (temp 0.6/top_p 0.95), etiquetar con el gold, guardar jsonl.
3. Subir el jsonl resultante a HF/Modal y seguir desde el paso `filtrar`.

(Pendiente: portar la función `generar()` a un notebook Kaggle autocontenido.)
