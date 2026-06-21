# CLAUDE.md — Interlace / Nova (instrucciones para Claude Code)

> **Retomas el proyecto (probablemente en otro ordenador). Haz esto PRIMERO:**
> 1. Lee `docs/benchmarks/fase3_prm/ESTADO_Y_RETOMAR.md` (handoff detallado: estado, comandos, bloqueos).
> 2. `git log --oneline -15` para ver lo último que se hizo.
> 3. Revisa `docs/benchmarks/` (resultados Fase 1/2/3).

## Quién y cómo trabajar
- Dueño: **Alex** (ejecuta y aprueba). Tú = **PM técnico con voto de arquitectura**.
- **Español, claro y sencillo.** **Reporta tras CADA paso. NO pases de fase sin que Alex revise.**
- **Estima el gasto de GPU antes de tandas grandes** (piloto pequeño primero). Honestidad: descontaminar
  datos de evaluación, auditar resultados, no inventar. Si algo choca con la realidad, pregunta.

## Qué es el proyecto
Dos modelos de razonamiento sobre `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`: **Nova** (maestro, máxima
capacidad) y **Kairo** (eficiente, después). **Nova-v0** = base congelada + motor de inferencia (Fase 1,
fuerte: AIME 23→63%, GPQA 34→43%, GSM8K 87→93%). Fase 2 (SFT) falló y se revirtió. **Nunca romper la base.**

## Estado actual: FASE 3 — verificador entrenado (ORM)
- **Pipeline (código):** `nova/forge/preparar_datos_verificador.py` (pasos `preparar`/`generar`/`filtrar`;
  fuentes `math` y `numina`; modo `--spawn` para Modal detached). Entreno: `nova/forge/sft_verificador.py`.
  Medición: `nova/inference/run_verif_eval_modal.py` (AIME 2023+2024+2025).
- **Datos** (en volumen Modal `nova-data` y backup HF `InterlaceAI/nova-verif-data`):
  - `verif_dataset_v1.jsonl` — LISTO (3200 soluciones / 400 problemas MATH).
  - `verif_dataset_numina.jsonl` — enriquecimiento tipo-AIME, **184/600 generados** (faltan idx 1184-1599).
    NuminaMath (olympiads+aops_forum) **triplica** los problemas con la correcta en minoría vs v1.
- **Siguiente:** terminar generación (416) → `filtrar` para `verif_dataset_v2.jsonl` →
  **entrenar el ORM (PARAR a que Alex revise)** → medir vs voto mayoritario en AIME. Si bate → Nova-v1; si no → LARC (Fase 4).

## Dónde vive todo (nube)
- **Código:** este repo GitHub. **Datos:** volumen Modal `nova-data`. **Backup de datos:** HF dataset privado `InterlaceAI/nova-verif-data`.
- Cuentas: Modal **`voidlinestudios12`**, Hugging Face **`InterlaceAI`**.

## Cómputo
- **Modal (preferido):** BLOQUEADO por *spending limit* del ciclo. Arreglo: subir el límite en
  `modal.com → Settings → Billing/Usage` (NO es saldo). Reanudar con `--detach --spawn` (si no, se cancela al desconectar).
- **Kaggle (alternativa):** `nova/forge/kaggle_generar_verificador.py` — notebook listo (GPU T4 + Internet +
  secreto `HF_TOKEN`); genera tirando los problemas de HF y sube el resultado a HF. Reanudable.

## Entorno (Windows)
Python 3.14 (PyManager) → usar `python -m pip`. Las CLIs (`modal`, `hf`) están en `...\Scripts` (en PATH al
abrir una terminal NUEVA). **`setx PYTHONUTF8 1`** o la CLI de Modal peta con los glyphs ✓ (cp1252).
Ejecuta `modal run` desde la raíz del repo.
