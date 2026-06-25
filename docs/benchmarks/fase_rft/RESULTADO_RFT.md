# Fase RFT — Resultado final (modelo PURO a N=1)

**Métrica oficial:** pass@1 a N=1 (modelo puro, una muestra/problema), media±desv. sobre semillas.
El verificador ORM se usó SOLO para construir el dataset dorado, no en inferencia (best-of-N no cuenta).

**Baseline a batir:** AIME N=1 = **21.6%** (90 problemas, estimado sobre 64 muestras/problema).

---

## Dos iteraciones

### Iteración 1 — conservadora (RTX 4090)
- **Config:** lr 1e-5, 2 épocas, ancla KL β=0.1 (fuerte), dataset dorado balanceado (664 problemas).
- **Diagnóstico durante el entreno:** `ce` inicial ≈ 0.20 (el modelo YA producía esas soluciones → poca señal nueva); el modelo apenas se movió (KL ≈ 0.001).

### Iteración 2 — agresiva + datos difíciles (H100)
- **Cambios:** dataset DORADO pesado hacia problemas DIFÍCILES (correcta-en-minoría): 236 ejemplos
  (98 hard oversampleados a 146 + 90 medios; **495 fáciles descartados**), dominado por olympiads/aops.
  lr **1e-4** (10×), 3 épocas, KL β=0.05, max_seq 8192.
- **Durante el entreno:** KL ≈ 0.027 (10× más que iter1) → esta vez el modelo SÍ se movió de verdad.

---

## Tabla de resultados (pass@1 a N=1)

| Benchmark | Base | RFT v1 (conservador) | RFT v2 (agresivo) |
|---|---|---|---|
| **AIME** (90) | **21.6%** | 21.1% ± 3.3 (plano) | **18.9% ± 2.5 (−2.7pp)** |
| **GSM8K** (200) | 81.1% | **83.5% ± 1.8 (+2.4pp)** | **78.8% ± 1.0 (−2.3pp)** |
| **GPQA** (198) | ~34% (Fase 1) | 33.7% ± 2.2 (plano) | no completado |

---

## Veredicto: el RFT NO es la palanca para AIME a N=1

- **iter1 (gentil):** seguro, **mejora GSM8K (+2.4pp)** pero deja **AIME plano**. No promociona (no sube AIME).
- **iter2 (agresivo):** **REGRESIÓN en AIME (−2.7) y GSM8K (−2.3)** → sobreajuste/estrechamiento al dataset
  pequeño de difíciles. **REVERTIR** (cumple las dos condiciones de revertir del spec).
- **Ninguna versión se promociona a Nova-v1.** La base congelada queda como Nova-v0.

### Por qué el RFT no mueve AIME a N=1
El rejection-sampling FT entrena sobre soluciones **que el modelo ya genera** (su propia salida correcta). En
problemas más fáciles que AIME mejora algo (se ve en GSM8K, iter1). Pero a la dificultad de AIME no aprende
caminos NUEVOS — solo refuerza los que ya tiene. Empujar más fuerte (iter2) no enseña caminos nuevos: solo
sobreajusta y estrecha → regresión.

## Siguiente paso (según el spec): RL (GRPO / RLVR)
La palanca para N=1 es **aprendizaje por refuerzo con recompensa de corrección verificable** (premiar acertar
contra el gold): enseña caminos de razonamiento NUEVOS en vez de imitar los existentes. Después, LARC (Fase 4).
**NO empezar hasta que Alex lo apruebe.**

## Artefactos
- Datos: HF `Quantumadvancedai/nova-verif-data` (rft_dorado_v2.jsonl, eval_rft_*.jsonl).
- Adaptadores RFT (referencia, NO promocionados): HF `Quantumadvancedai/nova-rft-v1`.
- Código: `nova/forge/vast_build_golden.py` (con pesado por dificultad), `nova/forge/vast_train_rft.py`,
  `nova/inference/vast_eval_n1.py`, `nova/inference/vast_compare_n1.py`.
