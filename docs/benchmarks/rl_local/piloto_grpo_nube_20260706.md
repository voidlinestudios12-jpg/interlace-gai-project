# Piloto GRPO (PASO 4) — 150 pasos en nube (RTX 4090) — 2026-07-06

## Resumen ejecutivo

Piloto de 150 pasos completado en **2 h 32 min** en una RTX 4090 de Vast.ai
(instancia 44036184, $0.348/h). **Gasto real total del día: $2.17** (incluye
una caja fallida y toda la depuración; presupuesto aprobado $3-8).
La infraestructura queda **totalmente validada** (entrenar en nube con vLLM
colocate, checkpoints y métricas a HF, reanudable, ~60 s/paso = 22× la 3060).
La señal de GRPO existe (57% de pasos con gradiente), pero con lr 1e-6 y solo
~85 pasos efectivos la política se movió tan poco (KL ~0.0004) que **aún no se
puede afirmar ni descartar que la reward suba**: eso es exactamente lo que
debe responder el run largo.

## Contexto: por qué se movió a nube

- Intento local (3060): 22,6 min/paso → 56 h proyectadas, GPU al 98% de VRAM
  y sobrecalentamiento. Además el cap de 4096 tokens (límite de los 11 GB)
  producía ~90% de pasos con grupo 100% truncado → gradiente cero.
- Decisión de Alex (2026-07-06): estrategia híbrida; piloto y run largo a nube.

## Los tres bugs cazados (lecciones para el repo)

1. **OOM en 4090 con cap 8192**: el pico fp32 de los logits (~4.6 GB para una
   secuencia de 8K con vocab 152k) no cabe ni con vLLM al 25%. Escalera
   probada: 8192/0.40 → OOM forward · 8192/0.25+expandable_segments → OOM
   backward · vllm-sleep → no soportado (cumem) · **6144/0.20 → cabe** (~21 GB
   pico). `expandable_segments` SÍ funciona en Linux real (roto solo en WSL2);
   `train_grpo.py` ahora lo auto-detecta.
2. **bitsandbytes roto en la caja** (`libnvJitLink.so.13`): innecesario en
   nube — con LoRA, AdamW normal cuesta ~150 MB más. Nuevo flag `--optim`.
3. **CRÍTICO — gradiente cero silencioso**: el modo por defecto de TRL 1.7.0
   para corregir la discrepancia trainer-vs-vLLM (`vllm_importance_sampling_mode
   ="sequence_mask"`) suma ~0.02 nats/token (normal en bf16) sobre TODA la
   secuencia → ratio de secuencia e^-100 ≈ 0 → **multiplica el gradiente por
   cero**. 29 pasos de un primer intento entrenaron NADA (grad_norm=0 en todos;
   archivado en `results/rl_nube/grpo_train_log_bugIS_20260706.jsonl`). Fix:
   `token_truncate` (TIS estándar). **Moraleja: vigilar grad_norm e
   importance_sampling_ratio SIEMPRE, no solo la reward.**

## Config final del piloto (validada)

`--use-vllm --vllm-mem 0.20 --num-generations 8 --max-completion 6144
--optim adamw_torch` + lr 1e-6, β=0.04, LoRA r16/α32, temp 1.0, 1 prompt/paso.
Commits `2f6a178` → `439525e`. Checkpoints cada 25 pasos en
`Quantumadvancedai/nova-rl-ckpt/piloto_nube` (6 checkpoints + final).

## Resultados (150 pasos, por tercios)

| Tramo | Pasos con señal | Reward media | Reward (pasos con señal) | Truncados | KL |
|---|---|---|---|---|---|
| 1-50    | 52% | 0.325 | 0.433 | 61% | 0.00038 |
| 51-100  | 54% | 0.285 | 0.454 | 65% | 0.00038 |
| 101-150 | 64% | 0.290 | 0.422 | 66% | 0.00037 |
| **Total** | **57%** | **0.300** | **0.435** | **64%** | **0.00037** |

- 49/150 grupos todo-truncados (gradiente cero) · 8/150 grupos todo-correctos.
- grad_norm ~2-4×10⁻² en pasos con señal (sano tras el fix); is_ratio 0.97-0.99.
- Curvas: `grpo_curvas_piloto_nube.png`.

## Contra los criterios BIEN/MAL del plan

- ✅ KL suave y acotada (plana en 0.0004; con este lr no puede explotar).
- ✅ Longitud media estable (~5300); sin colapso de diversidad.
- ❌ Truncados 64% ≫ 30%. El remedio del plan (subir cap) choca con la VRAM:
  **conflicto plan-vs-realidad** — ver opciones abajo.
- ➖ "Reward sube aunque despacio": no observable todavía. No es señal de MAL
  (el plan reserva ese veredicto para 200+ pasos; aquí hubo ~85 efectivos).

## Smoke test AIME-30 (N=1, semilla 201, adaptador final)

- **7/30 = 23.3%** (baseline oficial: 20.44% ± 1.68 en AIME-90 × 5 semillas).
  Con n=30 y 1 semilla NO es concluyente (el plan lo define como detector de
  desastres): veredicto = **sin desastre**, ligeramente por encima del
  baseline. Corrió en local (3060, mismo arnés vLLM del baseline, ~1 h).
- **Auditado**: regrading completo de las 30 con el comparador del arnés →
  0 discrepancias. Crudos: `baseline_n1_aime_eval_90_smoke_piloto_nube_seed201.jsonl`.
- **Sin señales de reward-hacking**: los 7 aciertos son razonamientos de
  2257-6160 tokens, ninguno truncado, sin respuestas-atajo; el estilo de
  salida es indistinguible del base (esperable con KL ~0.0004).

## Recomendación (decisión de Alex)

El cuello de botella es el truncamiento (64%): un tercio de los pasos no
aporta nada. Opciones para el run largo, de más a menos recomendada:

1. **4090 + liger-kernel** (`use_liger_kernel`, pérdida GRPO chunkeada que no
   materializa los logits): debería permitir cap 8192-10240 en 24 GB →
   truncados ↓ mucho. Validar 3 pasos (~$0.05) antes. Coste run largo
   (1000-2000 pasos a ~60-75 s): **$6-14**.
2. Igual que el piloto (6144/G=8) asumiendo el 64%: funciona pero desperdicia
   un tercio del cómputo. $6-12.
3. Bajar temperatura de grupo 1.0 → 0.8 (menos divagar = menos truncado);
   desviación del plan (menos diversidad de grupo) — solo con OK explícito.
4. GPU más grande (A100-40GB, ~$0.9-1.3/h): cap 12-16K sin liger. $18-40.

En todos los casos: mismas puertas del plan (run largo solo con OK de Alex,
early-stop ante señal MAL sostenida, eval final PASO 6 con el mismo arnés
local del baseline).
