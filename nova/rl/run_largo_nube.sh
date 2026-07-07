#!/usr/bin/env bash
# run_largo_nube.sh — Run largo de GRPO (PASO 5) en NUBE, con OK de Alex.
#
# Config: liger-kernel (pérdida chunkeada -> cap 8192 en 24 GB), vLLM colocate,
# G=8, temp 1.0, lr 1e-6, β=0.04, LoRA r16 — igual que el piloto salvo el cap.
# Robustez: reintenta con --resume si muere, sube checkpoints+métricas a HF
# tras cada salida Y ADEMÁS cada 30 min con un sidecar (un run de ~30 h no
# puede depender de la subida final).
# Uso: bash nova/rl/run_largo_nube.sh [pasos]   (por defecto 1500)

set -u
PASOS="${1:-1500}"
OUT="checkpoints/grpo_v1_largo"
LOG="results/rl_nube/grpo_largo.log"
JSONL="results/rl_nube/grpo_largo_log.jsonl"
MAX_REINTENTOS=40
export HF_HUB_DOWNLOAD_TIMEOUT=30

cd "$(dirname "$0")/../.."
mkdir -p results/rl_nube

subir_a_hf() {
  timeout 900 hf upload Quantumadvancedai/nova-rl-ckpt "$OUT" largo \
    --exclude "*.pt" --exclude "*optimizer*" --exclude "*scheduler*" \
    --commit-message "run largo GRPO: checkpoints (auto)" >> "$LOG" 2>&1 \
    && echo "subida a HF OK $(date '+%H:%M')" >> "$LOG" \
    || echo "AVISO: subida a HF fallida $(date '+%H:%M')" >> "$LOG"
  timeout 300 hf upload Quantumadvancedai/nova-rl-ckpt "$JSONL" \
    largo/grpo_largo_log.jsonl \
    --commit-message "run largo GRPO: métricas (auto)" >/dev/null 2>&1 || true
}

# Sidecar: mientras exista el fichero centinela, sube métricas y checkpoints
# cada 30 min (si el host muere, lo último perdido son <30 min de curvas;
# los checkpoints son cada 50 pasos ≈ 60-75 min).
CENTINELA=/tmp/.run_largo_activo
touch "$CENTINELA"
(
  while [ -f "$CENTINELA" ]; do
    sleep 1800
    [ -f "$CENTINELA" ] && subir_a_hf
  done
) &
SIDECAR=$!

# vllm-mem 0.18: validado 2026-07-07 en 4090 (0.25 revienta el buffer NCCL
# de la generación con cap 8192; 0.18 pasa sin workarounds)
ENTRENA=(python nova/rl/train_grpo.py --max-steps "$PASOS" --save-steps 50
  --out "$OUT" --log-jsonl "$JSONL" --optim adamw_torch --use-liger
  --use-vllm --vllm-mem 0.18 --num-generations 8 --max-completion 8192
  --temp 0.8)

ec=1
intento=0
while [ "$intento" -le "$MAX_REINTENTOS" ]; do
  if [ "$intento" -eq 0 ] && [ ! -d "$OUT" ]; then
    echo "=== RUN LARGO inicio $(date '+%F %H:%M') pasos=$PASOS ===" >> "$LOG"
    timeout 172800 "${ENTRENA[@]}" >> "$LOG" 2>&1
  else
    echo "=== RUN LARGO reanudación #$intento $(date '+%F %H:%M') ===" >> "$LOG"
    timeout 172800 "${ENTRENA[@]}" --resume >> "$LOG" 2>&1
  fi
  ec=$?
  echo "=== salida exit=$ec $(date '+%F %H:%M') ===" >> "$LOG"
  subir_a_hf
  if [ "$ec" -eq 0 ]; then
    echo "RUN LARGO COMPLETADO" >> "$LOG"
    break
  fi
  intento=$((intento + 1))
  sleep 30
done
rm -f "$CENTINELA"
kill "$SIDECAR" 2>/dev/null
[ "$ec" -ne 0 ] && echo "RUN LARGO ABORTADO: demasiados reintentos" >> "$LOG"
exit "$ec"
