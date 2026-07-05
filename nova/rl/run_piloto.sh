#!/usr/bin/env bash
# run_piloto.sh — Lanzador robusto del piloto GRPO (PASO 4).
#
# Relanza train_grpo.py con --resume si muere (OOM/driver/corte), hasta
# MAX_REINTENTOS. Sube checkpoints a HF tras cada salida del proceso.
# Uso: bash nova/rl/run_piloto.sh [pasos] (por defecto 150)

set -u
PASOS="${1:-150}"
OUT="checkpoints/grpo_v1"
LOG="results/rl_local/grpo_piloto.log"
JSONL="results/rl_local/grpo_train_log.jsonl"
MAX_REINTENTOS=20
export HF_HUB_DOWNLOAD_TIMEOUT=30

cd "$(dirname "$0")/../.."

subir_ckpts() {
  timeout 900 hf upload Quantumadvancedai/nova-rl-ckpt "$OUT" piloto \
    --exclude "*.pt" --exclude "*optimizer*" --exclude "*scheduler*" \
    --commit-message "piloto GRPO: checkpoints (auto)" >> "$LOG" 2>&1 \
    && echo "subida a HF OK $(date '+%H:%M')" >> "$LOG" \
    || echo "AVISO: subida a HF fallida $(date '+%H:%M')" >> "$LOG"
}

intento=0
while [ "$intento" -le "$MAX_REINTENTOS" ]; do
  if [ "$intento" -eq 0 ] && [ ! -d "$OUT" ]; then
    echo "=== PILOTO inicio $(date '+%F %H:%M') pasos=$PASOS ===" >> "$LOG"
    timeout 172800 python nova/rl/train_grpo.py --max-steps "$PASOS" \
      --save-steps 25 --out "$OUT" --log-jsonl "$JSONL" >> "$LOG" 2>&1
  else
    echo "=== PILOTO reanudación #$intento $(date '+%F %H:%M') ===" >> "$LOG"
    timeout 172800 python nova/rl/train_grpo.py --max-steps "$PASOS" \
      --save-steps 25 --out "$OUT" --log-jsonl "$JSONL" --resume >> "$LOG" 2>&1
  fi
  ec=$?
  echo "=== salida exit=$ec $(date '+%F %H:%M') ===" >> "$LOG"
  subir_ckpts
  if [ "$ec" -eq 0 ]; then
    echo "PILOTO COMPLETADO" >> "$LOG"
    exit 0
  fi
  intento=$((intento + 1))
  sleep 60  # dejar respirar al driver antes de reintentar
done
echo "PILOTO ABORTADO: demasiados reintentos" >> "$LOG"
exit 1
