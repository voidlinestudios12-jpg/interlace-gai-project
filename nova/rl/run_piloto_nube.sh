#!/usr/bin/env bash
# run_piloto_nube.sh — Lanzador robusto del piloto GRPO en NUBE (Vast.ai, 24 GB).
#
# Config de nube (aprovecha la VRAM grande): vLLM colocate, G=8, cap 8192
# (menos truncados = más señal; en la 3060 el cap 4096 daba ~90% de pasos
# con gradiente cero). Igual que run_piloto.sh: relanza con --resume si muere
# y sube checkpoints + métricas a HF tras cada salida.
# Uso: bash nova/rl/run_piloto_nube.sh [pasos]   (por defecto 150)

set -u
PASOS="${1:-150}"
OUT="checkpoints/grpo_v1_nube"
LOG="results/rl_nube/grpo_piloto.log"
JSONL="results/rl_nube/grpo_train_log.jsonl"
MAX_REINTENTOS=20
export HF_HUB_DOWNLOAD_TIMEOUT=30

cd "$(dirname "$0")/../.."
mkdir -p results/rl_nube

subir_a_hf() {
  timeout 900 hf upload Quantumadvancedai/nova-rl-ckpt "$OUT" piloto_nube \
    --exclude "*.pt" --exclude "*optimizer*" --exclude "*scheduler*" \
    --commit-message "piloto GRPO nube: checkpoints (auto)" >> "$LOG" 2>&1 \
    && echo "subida a HF OK $(date '+%H:%M')" >> "$LOG" \
    || echo "AVISO: subida a HF fallida $(date '+%H:%M')" >> "$LOG"
  # métricas también a HF: permite vigilar desde local aunque el SSH caiga
  timeout 300 hf upload Quantumadvancedai/nova-rl-ckpt "$JSONL" \
    piloto_nube/grpo_train_log.jsonl \
    --commit-message "piloto GRPO nube: métricas (auto)" >/dev/null 2>&1 || true
}

# Config validada en 4090 24 GB (2026-07-06): cap 8192 y vllm 0.40 dan OOM;
# 6144 + vllm 0.20 + adamw_torch caben (pico ~21 GB). bnb 8-bit roto en la
# caja (libnvJitLink) y además innecesario con LoRA.
ENTRENA=(python nova/rl/train_grpo.py --max-steps "$PASOS" --save-steps 25
  --out "$OUT" --log-jsonl "$JSONL" --optim adamw_torch
  --use-vllm --vllm-mem 0.20 --num-generations 8 --max-completion 6144)

intento=0
while [ "$intento" -le "$MAX_REINTENTOS" ]; do
  if [ "$intento" -eq 0 ] && [ ! -d "$OUT" ]; then
    echo "=== PILOTO NUBE inicio $(date '+%F %H:%M') pasos=$PASOS ===" >> "$LOG"
    timeout 86400 "${ENTRENA[@]}" >> "$LOG" 2>&1
  else
    echo "=== PILOTO NUBE reanudación #$intento $(date '+%F %H:%M') ===" >> "$LOG"
    timeout 86400 "${ENTRENA[@]}" --resume >> "$LOG" 2>&1
  fi
  ec=$?
  echo "=== salida exit=$ec $(date '+%F %H:%M') ===" >> "$LOG"
  subir_a_hf
  if [ "$ec" -eq 0 ]; then
    echo "PILOTO COMPLETADO" >> "$LOG"
    exit 0
  fi
  intento=$((intento + 1))
  sleep 30
done
echo "PILOTO ABORTADO: demasiados reintentos" >> "$LOG"
exit 1
