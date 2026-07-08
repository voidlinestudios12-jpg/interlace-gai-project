#!/usr/bin/env bash
# eval_final_rl.sh — PASO 6: evaluación final N=1 del adaptador RL (secuencial, local).
#
# 1) AIME-90 × semillas 101-105 con --lora (mismas semillas que el baseline oficial)
# 2) Anti-regresión: GSM8K-200 y GPQA-198, base y adaptador, semilla 101
# Todo reanudable (run_baseline_local.py hace flush por pregunta y salta lo hecho).
# Uso: bash nova/eval/eval_final_rl.sh <ruta_lora>

set -u
LORA="${1:?falta ruta del adaptador LoRA}"
LOG="results/rl_local/eval_final.log"
export HF_HUB_DOWNLOAD_TIMEOUT=30

cd "$(dirname "$0")/../.."

paso() { echo "=== $1 $(date '+%F %H:%M') ===" >> "$LOG"; }

# --- 1. AIME-90 x 5 semillas con adaptador ---
for seed in 101 102 103 104 105; do
  paso "AIME-90 rl seed $seed"
  python nova/eval/run_baseline_local.py --seed "$seed" \
    --lora "$LORA" --etiqueta rl_final >> "$LOG" 2>&1 \
    || { paso "FALLO aime seed $seed (reintento único)"; \
         python nova/eval/run_baseline_local.py --seed "$seed" \
           --lora "$LORA" --etiqueta rl_final >> "$LOG" 2>&1 \
           || { paso "ABORTADO aime seed $seed"; exit 1; }; }
done

# --- 2. Anti-regresión (base vs adaptador, semilla 101) ---
# GSM8K y GPQA son más cortos: max-tokens 8192 sobra y acelera.
for ds in "nova/data/gsm8k_eval_200.json num" "nova/data/gpqa_eval_198.json letra"; do
  set -- $ds
  nombre=$(basename "$1" .json)
  paso "anti-regresión $nombre BASE"
  python nova/eval/run_baseline_local.py --seed 101 --dataset "$1" --tipo "$2" \
    --max-tokens 8192 --etiqueta antireg_base >> "$LOG" 2>&1 \
    || { paso "ABORTADO $nombre base"; exit 1; }
  paso "anti-regresión $nombre RL"
  python nova/eval/run_baseline_local.py --seed 101 --dataset "$1" --tipo "$2" \
    --max-tokens 8192 --lora "$LORA" --etiqueta antireg_rl >> "$LOG" 2>&1 \
    || { paso "ABORTADO $nombre rl"; exit 1; }
done

paso "EVAL FINAL COMPLETA"
