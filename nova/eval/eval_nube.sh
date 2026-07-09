#!/usr/bin/env bash
# eval_nube.sh — PASO 6 en NUBE (4090): evalúa RL y BASE con el mismo arnés.
#
# Para una comparación justa en hardware nuevo se re-mide el baseline en la
# MISMA caja (regla del plan: ante la duda, re-mide ambos en el mismo sitio).
# Reanudable: run_baseline_local.py salta las preguntas ya hechas.
#
# Uso: bash nova/eval/eval_nube.sh "101 102 103" [si|no: anti-regresión]
set -u
SEEDS="${1:?lista de semillas, p.ej. \"101 102 103\"}"
ANTIREG="${2:-no}"
LORA="checkpoints/grpo_v1_largo/final"
LOG="results/rl_nube/eval_nube.log"
export HF_HUB_DOWNLOAD_TIMEOUT=30

cd "$(dirname "$0")/../.."
mkdir -p results/rl_nube results/rl_local checkpoints
paso() { echo "=== $1 $(date '+%F %H:%M') ===" >> "$LOG"; }

# Adaptador final desde HF si no está en disco
if [ ! -f "$LORA/adapter_model.safetensors" ]; then
  paso "descargando adaptador de HF"
  hf download Quantumadvancedai/nova-rl-ckpt --include "largo/final/*" \
    --local-dir checkpoints/_hf >> "$LOG" 2>&1
  mkdir -p "$(dirname "$LORA")"
  cp -r checkpoints/_hf/largo/final "$LORA"
fi

# AIME-90: primero todas las semillas con adaptador, luego base
for seed in $SEEDS; do
  paso "AIME rl seed $seed"
  python nova/eval/run_baseline_local.py --seed "$seed" --lote 24 \
    --lora "$LORA" --etiqueta rl_final_nube >> "$LOG" 2>&1 \
    || { paso "FALLO rl seed $seed"; exit 1; }
done
for seed in $SEEDS; do
  paso "AIME base seed $seed"
  python nova/eval/run_baseline_local.py --seed "$seed" --lote 24 \
    --etiqueta base_nube >> "$LOG" 2>&1 \
    || { paso "FALLO base seed $seed"; exit 1; }
done

# Anti-regresión (GSM8K + GPQA, base y RL, semilla 101).
# OJO: gpqa_eval_198.json es gated y NO está en el repo -> scp previo.
if [ "$ANTIREG" = "si" ]; then
  for par in "nova/data/gsm8k_eval_200.json num" "nova/data/gpqa_eval_198.json letra"; do
    set -- $par
    nombre=$(basename "$1" .json)
    paso "antireg $nombre base"
    python nova/eval/run_baseline_local.py --seed 101 --dataset "$1" --tipo "$2" \
      --max-tokens 8192 --lote 24 --etiqueta antireg_base_nube >> "$LOG" 2>&1 \
      || { paso "FALLO $nombre base"; exit 1; }
    paso "antireg $nombre rl"
    python nova/eval/run_baseline_local.py --seed 101 --dataset "$1" --tipo "$2" \
      --max-tokens 8192 --lote 24 --lora "$LORA" --etiqueta antireg_rl_nube >> "$LOG" 2>&1 \
      || { paso "FALLO $nombre rl"; exit 1; }
  done
fi

paso "EVAL NUBE COMPLETA"
