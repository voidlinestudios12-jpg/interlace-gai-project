#!/usr/bin/env bash
# Nova — COMPLETAR iter2: faltaba el anti-regresión de GPQA (rft v2 + base) y la tabla final.
# Reutiliza de HF lo ya medido (eval_rft_aime/gsm8k v2, eval_base_gsm8k, aime_gen). El adaptador
# v2 ya está en HF nova-rft-v1. Genera solo GPQA (rft + base) -> compara -> reporte -> push.
# Variables: HF_TOKEN (oblig), GITHUB_TOKEN (push).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORK="/workspace"; LOG="$WORK/rft_run.log"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo " Nova iter2 FINISH (GPQA anti-regresión + tabla)   $(date)"
echo "============================================================"
[ -z "${HF_TOKEN:-}" ] && { echo "[ERROR] falta HF_TOKEN"; exit 1; }
run_paso () { local n="$1"; shift; echo ""; echo ">>> $n  $(date)"; if "$@"; then echo "[$n] OK $(date)"; else echo "[$n] FALLO rc=$? $(date)"; return 1; fi; }

echo ">>> SETUP..."
pip install --quiet --upgrade pip
pip install --quiet "vllm>=0.6" || pip install --quiet vllm
pip install --quiet "transformers>=4.44" "peft>=0.11" "accelerate>=0.30" huggingface_hub requests pandas tqdm
echo "[SETUP] OK"

# Generar SOLO lo que falta: GPQA del rft v2 y del base
MODELO=rft  BENCHES="gpqa:4" run_paso "EVAL RFT gpqa"  python "$REPO_ROOT/nova/inference/vast_eval_n1.py"
MODELO=base BENCHES="gpqa:4" run_paso "EVAL BASE gpqa" python "$REPO_ROOT/nova/inference/vast_eval_n1.py"
# Tabla final (descarga de HF aime/gsm8k de rft v2 + base)
run_paso "COMPARAR" python "$REPO_ROOT/nova/inference/vast_compare_n1.py"

echo ""; echo ">>> GitHub..."
cd "$REPO_ROOT"
git config user.email "vastai-bot@nova"; git config user.name "Nova Bot"
git add docs/benchmarks/fase_rft/ 2>/dev/null
if git diff --cached --quiet; then echo "[GIT] sin cambios"; else
  git commit -m "fase RFT iter2: tabla final completa con GPQA anti-regresion"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    RURL=$(git remote get-url origin)
    git remote set-url origin "$(echo "$RURL" | sed "s|https://github.com/|https://$GITHUB_TOKEN@github.com/|")"
    git push origin HEAD && echo "[GIT] push OK"; git remote set-url origin "$RURL"
  fi
fi
echo ""; echo "==== TABLA FINAL ===="; cat "$REPO_ROOT/docs/benchmarks/fase_rft/report_rft_n1.md" 2>/dev/null
echo "[DONE_FINAL] $(date)"
