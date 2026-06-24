#!/usr/bin/env bash
# Nova — Fase RFT (modelo puro a N=1). Orquestador para Vast.ai (RTX 4090).
# Ejecutar: bash nova/vast_rft_run.sh
# Variables: HF_TOKEN (obligatoria), GITHUB_TOKEN (para push final).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORK="/workspace"
LOG="$WORK/rft_run.log"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Nova — Fase RFT (modelo PURO a N=1)   $(date)"
echo " REPO_ROOT=$REPO_ROOT"
echo "============================================================"
[ -z "${HF_TOKEN:-}" ] && { echo "[ERROR] falta HF_TOKEN"; exit 1; }

run_paso () {  # nombre, comando...  (no aborta el script si un paso falla; lo registra)
  local nombre="$1"; shift
  echo ""; echo ">>> $nombre  $(date)"
  if "$@"; then echo "[$nombre] OK $(date)"; else echo "[$nombre] FALLO (rc=$?) $(date)"; return 1; fi
}

# ---------------- SETUP ----------------
echo ">>> SETUP dependencias..."
pip install --quiet --upgrade pip
pip install --quiet "vllm>=0.6" || pip install --quiet vllm
pip install --quiet "transformers>=4.44" "peft>=0.11" "bitsandbytes>=0.43" \
    "accelerate>=0.30" "datasets>=2.18" huggingface_hub scikit-learn requests pandas tqdm
echo "[SETUP] OK"

# ---------------- PASO 2: dataset dorado (verificador) ----------------
run_paso "PASO 2 (dataset dorado)" python "$REPO_ROOT/nova/forge/vast_build_golden.py"

# ---------------- PASO 3: entrenar RFT ----------------
run_paso "PASO 3 (entrenar RFT)" python "$REPO_ROOT/nova/forge/vast_train_rft.py"

# ---------------- PASO 4a: medir RFT (aime + anti-regresion) ----------------
MODELO=rft BENCHES="aime:16,gsm8k:4,gpqa:4" run_paso "PASO 4a (eval RFT)" \
    python "$REPO_ROOT/nova/inference/vast_eval_n1.py"

# ---------------- PASO 4b: medir BASE (gsm8k + gpqa; AIME base se reutiliza de aime_gen) ----------------
MODELO=base BENCHES="gsm8k:4,gpqa:4" run_paso "PASO 4b (eval BASE)" \
    python "$REPO_ROOT/nova/inference/vast_eval_n1.py"

# ---------------- PASO 4c: comparar y veredicto ----------------
run_paso "PASO 4c (comparar)" python "$REPO_ROOT/nova/inference/vast_compare_n1.py"

# ---------------- GUARDAR EN GITHUB ----------------
echo ""; echo ">>> guardando en GitHub..."
cd "$REPO_ROOT"
git config user.email "vastai-bot@nova" ; git config user.name "Nova RFT Bot"
git add docs/benchmarks/fase_rft/ 2>/dev/null
if git diff --cached --quiet; then
  echo "[GIT] sin cambios"
else
  git commit -m "fase RFT: resultados modelo puro N=1 (AIME + anti-regresion)"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    RURL=$(git remote get-url origin)
    git remote set-url origin "$(echo "$RURL" | sed "s|https://github.com/|https://$GITHUB_TOKEN@github.com/|")"
    git push origin HEAD && echo "[GIT] push OK"
    git remote set-url origin "$RURL"
  fi
fi

echo ""; echo "============================================================"
echo " FASE RFT COMPLETADA  $(date)"
echo "============================================================"
cat "$REPO_ROOT/docs/benchmarks/fase_rft/report_rft_n1.md" 2>/dev/null
echo ""; echo ">>> RECUERDA destruir la instancia Vast.ai <<<"
echo "[DONE_FINAL] $(date)"
