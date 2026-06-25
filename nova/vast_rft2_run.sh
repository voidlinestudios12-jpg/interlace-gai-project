#!/usr/bin/env bash
# Nova — Fase RFT ITERACIÓN 2 (datos pesados hacia difíciles + lr 1e-4). Para Vast.ai (H100).
# Ejecutar: bash nova/vast_rft2_run.sh   Variables: HF_TOKEN (oblig), GITHUB_TOKEN (push).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORK="/workspace"; LOG="$WORK/rft_run.log"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
# Config iteración 2
export RFT_GOLD_FILE="rft_dorado_v2.jsonl"
export RFT_LR="1e-4"
export RFT_EPOCHS="3"
export RFT_BETA_KL="0.05"
export RFT_MAX_SEQ="8192"
export RFT_GRAD_ACCUM="16"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Nova RFT ITER 2 (hard-weighted + lr=$RFT_LR)   $(date)"
echo "============================================================"
[ -z "${HF_TOKEN:-}" ] && { echo "[ERROR] falta HF_TOKEN"; exit 1; }

run_paso () { local n="$1"; shift; echo ""; echo ">>> $n  $(date)"; if "$@"; then echo "[$n] OK $(date)"; else echo "[$n] FALLO rc=$? $(date)"; return 1; fi; }

echo ">>> SETUP dependencias..."
pip install --quiet --upgrade pip
pip install --quiet "vllm>=0.6" || pip install --quiet vllm
pip install --quiet "transformers>=4.44" "peft>=0.11" "accelerate>=0.30" \
    "datasets>=2.18" huggingface_hub scikit-learn requests pandas tqdm
echo "[SETUP] OK"

# Reutilizar evaluaciones del BASE de iter1 (no cambian) + limpiar las del RFT viejo
echo ">>> reutilizando base eval + limpiando rft eval viejo..."
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
t=os.environ["HF_TOKEN"]; W="/workspace"
for fn in ["eval_base_gsm8k.jsonl","aime_gen.jsonl"]:
    try:
        hf_hub_download("Quantumadvancedai/nova-verif-data",fn,repo_type="dataset",token=t,local_dir=W)
        print("reusando",fn,flush=True)
    except Exception as e:
        print("aviso",fn,repr(e)[:80],flush=True)
PY
rm -f /workspace/eval_rft_aime.jsonl /workspace/eval_rft_gsm8k.jsonl /workspace/eval_rft_gpqa.jsonl /workspace/eval_base_gpqa.jsonl

run_paso "PASO 2 (dorado v2 hard)" python "$REPO_ROOT/nova/forge/vast_build_golden.py"
run_paso "PASO 3 (entrenar RFT v2)" python "$REPO_ROOT/nova/forge/vast_train_rft.py"
MODELO=rft BENCHES="aime:16,gsm8k:4,gpqa:4" run_paso "PASO 4a (eval RFT)" python "$REPO_ROOT/nova/inference/vast_eval_n1.py"
MODELO=base BENCHES="gpqa:4" run_paso "PASO 4b (eval BASE gpqa)" python "$REPO_ROOT/nova/inference/vast_eval_n1.py"
run_paso "PASO 4c (comparar)" python "$REPO_ROOT/nova/inference/vast_compare_n1.py"

echo ""; echo ">>> guardando en GitHub..."
cd "$REPO_ROOT"
git config user.email "vastai-bot@nova"; git config user.name "Nova RFT2 Bot"
git add docs/benchmarks/fase_rft/ 2>/dev/null
if git diff --cached --quiet; then echo "[GIT] sin cambios"; else
  git commit -m "fase RFT iter2: resultados modelo puro N=1 (hard-weighted + lr 1e-4)"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    RURL=$(git remote get-url origin)
    git remote set-url origin "$(echo "$RURL" | sed "s|https://github.com/|https://$GITHUB_TOKEN@github.com/|")"
    git push origin HEAD && echo "[GIT] push OK"; git remote set-url origin "$RURL"
  fi
fi
echo ""; echo "============================================================"
echo " RFT ITER 2 COMPLETADA  $(date)"; echo "============================================================"
cat "$REPO_ROOT/docs/benchmarks/fase_rft/report_rft_n1.md" 2>/dev/null
echo "[DONE_FINAL] $(date)"
