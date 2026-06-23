#!/usr/bin/env bash
# Nova Fase 3 — orquestador completo para Vast.ai
# Ejecutar: bash nova/vast_run.sh
# Variables requeridas: HF_TOKEN (y opcionalmente GITHUB_TOKEN para push final)
# Se ejecuta desde la raíz del repo clonado (/workspace/nova-repo o similar).

set -e  # abortar en cualquier error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORK="/workspace"
LOG="$WORK/vast_run.log"

# Redirigir stdout y stderr al log (y también a consola)
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Nova Fase 3 — Vast.ai Pipeline"
echo " $(date)"
echo " SCRIPT_DIR=$SCRIPT_DIR"
echo " REPO_ROOT=$REPO_ROOT"
echo "============================================================"

# Verificar variables de entorno requeridas
if [ -z "$HF_TOKEN" ]; then
  echo "[ERROR] Falta HF_TOKEN"
  exit 1
fi

# ---------------------------------------------------------------------------
# SETUP: instalar dependencias
# ---------------------------------------------------------------------------
echo ""
echo ">>> SETUP: instalando dependencias..."

pip install --quiet --upgrade pip

# vLLM (generación) — nota: no compatible con bitsandbytes en mismo proceso
pip install --quiet "vllm>=0.4.0" || pip install --quiet vllm

# bitsandbytes + peft + transformers (entrenamiento)
pip install --quiet "transformers>=4.40" "peft>=0.10" "bitsandbytes>=0.43" \
    "accelerate>=0.28" "datasets>=2.18" huggingface_hub scikit-learn tqdm

echo "[SETUP] dependencias instaladas"

# ---------------------------------------------------------------------------
# PASO 1: generar soluciones NuminaMath restantes (416 problemas)
# ---------------------------------------------------------------------------
echo ""
echo ">>> PASO 1/5: generando NuminaMath (416 problemas × K=8)..."
echo "    Estimado: ~1-2h en RTX 3090"
python "$REPO_ROOT/nova/forge/vast_gen_numina.py"
echo "[PASO 1] COMPLETADO $(date)"

# ---------------------------------------------------------------------------
# PASO 2: construir dataset v2 (v1 + numina mixtos)
# ---------------------------------------------------------------------------
echo ""
echo ">>> PASO 2/5: construyendo verif_dataset_v2..."
python "$REPO_ROOT/nova/forge/vast_build_v2.py"
echo "[PASO 2] COMPLETADO $(date)"

# ---------------------------------------------------------------------------
# PASO 3: entrenar ORM (QLoRA 4-bit)
# ---------------------------------------------------------------------------
echo ""
echo ">>> PASO 3/5: entrenando ORM verificador (QLoRA, ~1-2h)..."
echo "    Nota: bitsandbytes activo — proceso separado de vLLM"
python "$REPO_ROOT/nova/forge/vast_train_orm.py"
echo "[PASO 3] COMPLETADO $(date)"

# ---------------------------------------------------------------------------
# PASO 4: generar soluciones AIME 2023+2024+2025 (N=64)
# ---------------------------------------------------------------------------
echo ""
echo ">>> PASO 4/5: generando AIME 2023+2024+2025 (K=64 por problema)..."
echo "    Estimado: ~30-60min en RTX 3090"
python "$REPO_ROOT/nova/inference/vast_gen_aime.py"
echo "[PASO 4] COMPLETADO $(date)"

# ---------------------------------------------------------------------------
# PASO 5: puntuar con ORM + generar tabla final
# ---------------------------------------------------------------------------
echo ""
echo ">>> PASO 5/5: puntuando con ORM y generando tabla..."
python "$REPO_ROOT/nova/inference/vast_score_aime.py"
echo "[PASO 5] COMPLETADO $(date)"

# ---------------------------------------------------------------------------
# GUARDAR RESULTADOS EN GITHUB
# ---------------------------------------------------------------------------
echo ""
echo ">>> Guardando resultados en GitHub..."

REPORT_MD="$WORK/report_verif_eval.md"
REPORT_JSON="$WORK/report_verif_eval.json"
DEST_DIR="$REPO_ROOT/docs/benchmarks/fase3_prm"

mkdir -p "$DEST_DIR"
cp "$REPORT_MD" "$DEST_DIR/report_verif_eval.md"
cp "$REPORT_JSON" "$DEST_DIR/report_verif_eval.json"

cd "$REPO_ROOT"

git config user.email "vastai-bot@nova-project"
git config user.name "Nova Vast.ai Bot"
git add "docs/benchmarks/fase3_prm/"

if git diff --cached --quiet; then
  echo "[GIT] Sin cambios nuevos que commitear"
else
  git commit -m "fase3: resultados ORM verificador AIME 2023+2024+2025

Pipeline completo ejecutado en Vast.ai RTX 3090.
Archivos: report_verif_eval.md, report_verif_eval.json"

  if [ -n "$GITHUB_TOKEN" ]; then
    # Configurar remote con PAT para poder hacer push
    REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
    if echo "$REMOTE_URL" | grep -q "github.com"; then
      # Insertar token en URL
      AUTHED_URL=$(echo "$REMOTE_URL" | sed "s|https://github.com/|https://$GITHUB_TOKEN@github.com/|")
      git remote set-url origin "$AUTHED_URL"
      git push origin HEAD
      echo "[GIT] push a GitHub OK"
      # Limpiar token de la URL remota después del push (seguridad)
      git remote set-url origin "$REMOTE_URL"
    else
      echo "[GIT] aviso: remote no es GitHub: $REMOTE_URL"
    fi
  else
    echo "[GIT] aviso: GITHUB_TOKEN no definido, no se hace push automático"
  fi
fi

# ---------------------------------------------------------------------------
# RESUMEN FINAL
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " PIPELINE COMPLETADO"
echo " $(date)"
echo "============================================================"
echo ""
echo "Reporte:"
cat "$DEST_DIR/report_verif_eval.md"
echo ""
echo "Log completo en: $LOG"
echo ""
echo ">>> Recuerda ELIMINAR la instancia Vast.ai para no gastar créditos <<<"
