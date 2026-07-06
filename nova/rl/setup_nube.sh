#!/usr/bin/env bash
# setup_nube.sh — Prepara una caja de Vast.ai para el piloto GRPO.
#
# Instala las MISMAS versiones que el entorno local `nova` (comparabilidad):
# torch 2.11.0 · trl 1.7.0 · transformers 5.12.1 · peft 0.19.1 ·
# bitsandbytes 0.49.2 · accelerate 1.14.0 · datasets 5.0.0 · vllm 0.24.0
#
# Uso (en la caja): GITHUB_PAT=... HF_TOKEN=... bash setup_nube.sh
set -euo pipefail
export HF_HUB_DOWNLOAD_TIMEOUT=30
export DEBIAN_FRONTEND=noninteractive

: "${GITHUB_PAT:?falta GITHUB_PAT}"
: "${HF_TOKEN:?falta HF_TOKEN}"

cd /workspace 2>/dev/null || cd ~

# Las imágenes *-runtime no traen compilador C y Triton (vLLM/torch.compile)
# compila un módulo al vuelo -> sin gcc revienta con "Failed to find C compiler"
if ! command -v gcc >/dev/null; then
  apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends gcc g++ >/dev/null
fi

if [ ! -d interlace-gai-project ]; then
  git clone "https://${GITHUB_PAT}@github.com/voidlinestudios12-jpg/interlace-gai-project.git"
fi
cd interlace-gai-project

pip install -q --upgrade pip
# vllm primero fija el torch compatible; el resto se resuelve en la misma pasada
pip install -q vllm==0.24.0 trl==1.7.0 transformers==5.12.1 peft==0.19.1 \
  bitsandbytes==0.49.2 accelerate==1.14.0 datasets==5.0.0

hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || hf auth login --token "$HF_TOKEN"
mkdir -p data/verif results/rl_nube checkpoints
hf download Quantumadvancedai/nova-verif-data rl_dataset_v1.jsonl \
  --repo-type dataset --local-dir data/verif
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B >/dev/null

echo "=== SETUP OK ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, trl, transformers, vllm, peft; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| trl', trl.__version__, '| tfm', transformers.__version__, '| vllm', vllm.__version__, '| peft', peft.__version__)"
