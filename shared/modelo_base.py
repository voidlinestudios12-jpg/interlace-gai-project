"""Carga compartida del modelo base (DeepSeek-R1-Distill-Qwen-1.5B) para Nova.

Centraliza: id del modelo, ajustes de muestreo ya validados, tokenizer + plantilla
de chat (SIN system prompt, igual que el arnes nova/eval/run_benchmark.py), y la
creacion del motor de inferencia vLLM. Se reutiliza desde la Fase 1 en adelante.
"""
import os

# Ajustes de entorno seguros (memoria GPU + evitar compilar kernels al vuelo en imagenes sin nvcc).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEMPERATURE = 0.6   # recomendacion oficial de DeepSeek-R1 (ya validada en el arnes)
TOP_P = 0.95


def cargar_tokenizer():
    """Tokenizer del modelo base."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID)


def mensajes(contenido: str):
    """Conversacion en formato chat, SIN system prompt (como el arnes), para llm.chat()."""
    return [{"role": "user", "content": contenido}]


def prompt_ids(tokenizer, contenido: str):
    """Token ids del prompt en formato chat (si se necesita la via de bajo nivel)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": contenido}], add_generation_prompt=True, tokenize=True
    )


def crear_llm(max_model_len: int = 36864, gpu_memory_utilization: float = 0.92,
              enable_lora: bool = False, max_lora_rank: int = 32):
    """Crea el motor vLLM con el modelo base en float16. Si enable_lora, admite
    cargar un adaptador LoRA (Nova-v1) en tiempo de inferencia."""
    from vllm import LLM
    kw = dict(model=MODEL_ID, dtype="float16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True)
    if enable_lora:
        kw["enable_lora"] = True
        kw["max_lora_rank"] = max_lora_rank
    return LLM(**kw)
