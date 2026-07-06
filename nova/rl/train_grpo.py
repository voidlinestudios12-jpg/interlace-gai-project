"""
train_grpo.py — PASO 3/4 de la fase RL: GRPO con LoRA sobre la base congelada.

MODO ROBUSTO (use_vllm=False): generación con transformers, UNA sola copia del
modelo en VRAM -> imposible OOM por duplicación. Lento; da igual (3060, 11 GB).

Config del plan: LoRA r=16 alpha=32 (q/k/v/o + MLP), base congelada bf16,
lr 1e-6 warmup 10, KL beta 0.04, grupo G=6, max_completion 4096, temperatura
de grupo 1.0 / top_p 0.95, 1 prompt por paso (batch 1 x grad_accum 6),
gradient checkpointing, AdamW 8-bit. API verificada contra TRL 1.7.0.

Recompensa (anti reward-hacking, simple y verificable):
  - truncada o sin \\boxed{} -> 0.0
  - correcta (extractor + comparador del arnés) -> 1.0
  - +0.1 si formato correcto (UN solo \\boxed{} y terminada con EOS)

Logging por paso a jsonl (flush+fsync) con las señales de vigilancia del plan:
reward, kl, longitud media, % truncados, % grupos sin señal.

Uso:
    python nova/rl/train_grpo.py --max-steps 300            # piloto
    python nova/rl/train_grpo.py --max-steps 300 --resume   # reanudar
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nova" / "eval"))
from run_benchmark import extraer_num, comparar_num  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET = REPO_ROOT / "data" / "verif" / "rl_dataset_v1.jsonl"


def parsear_args():
    p = argparse.ArgumentParser(description="GRPO + LoRA en modo robusto (TRL 1.7.0)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", default="checkpoints/grpo_v1")
    p.add_argument("--resume", action="store_true",
                   help="reanuda del último checkpoint de --out")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--num-generations", type=int, default=6)
    p.add_argument("--max-completion", type=int, default=4096)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-jsonl", default="results/rl_local/grpo_train_log.jsonl")
    # --- nube (VRAM grande): vLLM colocate + batch configurable ---
    p.add_argument("--use-vllm", action="store_true",
                   help="generación con vLLM en modo colocate (nube, 24+ GB)")
    p.add_argument("--vllm-mem", type=float, default=0.40,
                   help="fracción de VRAM para vLLM en colocate")
    p.add_argument("--vllm-sleep", action="store_true",
                   help="dormir vLLM durante la fase de entrenamiento (más margen)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=0,
                   help="0 = num_generations/batch_size (1 prompt por paso)")
    p.add_argument("--optim", default="adamw_bnb_8bit",
                   help="local: adamw_bnb_8bit (11 GB) · nube: adamw_torch "
                        "(con LoRA el ahorro de 8-bit es ~150 MB y bnb es frágil)")
    p.add_argument("--use-liger", action="store_true",
                   help="pérdida GRPO chunkeada de liger-kernel: no materializa "
                        "los logits (L,152k) -> permite cap 8192+ en 24 GB. "
                        "Compatible con nuestra LoRA (no toca lm_head) y con la "
                        "corrección IS de vLLM (pasa vllm_is_ratio).")
    return p.parse_args()


def cargar_dataset_rl(seed):
    """rl_dataset_v1.jsonl -> Dataset con prompt conversacional + gold.
    Mismo prompt que en evaluación (sin system prompt, oficial DeepSeek)."""
    from datasets import Dataset
    filas = []
    with open(DATASET, encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                d = json.loads(linea)
                filas.append({
                    "prompt": [{"role": "user", "content":
                                d["problema"] + "\n\nPlease reason step by step, and "
                                "put your final answer within \\boxed{}."}],
                    "gold": str(d["gold"]).strip(),
                })
    return Dataset.from_list(filas).shuffle(seed=seed)


def texto_de(completion):
    """Texto de una completion (formato conversacional o cadena)."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def crear_reward(max_completion, eos_ids):
    """Recompensa verificable del plan. Cerrada sobre el cap de longitud y los
    ids de EOS para detectar truncamiento desde completion_ids."""

    def reward_exactitud(prompts, completions, completion_ids, gold, **kwargs):
        recompensas = []
        for comp, ids, g in zip(completions, completion_ids, gold):
            texto = texto_de(comp)
            truncada = len(ids) >= max_completion and (len(ids) == 0 or ids[-1] not in eos_ids)
            n_boxed = texto.count("\\boxed{")
            if truncada or n_boxed == 0:
                recompensas.append(0.0)
                continue
            r = 0.0
            pred = extraer_num(texto)
            if comparar_num(pred, g):
                r += 1.0
            if n_boxed == 1:
                r += 0.1  # formato correcto: razonamiento + UN \boxed{}
            recompensas.append(r)
        return recompensas

    return reward_exactitud


def crear_callback_jsonl(ruta):
    """Callback que escribe cada log del trainer a jsonl con flush+fsync
    (sobrevive a cortes). Señales del plan: reward, kl, longitud, % truncados."""
    from transformers import TrainerCallback

    class LogJsonlCallback(TrainerCallback):
        def __init__(self):
            self.ruta = Path(ruta)
            self.ruta.parent.mkdir(parents=True, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            registro = {"step": state.global_step,
                        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            registro.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
            with open(self.ruta, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    return LogJsonlCallback()


def main():
    args = parsear_args()
    # OJO: expandable_segments (que run_benchmark.py pone al importarse) usa la
    # API VMM de CUDA, rota en WSL2 -> "CUDA driver error: device not ready" en
    # el backward. En Linux real (nube) sí funciona y reduce la fragmentación
    # (clave con secuencias de 8K: el pico fp32 de los logits es de ~5 GB).
    import platform
    if "microsoft" in platform.uname().release.lower():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""  # WSL2: alocador por defecto
    else:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eos_ids = {tokenizer.eos_token_id}
    dataset = cargar_dataset_rl(args.seed)
    print(f"Dataset RL: {len(dataset)} problemas | pasos: {args.max_steps} | "
          f"G={args.num_generations} | max_completion={args.max_completion}", flush=True)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    grad_accum = args.grad_accum or max(1, args.num_generations // args.batch_size)
    vllm_kwargs = {}
    if args.use_vllm:
        vllm_kwargs = dict(
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_mem,
            vllm_max_model_length=768 + args.max_completion,
            vllm_enable_sleep_mode=args.vllm_sleep,
            # OJO: el modo por defecto (sequence_mask) suma la discrepancia
            # trainer-vs-vLLM (~0.02 nats/token en bf16) sobre TODA la secuencia:
            # con miles de tokens el ratio se hunde a e^-100 y multiplica el
            # gradiente por ~0 -> el modelo no aprende NADA (visto en la 4090,
            # 29 pasos con grad_norm=0). token_truncate = TIS estándar: ratio
            # por token ~1, truncado en clip_max.
            vllm_importance_sampling_mode="token_truncate",
        )

    out_dir = str(REPO_ROOT / args.out)
    config = GRPOConfig(
        output_dir=out_dir,
        max_steps=args.max_steps,
        seed=args.seed,
        bf16=True,
        # --- GRPO ---
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=1.0,          # diversidad del grupo
        top_p=0.95,
        beta=args.beta,           # KL que ancla a la base
        # loss_type por defecto ("dapo"): agregación GRPO sin sesgo de longitud
        # --- memoria ---
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        use_liger_kernel=args.use_liger,
        use_vllm=args.use_vllm,   # local: robusto (False) · nube: colocate
        **vllm_kwargs,
        # --- optimización ---
        learning_rate=args.lr,
        warmup_steps=10,
        lr_scheduler_type="constant_with_warmup",
        max_grad_norm=1.0,
        # --- robustez / logging ---
        save_steps=args.save_steps,
        save_strategy="steps",
        logging_steps=1,
        report_to="none",
        # WSL2 + 11 GB: vaciar la cache de CUDA cada paso reduce fragmentación
        # (el pico del backward con logits (L,152k) roza el límite de VRAM)
        torch_empty_cache_steps=1,
        dataloader_num_workers=0,  # RAM WSL2 limitada
        # OJO: la clave es `dtype` (transformers 5.x); sin ella carga float32
        model_init_kwargs={"dtype": torch.bfloat16},
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=crear_reward(args.max_completion, eos_ids),
        args=config,
        train_dataset=dataset,
        peft_config=lora,
        callbacks=[crear_callback_jsonl(REPO_ROOT / args.log_jsonl)],
    )

    resume = False
    if args.resume:
        ckpts = sorted(Path(out_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume = str(ckpts[-1])
            print(f"Reanudando desde {resume}", flush=True)
        else:
            print("No hay checkpoints; empezando de cero.", flush=True)

    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(os.path.join(out_dir, "final"))
    print("ENTRENAMIENTO COMPLETADO", flush=True)


if __name__ == "__main__":
    main()
