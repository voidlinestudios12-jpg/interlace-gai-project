"""Fase 2 — SFT con QLoRA (transformers + peft + bitsandbytes) sobre la base
CONGELADA -> adaptador Nova-v1. (Método estándar y robusto; alternativa a Unsloth.)

- Base en 4-bit (NF4) congelada; se entrenan solo adaptadores LoRA.
- DIA 0 ≡ BASE: LoRA arranca con B=0 (sin efecto) -> se verifica.
- Perdida SOLO sobre la respuesta del asistente (mascara manual del prompt -> -100).
- Guarda el adaptador en el volumen nova-data.

Validacion barata:  modal run nova/forge/sft.py --datos /data/sft/light_r1_n1000.jsonl --max-steps 30 --salida /data/adapters/_val_sft
Entrenamiento real: modal run --detach nova/forge/sft.py --datos /data/sft/light_r1_n3000.jsonl --epochs 3
"""
import modal

app = modal.App("nova-sft")
vol = modal.Volume.from_name("nova-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets")
)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@app.function(gpu="A10G", image=image, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 6)
def entrenar(datos, salida, epochs, max_steps, max_seq, r, lr):
    import os
    os.environ["HF_HOME"] = "/cache"
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=r, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.config.use_cache = False

    # --- DIA 0 ≡ BASE: LoRA arranca en identidad (matrices B a cero) ---
    maxB = max((float(p.abs().max()) for n, p in model.named_parameters() if "lora_B" in n), default=0.0)
    print(f"[SFT] max|lora_B| inicial = {maxB:.2e}  (≈0 => dia 0 ≡ base, no se rompe nada)", flush=True)

    ds = load_dataset("json", data_files=datos, split="train")

    def tokenizar(ej):
        msgs = ej["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        ids = tok(full, truncation=True, max_length=max_seq, add_special_tokens=False)["input_ids"]
        plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
        labels = list(ids)
        for i in range(min(plen, len(labels))):
            labels[i] = -100  # no se entrena sobre el enunciado, solo sobre la respuesta
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    ds = ds.map(tokenizar, remove_columns=ds.column_names)
    print(f"[SFT] ejemplos tokenizados: {len(ds)} | ejemplo len={len(ds[0]['input_ids'])} tokens", flush=True)

    args = TrainingArguments(
        per_device_train_batch_size=1, gradient_accumulation_steps=8, warmup_ratio=0.03,
        num_train_epochs=epochs, max_steps=max_steps, learning_rate=lr, logging_steps=5,
        optim="paged_adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine", seed=1234,
        output_dir="/data/sft/ckpt", bf16=True, report_to="none", save_strategy="no")
    collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)

    stats = trainer.train()
    loss = float(getattr(stats, "training_loss", 0.0))
    print(f"[SFT] loss final = {loss:.4f}", flush=True)

    model.save_pretrained(salida)
    tok.save_pretrained(salida)
    vol.commit()
    print(f"[SFT] adaptador Nova-v1 guardado en {salida}", flush=True)
    return {"salida": salida, "max_lora_B_inicial": maxB, "loss": loss, "n": len(ds)}


@app.local_entrypoint()
def main(datos: str = "/data/sft/light_r1_n1000.jsonl", salida: str = "/data/adapters/nova-v1-sft",
         epochs: float = 1.0, max_steps: int = -1, max_seq: int = 8192, r: int = 16, lr: float = 2e-4, gpu: str = "A10G"):
    fn = entrenar if gpu == "A10G" else entrenar.with_options(gpu=gpu)
    print(fn.remote(datos, salida, epochs, max_steps, max_seq, r, lr))
