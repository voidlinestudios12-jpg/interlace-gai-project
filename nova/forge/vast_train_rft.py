"""VAST.AI — RFT PASO 3: entrenar el modelo PURO con rejection-sampling FT.

Entrena un adaptador LoRA sobre la base CONGELADA con el dataset dorado
(rft_dorado_v1.jsonl). El objetivo es mejorar el modelo a N=1 (una sola muestra),
no best-of-N.

DISCIPLINA (todo lo que falló en el SFT anterior, arreglado):
  - FORMATO EXACTO de generación: prompt = apply_chat_template([user],
    add_generation_prompt=True) -> termina en `<|Assistant|><think>\\n`. Luego
    se añade la solución (que ya empieza tras ese <think>) + EOS.  ⇒ UN solo
    <think>. (El bug anterior pasaba la respuesta por la plantilla y duplicaba
    <think>.)
  - Base CONGELADA, solo LoRA. lr BAJO (1e-5). 1-2 épocas.
  - ANCLA KL contra la base (adaptador desactivado) en los tokens de la
    respuesta, para no olvidar lo que ya sabe.
  - Día 0 ≡ base: LoRA arranca con B=0 (se verifica).

Variables: HF_TOKEN. Args por entorno: RFT_EPOCHS, RFT_LR, RFT_BETA_KL, RFT_R.
Salida: adaptador en /workspace/nova_rft_v1 + subida a HF (nova-rft-v1).
"""
import os, sys, json

HF_REPO_DATA = "Quantumadvancedai/nova-verif-data"
HF_REPO_MODEL = "Quantumadvancedai/nova-rft-v1"
TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "Falta HF_TOKEN"
WORK = "/workspace"
SALIDA = "/workspace/nova_rft_v1"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
BASE_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Hiperparámetros (con disciplina)
EPOCHS = float(os.environ.get("RFT_EPOCHS", "2"))
LR = float(os.environ.get("RFT_LR", "1e-5"))
BETA_KL = float(os.environ.get("RFT_BETA_KL", "0.1"))
LORA_R = int(os.environ.get("RFT_R", "16"))
MAX_SEQ = int(os.environ.get("RFT_MAX_SEQ", "4096"))
GRAD_ACCUM = int(os.environ.get("RFT_GRAD_ACCUM", "16"))
SEED = 1234
SUFIJO = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

from huggingface_hub import HfApi, hf_hub_download, create_repo
api = HfApi(token=TOKEN)


def _baja(fn):
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        return p
    return hf_hub_download(HF_REPO_DATA, fn, repo_type="dataset", token=TOKEN, local_dir=WORK)


gold_path = _baja("rft_dorado_v1.jsonl")
filas = [json.loads(l) for l in open(gold_path, encoding="utf-8") if l.strip()]
print(f"[RFT] {len(filas)} problemas dorados cargados", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tok = AutoTokenizer.from_pretrained(BASE_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def construir(problema, solucion):
    """Tokens EXACTOS como en generación: prompt(chat, gen_prompt) + solucion + EOS.
    labels enmascara el prompt (-100). Devuelve None si excede MAX_SEQ."""
    user = problema + SUFIJO
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=True, return_dict=False)
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]
    sol_ids = tok(solucion, add_special_tokens=False).input_ids
    eos = tok.eos_token_id
    input_ids = prompt_ids + sol_ids + [eos]
    if len(input_ids) > MAX_SEQ:
        return None
    labels = [-100] * len(prompt_ids) + sol_ids + [eos]
    return {"input_ids": input_ids, "labels": labels, "n_prompt": len(prompt_ids)}


ejemplos = []
saltados = 0
for r in filas:
    e = construir(r["problema"], r["solucion"])
    if e is None:
        saltados += 1
    else:
        ejemplos.append(e)
print(f"[RFT] ejemplos usables: {len(ejemplos)} (saltados por longitud>{MAX_SEQ}: {saltados})", flush=True)
assert ejemplos, "0 ejemplos usables"

# Verificación de formato en 1 ejemplo (que no haya doble <think>)
muestra = tok.decode(ejemplos[0]["input_ids"][:ejemplos[0]["n_prompt"] + 5])
n_think = muestra.count("<think>")
print(f"[RFT] CHEQUEO FORMATO: '<think>' aparece {n_think} vez/veces en el prompt+inicio "
      f"(debe ser 1). Inicio respuesta: ...{muestra[-60:]!r}", flush=True)
assert n_think == 1, "FORMATO MAL: <think> no aparece exactamente 1 vez"

longs = sorted(len(e["input_ids"]) for e in ejemplos)
print(f"[RFT] longitud tokens: min={longs[0]} mediana={longs[len(longs)//2]} max={longs[-1]}", flush=True)


class DS(Dataset):
    def __init__(self, ej): self.ej = ej
    def __len__(self): return len(self.ej)
    def __getitem__(self, i):
        e = self.ej[i]
        return {"input_ids": e["input_ids"], "labels": e["labels"],
                "attention_mask": [1] * len(e["input_ids"])}


def collate(batch):
    maxl = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    ids, lab, att = [], [], []
    for b in batch:
        n = maxl - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad] * n)
        lab.append(b["labels"] + [-100] * n)
        att.append(b["attention_mask"] + [0] * n)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att)}


# --------------------------------------------------------------------------
# Modelo: base 4-bit congelada + LoRA
# --------------------------------------------------------------------------
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_ID, quantization_config=bnb, dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0})
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.config.use_cache = False
model.print_trainable_parameters()

# Día 0 ≡ base
maxB = max((float(p.abs().max()) for n, p in model.named_parameters() if "lora_B" in n), default=0.0)
print(f"[RFT] max|lora_B| inicial = {maxB:.2e}  (≈0 => día 0 ≡ base)", flush=True)
assert maxB < 1e-6, "LoRA B no arranca en 0"


def masked_kl(logits, ref_logits, mask, chunk=512):
    """KL(policy||ref) media sobre los tokens de respuesta (mask=True), por trozos
    para no reventar memoria. logits/ref: [B,T,V] (ya desplazados); mask: [B,T]."""
    B, T, V = logits.shape
    sl = logits.reshape(B * T, V)
    rl = ref_logits.reshape(B * T, V)
    idx = mask.reshape(B * T).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return logits.new_zeros(())
    total = logits.new_zeros(())
    for c in range(0, idx.numel(), chunk):
        sel = idx[c:c + chunk]
        lp = F.log_softmax(sl[sel].float(), dim=-1)
        rp = F.log_softmax(rl[sel].float(), dim=-1)
        total = total + (lp.exp() * (lp - rp)).sum(-1).sum()
    return total / idx.numel()


class RFTTrainer(Trainer):
    beta_kl = BETA_KL

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs["labels"]
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                             shift_labels.reshape(-1), ignore_index=-100)
        loss = ce
        kl_val = 0.0
        if self.beta_kl > 0:
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logits = model(input_ids=inputs["input_ids"],
                                       attention_mask=inputs["attention_mask"]).logits
            ref_shift = ref_logits[:, :-1, :].contiguous()
            mask = (shift_labels != -100)
            kl = masked_kl(shift_logits, ref_shift, mask)
            loss = ce + self.beta_kl * kl
            kl_val = float(kl.detach())
        if self.state.global_step % 20 == 0:
            print(f"  [step {self.state.global_step}] ce={float(ce.detach()):.4f} kl={kl_val:.4f}", flush=True)
        return (loss, out) if return_outputs else loss


args = TrainingArguments(
    output_dir="/workspace/_rft_ckpt", num_train_epochs=EPOCHS,
    per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR, warmup_ratio=0.03, lr_scheduler_type="cosine",
    logging_steps=10, save_strategy="no", bf16=True, report_to="none",
    optim="paged_adamw_8bit", gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False, seed=SEED, dataloader_num_workers=2)

trainer = RFTTrainer(model=model, args=args, train_dataset=DS(ejemplos), data_collator=collate)

print(f"\n[RFT] ===== CONFIG ENTRENAMIENTO =====", flush=True)
print(f"[RFT] dataset dorado: {len(ejemplos)} ejemplos | epochs={EPOCHS} lr={LR} "
      f"beta_kl={BETA_KL} r={LORA_R} max_seq={MAX_SEQ} grad_accum={GRAD_ACCUM}", flush=True)
print(f"[RFT] base CONGELADA (4-bit) + LoRA | optim=paged_adamw_8bit bf16", flush=True)

stats = trainer.train()
loss = float(getattr(stats, "training_loss", 0.0))
print(f"[RFT] loss final = {loss:.4f}", flush=True)

os.makedirs(SALIDA, exist_ok=True)
model.save_pretrained(SALIDA)
tok.save_pretrained(SALIDA)
info = {"base": BASE_ID, "epochs": EPOCHS, "lr": LR, "beta_kl": BETA_KL, "r": LORA_R,
        "max_seq": MAX_SEQ, "grad_accum": GRAD_ACCUM, "n_ejemplos": len(ejemplos),
        "loss_final": loss, "hf_repo": HF_REPO_MODEL}
json.dump(info, open(f"{SALIDA}/info_rft.json", "w"), indent=2)
print(f"[RFT] adaptador guardado en {SALIDA}", flush=True)

try:
    create_repo(HF_REPO_MODEL, repo_type="model", private=True, exist_ok=True, token=TOKEN)
    api.upload_folder(folder_path=SALIDA, repo_id=HF_REPO_MODEL, repo_type="model",
                      commit_message=f"nova-rft-v1 loss={loss:.4f} n={len(ejemplos)}")
    print(f"[RFT] subido a HF: {HF_REPO_MODEL}", flush=True)
except Exception as e:
    print(f"[RFT] aviso HF: {repr(e)[:160]}", flush=True)
print("[RFT] PASO 3 COMPLETADO", flush=True)
