"""VAST.AI — Paso 3: entrena el ORM (verificador) sobre verif_dataset_v2.jsonl.

Clasificador QLoRA 4-bit sobre la base congelada (DeepSeek-R1-Distill-Qwen-1.5B):
dada (problema, solucion), predice P(correcta). Split por PROBLEMA (sin fuga).
Guarda adaptador + cabeza en /workspace/verificador_v1 y sube a HF.

Variables: HF_TOKEN, HF_HOME (opcional)
"""
import os, sys, json, random

HF_REPO_DATA = "Quantumadvancedai/nova-verif-data"
HF_REPO_MODEL = "Quantumadvancedai/nova-verificador-v1"
TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "Falta HF_TOKEN"
WORK = "/workspace"
SALIDA = "/workspace/verificador_v1"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

from huggingface_hub import HfApi, hf_hub_download, create_repo
api = HfApi(token=TOKEN)


def _baja(fn):
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        return p
    return hf_hub_download(HF_REPO_DATA, fn, repo_type="dataset", token=TOKEN, local_dir=WORK)


v2_path = _baja("verif_dataset_v2.jsonl")
filas = [json.loads(l) for l in open(v2_path, encoding="utf-8") if l.strip()]
print(f"[TRAIN] {len(filas)} soluciones cargadas de v2", flush=True)

# Hiper-parámetros
EPOCHS = 2
LR = 1e-4
LORA_R = 16
BATCH = 4
GRAD_ACCUM = 4
MAX_LEN = 2048
VAL_FRAC = 0.1
SEED = 1234

# Split POR PROBLEMA (no por solución — evita fuga)
idxs = sorted({r["idx"] for r in filas})
random.Random(SEED).shuffle(idxs)
n_val = max(1, int(len(idxs) * VAL_FRAC))
val_ids = set(idxs[:n_val])
tr = [r for r in filas if r["idx"] not in val_ids]
va = [r for r in filas if r["idx"] in val_ids]
pos_tr = sum(r["etiqueta"] for r in tr)
print(f"[TRAIN] problemas: {len(idxs)} | train {len(tr)} sol ({pos_tr} pos), val {len(va)} sol", flush=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorWithPadding,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import roc_auc_score


def _tokenizar(tok, problema, solucion, max_len):
    """Prefijo del problema + COLA de la solucion (donde esta \\boxed{}) + cierre."""
    pref = tok(f"Problema:\n{problema}\n\nSolucion propuesta:\n", add_special_tokens=False).input_ids
    cierre = tok("\n\n¿La respuesta final es correcta?", add_special_tokens=False).input_ids
    sol = tok(solucion, add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    presup = max_len - len(bos) - len(pref) - len(cierre)
    if presup < 0:
        pref = pref[:max(0, len(pref) + presup)]
        presup = 0
    sol = sol[-presup:] if presup > 0 else []
    return bos + pref + sol + cierre


tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


class DS(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        ids = _tokenizar(tok, r["problema"], r["texto"], MAX_LEN)
        return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": int(r["etiqueta"])}


ds_tr, ds_va = DS(tr), DS(va)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2, quantization_config=bnb, torch_dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora = LoraConfig(
    r=LORA_R, lora_alpha=2*LORA_R, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    modules_to_save=["score"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
model.config.use_cache = False

n0 = sum(1 for r in tr if r["etiqueta"] == 0)
n1 = len(tr) - n0
w = torch.tensor([len(tr)/(2*max(1,n0)), len(tr)/(2*max(1,n1))], dtype=torch.float)
print(f"[TRAIN] pesos clase: neg={w[0]:.3f} pos={w[1]:.3f}", flush=True)


class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=w.to(out.logits.device))(out.logits, labels)
        return (loss, out) if return_outputs else loss


def metrics(ep):
    logits, labels = ep
    prob = torch.softmax(torch.tensor(logits), dim=-1)[:,1].numpy()
    pred = prob >= 0.5
    acc = float((pred == labels).mean())
    try: auc = float(roc_auc_score(labels, prob)) if len(set(labels.tolist()))>1 else 0.0
    except Exception: auc = 0.0
    return {"accuracy": acc, "auc": auc}


args = TrainingArguments(
    output_dir="/workspace/_verif_ckpt", num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, warmup_ratio=0.03,
    lr_scheduler_type="cosine", logging_steps=20, eval_strategy="epoch",
    save_strategy="no", bf16=True, report_to="none", optim="paged_adamw_8bit",
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False, seed=SEED)
trainer = WTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                   data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)

ev0 = trainer.evaluate()
print(f"[TRAIN] val inicial: acc={ev0.get('eval_accuracy',0):.3f} auc={ev0.get('eval_auc',0):.3f}", flush=True)
trainer.train()
ev = trainer.evaluate()
print(f"[TRAIN] val final: acc={ev.get('eval_accuracy',0):.3f} auc={ev.get('eval_auc',0):.3f}", flush=True)

os.makedirs(SALIDA, exist_ok=True)
model.save_pretrained(SALIDA)
tok.save_pretrained(SALIDA)
info = {"base": MODEL_ID, "epochs": EPOCHS, "lr": LR, "r": LORA_R, "max_len": MAX_LEN,
        "val_metrics": ev, "val_inicial": ev0, "train_sol": len(tr), "val_sol": len(va),
        "hf_model_repo": HF_REPO_MODEL}
json.dump(info, open(f"{SALIDA}/info_entrenamiento.json", "w"), indent=2)
print(f"[TRAIN] guardado en {SALIDA}", flush=True)

# subir a HF
try:
    create_repo(HF_REPO_MODEL, repo_type="model", private=True, exist_ok=True, token=TOKEN)
    api.upload_folder(folder_path=SALIDA, repo_id=HF_REPO_MODEL, repo_type="model",
                      commit_message=f"nova-verificador-v1 acc={ev.get('eval_accuracy',0):.3f} auc={ev.get('eval_auc',0):.3f}")
    print(f"[TRAIN] subido a HF: {HF_REPO_MODEL}", flush=True)
except Exception as e:
    print(f"[TRAIN] aviso HF: {repr(e)[:160]}", flush=True)
print("[TRAIN] PASO 3 COMPLETADO", flush=True)
