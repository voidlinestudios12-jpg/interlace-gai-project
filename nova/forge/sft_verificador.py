"""FASE 3 — PASO 2: entrenar el VERIFICADOR (ORM = Outcome Reward Model).

Un clasificador que, dado (problema, solucion completa), predice P(correcta). Se construye
como CABEZA DE CLASIFICACION sobre la BASE CONGELADA (DeepSeek-R1-Distill-Qwen-1.5B) en
4-bit (QLoRA) + LoRA en el tronco. La base NO se toca: solo se entrenan los adaptadores LoRA
y la cabeza de puntuacion ('score').

Datos: /data/verif_dataset_v1.jsonl (PASO 1) — soluciones de Nova-v0 sobre MATH train
descontaminado, etiquetadas correcto/incorrecto.

Validacion HONESTA: el split train/val se hace POR PROBLEMA (no por solucion), de modo que
ninguna solucion de un problema de validacion aparece en entrenamiento (evita fuga).

Guardado: adaptador + cabeza + tokenizer en /data/verificador_v1 (volumen). Se intenta subir
al Hub privado como nova-verificador-v1 si hay token de escritura (HF_TOKEN); si no, queda en
el volumen (y se descarga en local) y se documenta el comando para subirlo despues.

Uso:  modal run nova/forge/sft_verificador.py --epochs 2
Descargar: modal volume get nova-data verificador_v1 ./resultados/verificador_v1
"""
import modal

app = modal.App("nova-verif-train")
vol = modal.Volume.from_name("nova-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==4.55.0", "peft", "bitsandbytes",
                 "accelerate", "datasets", "scikit-learn")
    .add_local_file("shared/modelo_base.py", "/root/modelo_base.py")
)

DATASET_FILE = "/data/verif_dataset_v1.jsonl"
SALIDA = "/data/verificador_v1"
HUB_REPO = "nova-verificador-v1"


def _tokenizar(tok, problema, solucion, max_len):
    """Construye los ids: prefijo del problema + COLA de la solucion (donde esta la respuesta
    final y la conclusion) + pregunta de cierre. Truncar por la cola conserva el \\boxed final,
    que es lo mas informativo para un verificador de RESULTADO."""
    pref = tok(f"Problema:\n{problema}\n\nSolucion propuesta:\n", add_special_tokens=False).input_ids
    cierre = tok("\n\n¿La respuesta final es correcta?", add_special_tokens=False).input_ids
    sol = tok(solucion, add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    presup = max_len - len(bos) - len(pref) - len(cierre)
    if presup < 0:  # problema larguisimo: recorta el prefijo por seguridad
        pref = pref[:max(0, len(pref) + presup)]
        presup = 0
    sol = sol[-presup:] if presup > 0 else []
    return bos + pref + sol + cierre


@app.function(gpu="A10G", image=image, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 5)
def entrenar(epochs, lr, r, batch, grad_accum, max_len, val_frac, seed):
    import json
    import os
    import random
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache"
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorWithPadding,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from sklearn.metrics import roc_auc_score
    import modelo_base as mb

    if not os.path.exists(DATASET_FILE):
        raise RuntimeError(f"No existe {DATASET_FILE}; ejecuta primero el PASO 1 (generar)")
    filas = [json.loads(l) for l in open(DATASET_FILE, encoding="utf-8") if l.strip()]
    print(f"[TRAIN] {len(filas)} soluciones cargadas", flush=True)

    # --- split POR PROBLEMA (sin fuga) ---
    idxs = sorted({r["idx"] for r in filas})
    random.Random(seed).shuffle(idxs)
    n_val = max(1, int(len(idxs) * val_frac))
    val_ids = set(idxs[:n_val])
    tr = [r for r in filas if r["idx"] not in val_ids]
    va = [r for r in filas if r["idx"] in val_ids]
    pos_tr = sum(r["etiqueta"] for r in tr)
    print(f"[TRAIN] problemas: {len(idxs)} (val {len(val_ids)}) | train {len(tr)} sol ({pos_tr} pos), val {len(va)} sol", flush=True)

    tok = AutoTokenizer.from_pretrained(mb.MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            ids = _tokenizar(tok, r["problema"], r["texto"], max_len)
            return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": int(r["etiqueta"])}

    ds_tr, ds_va = DS(tr), DS(va)

    # --- modelo: base 4-bit + cabeza de clasificacion + LoRA ---
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        mb.MODEL_ID, num_labels=2, quantization_config=bnb, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # --- pesos de clase (desbalanceo) ---
    n0 = sum(1 for r in tr if r["etiqueta"] == 0)
    n1 = len(tr) - n0
    w = torch.tensor([len(tr) / (2 * max(1, n0)), len(tr) / (2 * max(1, n1))], dtype=torch.float)
    print(f"[TRAIN] pesos de clase: neg={w[0]:.3f} pos={w[1]:.3f}", flush=True)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=w.to(out.logits.device))(out.logits, labels)
            return (loss, out) if return_outputs else loss

    def metrics(ep):
        logits, labels = ep
        prob = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        pred = prob >= 0.5
        acc = float((pred == labels).mean())
        try:
            auc = float(roc_auc_score(labels, prob)) if len(set(labels.tolist())) > 1 else 0.0
        except Exception:
            auc = 0.0
        return {"accuracy": acc, "auc": auc}

    args = TrainingArguments(
        output_dir="/data/_verif_ckpt", num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum, learning_rate=lr, warmup_ratio=0.03,
        lr_scheduler_type="cosine", logging_steps=10, eval_strategy="epoch",
        save_strategy="no", bf16=True, report_to="none", optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, seed=seed)
    trainer = WTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                       data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    ev0 = trainer.evaluate()
    print(f"[TRAIN] val inicial (cabeza aleatoria): {ev0}", flush=True)
    trainer.train()
    ev = trainer.evaluate()
    print(f"[TRAIN] val final: {ev}", flush=True)

    # --- guardar adaptador + cabeza + tokenizer en el volumen ---
    os.makedirs(SALIDA, exist_ok=True)
    model.save_pretrained(SALIDA)
    tok.save_pretrained(SALIDA)
    info = {"base": mb.MODEL_ID, "epochs": epochs, "lr": lr, "r": r, "max_len": max_len,
            "val_problemas": len(val_ids), "val_metrics": ev,
            "val_inicial": ev0, "train_sol": len(tr), "val_sol": len(va),
            "pos_train": n1, "neg_train": n0}
    json.dump(info, open(f"{SALIDA}/info_entrenamiento.json", "w"), indent=2)
    vol.commit()
    print(f"[TRAIN] guardado en {SALIDA}", flush=True)

    # --- intentar subir al Hub privado (si hay token de escritura) ---
    subido = False
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import HfApi, create_repo
            who = HfApi(token=token).whoami()
            repo_id = f"{who['name']}/{HUB_REPO}"
            create_repo(repo_id, private=True, exist_ok=True, token=token)
            model.push_to_hub(repo_id, private=True, token=token)
            tok.push_to_hub(repo_id, private=True, token=token)
            subido = repo_id
            print(f"[TRAIN] subido a HF Hub privado: {repo_id}", flush=True)
        except Exception as e:
            print(f"[TRAIN] aviso: no se pudo subir a HF Hub ({repr(e)[:160]}); queda en el volumen.", flush=True)
    else:
        print("[TRAIN] sin HF_TOKEN: el verificador queda en el volumen (y se descargara en local).", flush=True)

    return {"val": ev, "val_inicial": ev0, "subido_hub": subido, "salida": SALIDA}


@app.local_entrypoint()
def main(epochs: int = 2, lr: float = 1e-4, r: int = 16, batch: int = 4,
         grad_accum: int = 4, max_len: int = 2048, val_frac: float = 0.1, seed: int = 1234):
    print(entrenar.remote(epochs, lr, r, batch, grad_accum, max_len, val_frac, seed))
