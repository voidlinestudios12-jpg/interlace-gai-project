"""KAGGLE — Generacion de datos del verificador (Fase 3) en GPU. ALTERNATIVA a Modal.

COMO USAR (en https://www.kaggle.com/code -> New Notebook):
  1. Settings -> Accelerator -> GPU T4 x1 (o P100).
  2. Settings -> Internet -> ON.
  3. Add-ons -> Secrets -> nuevo secreto  HF_TOKEN = tu token de Hugging Face
     (cuenta InterlaceAI, con permiso de ESCRITURA).
  4. Pega ESTE archivo entero en una celda y dale a "Run All".

QUE HACE:
  - Descarga del dataset privado HF `InterlaceAI/nova-verif-data` la lista de problemas
    (verif_problemas_numina.jsonl) y el dataset PARCIAL ya generado (verif_dataset_numina.jsonl).
  - Genera K=8 soluciones por problema PENDIENTE con Nova-v0 (DeepSeek-R1-Distill-Qwen-1.5B,
    vLLM, temp 0.6 / top_p 0.95) y las etiqueta correcto/incorrecto comparando con el gold.
  - REANUDABLE: salta los idx ya hechos. Sube el resultado a HF cada pocos lotes y al final
    (y queda en /kaggle/working como salida del notebook).

Al llegar a 600/600, continua en Modal/otro con el paso 'filtrar' de
nova/forge/preparar_datos_verificador.py para construir verif_dataset_v2.jsonl.
"""
import os
import sys
import subprocess
import json
import re

HF_REPO = "InterlaceAI/nova-verif-data"
PROB_FILE = "verif_problemas_numina.jsonl"
DS_FILE = "verif_dataset_numina.jsonl"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEMPERATURE, TOP_P, K, MAX_TOKENS = 0.6, 0.95, 8, 12288
CHUNK_P = 8        # problemas por lote
SUBIR_CADA = 5     # subir a HF cada N lotes (seguridad ante cortes de sesion)
WORK = "/kaggle/working"


def _sh(cmd):
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)


# 1) dependencias (vLLM + huggingface_hub)
try:
    import vllm  # noqa: F401
except Exception:
    _sh(f"{sys.executable} -m pip install -q -U vllm")
_sh(f"{sys.executable} -m pip install -q -U huggingface_hub")

from huggingface_hub import HfApi, hf_hub_download


def _hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


TOKEN = _hf_token()
assert TOKEN, "Falta el secreto HF_TOKEN (Add-ons -> Secrets)."
api = HfApi(token=TOKEN)


# ---- extractores IDENTICOS a nova/eval/run_benchmark.py (arnes validado) ----
def extraer_boxed(t):
    idx = t.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + 7
    depth = 1
    out = []
    while i < len(t) and depth > 0:
        c = t[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth > 0:
            out.append(c)
        i += 1
    return "".join(out)


def limpiar(s):
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mathrm\{[^}]*\}", "", s)
    for tok in ["\\$", "\\%", "\\,", "\\!", "\\;", "\\ ", "\\left", "\\right", "$", "%", ","]:
        s = s.replace(tok, "")
    s = s.replace("\\", "").strip()
    m = re.search(r"-?\d+\.?\d*", s)
    return m.group(0) if m else ""


def extraer_num(t):
    bx = extraer_boxed(t)
    if bx is not None:
        n = limpiar(bx)
        if n:
            return n
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "") if nums else ""


def comparar_num(pred, gold):
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False


def construir_prompt(p):
    return p + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


# 2) descargar problemas + dataset parcial de HF
def _baja(fn):
    try:
        return hf_hub_download(HF_REPO, fn, repo_type="dataset", token=TOKEN, local_dir=WORK)
    except Exception as e:
        print("aviso: no se pudo bajar", fn, repr(e)[:140], flush=True)
        return None


prob_path = _baja(PROB_FILE)
problemas = [json.loads(l) for l in open(prob_path, encoding="utf-8") if l.strip()]
ds_path = os.path.join(WORK, DS_FILE)
_baja(DS_FILE)  # dataset parcial (p.ej. 184) si ya existe
hechos = set()
if os.path.exists(ds_path):
    for l in open(ds_path, encoding="utf-8"):
        l = l.strip()
        if l:
            try:
                hechos.add(json.loads(l)["idx"])
            except Exception:
                pass
pend = [p for p in problemas if p["idx"] not in hechos]
print(f"problemas={len(problemas)} hechos={len(hechos)} pendientes={len(pend)}", flush=True)


def _subir():
    api.upload_file(path_or_fileobj=ds_path, path_in_repo=DS_FILE, repo_id=HF_REPO,
                    repo_type="dataset", commit_message=f"Kaggle: {len(hechos)} problemas hechos")
    print(f"  -> subido a HF: {len(hechos)} problemas", flush=True)


# 3) generar lo pendiente
if pend:
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, dtype="float16", max_model_len=MAX_TOKENS + 2048,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    sp = SamplingParams(n=K, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, logprobs=1)
    with open(ds_path, "a", encoding="utf-8") as f:
        for ci, s in enumerate(range(0, len(pend), CHUNK_P)):
            lote = pend[s:s + CHUNK_P]
            convs = [[{"role": "user", "content": construir_prompt(p["problema"])}] for p in lote]
            outs = llm.chat(convs, sp)
            for p, out in zip(lote, outs):
                for o in out.outputs:
                    nt = len(o.token_ids)
                    cert = (o.cumulative_logprob / nt) if (o.cumulative_logprob is not None and nt) else 0.0
                    pred = extraer_num(o.text)
                    et = 1 if comparar_num(pred, p["gold"]) else 0
                    f.write(json.dumps({
                        "idx": p["idx"], "problema": p["problema"], "gold": p["gold"],
                        "fuente": p.get("fuente"), "level": p.get("level"), "texto": o.text,
                        "pred": pred, "certeza": cert, "etiqueta": et,
                        "truncado": (o.finish_reason == "length"),
                    }, ensure_ascii=False) + "\n")
                hechos.add(p["idx"])
            f.flush()
            os.fsync(f.fileno())
            print(f"lote {ci + 1}: {len(hechos)}/{len(problemas)} problemas", flush=True)
            if (ci + 1) % SUBIR_CADA == 0:
                _subir()
    _subir()

# 4) balance final
filas = [json.loads(l) for l in open(ds_path, encoding="utf-8") if l.strip()]
pos = sum(1 for r in filas if r["etiqueta"] == 1)
nprob = len({r["idx"] for r in filas})
print(f"TOTAL: {len(filas)} soluciones | {pos} correctas / {len(filas) - pos} incorrectas | {nprob} problemas", flush=True)
print("Listo. Si nprob=600, sigue con el paso 'filtrar' para construir verif_dataset_v2.jsonl.", flush=True)
