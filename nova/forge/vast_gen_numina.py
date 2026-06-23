"""VAST.AI — Paso 1: genera los 416 problemas NuminaMath restantes.

Descarga problemas + dataset parcial de HF InterlaceAI/nova-verif-data,
genera K=8 soluciones con Nova-v0 (vLLM), sube a HF cada N lotes.
REANUDABLE: salta los idx ya hechos.

Ejecutar en la GPU: python nova/forge/vast_gen_numina.py
Variables requeridas: HF_TOKEN, HF_HOME (opcional, default /workspace/hf_cache)
"""
import os, sys, json, re

HF_REPO = "Quantumadvancedai/nova-verif-data"
PROB_FILE = "verif_problemas_numina.jsonl"
DS_FILE = "verif_dataset_numina.jsonl"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEMPERATURE, TOP_P, K, MAX_TOKENS = 0.6, 0.95, 8, 12288
CHUNK_P = 8
SUBIR_CADA = 5  # subir a HF cada N lotes (seguridad ante cortes)
WORK = "/workspace"

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "Falta HF_TOKEN en el entorno"

from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=TOKEN)


def _baja(fn):
    try:
        p = hf_hub_download(HF_REPO, fn, repo_type="dataset", token=TOKEN, local_dir=WORK)
        print(f"  descargado {fn} -> {p}", flush=True)
        return p
    except Exception as e:
        print(f"  aviso: no se pudo bajar {fn}: {repr(e)[:120]}", flush=True)
        return None


def extraer_boxed(t):
    idx = t.rfind("\\boxed{")
    if idx == -1:
        return None
    i, depth, out = idx + 7, 1, []
    while i < len(t) and depth > 0:
        c = t[i]
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        if depth > 0: out.append(c)
        i += 1
    return "".join(out)


def extraer_num(t):
    bx = extraer_boxed(t)
    if bx is not None:
        s = re.sub(r"\\text\{[^}]*\}", "", bx)
        s = re.sub(r"\\mathrm\{[^}]*\}", "", s)
        for tok in ["\\$","\\%","\\,","\\!","\\;"," ","\\left","\\right","$","%",","]:
            s = s.replace(tok, "")
        s = s.replace("\\", "").strip()
        m = re.search(r"-?\d+\.?\d*", s)
        if m: return m.group(0)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "") if nums else ""


def comparar_num(pred, gold):
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False


def construir_prompt(p):
    return p + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


prob_path = _baja(PROB_FILE)
assert prob_path, f"No se pudo bajar {PROB_FILE}"
problemas = [json.loads(l) for l in open(prob_path, encoding="utf-8") if l.strip()]
ds_path = os.path.join(WORK, DS_FILE)
_baja(DS_FILE)

hechos = set()
if os.path.exists(ds_path):
    for l in open(ds_path, encoding="utf-8"):
        l = l.strip()
        if l:
            try: hechos.add(json.loads(l)["idx"])
            except Exception: pass
pend = [p for p in problemas if p["idx"] not in hechos]
print(f"[GEN] problemas={len(problemas)} hechos={len(hechos)} pendientes={len(pend)}", flush=True)

if not pend:
    print("[GEN] nada pendiente — ya completo", flush=True)
    sys.exit(0)

from vllm import LLM, SamplingParams
llm = LLM(model=MODEL_ID, dtype="float16", max_model_len=MAX_TOKENS + 2048,
          gpu_memory_utilization=0.90, trust_remote_code=True)
sp = SamplingParams(n=K, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, logprobs=1)


def _subir():
    try:
        api.upload_file(path_or_fileobj=ds_path, path_in_repo=DS_FILE, repo_id=HF_REPO,
                        repo_type="dataset", commit_message=f"vast: {len(hechos)}/{len(problemas)} problemas")
        print(f"  -> subido a HF: {len(hechos)}/{len(problemas)}", flush=True)
    except Exception as e:
        print(f"  aviso subida HF: {repr(e)[:120]}", flush=True)


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
        f.flush(); os.fsync(f.fileno())
        print(f"[GEN] lote {ci + 1}: {len(hechos)}/{len(problemas)} problemas", flush=True)
        if (ci + 1) % SUBIR_CADA == 0:
            _subir()

_subir()
filas = [json.loads(l) for l in open(ds_path, encoding="utf-8") if l.strip()]
pos = sum(1 for r in filas if r["etiqueta"] == 1)
nprob = len({r["idx"] for r in filas})
print(f"[GEN] TOTAL: {len(filas)} sol | {pos} correctas / {len(filas)-pos} incorrectas | {nprob} problemas", flush=True)
print("[GEN] PASO 1 COMPLETADO", flush=True)
