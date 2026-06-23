"""VAST.AI — Paso 4: genera N=64 soluciones para AIME 2023+2024+2025 con vLLM.

Descarga AIME de HF (públicos). Genera K=64 soluciones por problema.
Guarda /workspace/aime_gen.jsonl y lo sube a HF como backup.

Variables: HF_TOKEN
"""
import os, sys, json, re

HF_REPO_DATA = "Quantumadvancedai/nova-verif-data"
TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "Falta HF_TOKEN"
WORK = "/workspace"
SALIDA = os.path.join(WORK, "aime_gen.jsonl")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
K = 64
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 12288

from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=TOKEN)


# ---------------------------------------------------------------------------
# Extracción de respuesta numérica (AIME: entero 0-999)
# ---------------------------------------------------------------------------
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
        for tok in ["\\$", "\\%", "\\,", "\\!", "\\;", " ", "\\left", "\\right", "$", "%", ","]:
            s = s.replace(tok, "")
        s = s.replace("\\", "").strip()
        m = re.search(r"\d+", s)
        if m:
            return m.group(0)
    nums = re.findall(r"\d+", t[-2000:])
    return nums[-1] if nums else ""


def comparar(pred, gold):
    try:
        return int(pred) == int(gold)
    except (TypeError, ValueError):
        return str(pred).strip() == str(gold).strip()


# ---------------------------------------------------------------------------
# Cargar AIME 2023 + 2024 + 2025
# ---------------------------------------------------------------------------
FUENTES = [
    # (hf_dataset_id, split, anio)
    ("AI-MO/aimo-validation-aime", "train", "aime_2023"),
    ("Maxwell-Jia/AIME_2024_I", "train", "aime_2024_I"),
    ("Maxwell-Jia/AIME_2024_II", "train", "aime_2024_II"),
    ("Maxwell-Jia/AIME_2025_I", "train", "aime_2025_I"),
    ("Maxwell-Jia/AIME_2025_II", "train", "aime_2025_II"),
]

problemas = []


def _normalizar_gold(v):
    try:
        return str(int(float(str(v).strip().replace(",", ""))))
    except Exception:
        return str(v).strip()


from datasets import load_dataset

for hf_id, split, fuente in FUENTES:
    try:
        ds = load_dataset(hf_id, split=split, trust_remote_code=True)
        for row in ds:
            prob_txt = row.get("problem", row.get("Problem", row.get("Question", "")))
            gold_raw = row.get("answer", row.get("Answer", row.get("solution", "")))
            if not prob_txt:
                continue
            gold = _normalizar_gold(gold_raw)
            problemas.append({"problema": prob_txt.strip(), "gold": gold, "fuente": fuente})
        print(f"[GEN_AIME] {fuente}: {len([p for p in problemas if p['fuente']==fuente])} probs", flush=True)
    except Exception as e:
        print(f"[GEN_AIME] aviso {hf_id}: {repr(e)[:120]}", flush=True)

# Deduplicar por enunciado (a veces los datasets tienen overlap)
seen = set()
uniq = []
for p in problemas:
    k = p["problema"][:120]
    if k not in seen:
        seen.add(k)
        uniq.append(p)
problemas = uniq

# Asignar idx estable (orden de aparición)
for i, p in enumerate(problemas):
    p["idx"] = i

print(f"[GEN_AIME] {len(problemas)} problemas AIME total (deduplicados)", flush=True)
assert len(problemas) >= 30, f"Muy pocos problemas AIME: {len(problemas)}"

# Reanudar si ya existe output parcial
hechos = {}
if os.path.exists(SALIDA):
    for l in open(SALIDA, encoding="utf-8"):
        l = l.strip()
        if l:
            try:
                r = json.loads(l)
                hechos[r["idx"]] = hechos.get(r["idx"], [])
                hechos[r["idx"]].append(r)
            except Exception:
                pass
pend = [p for p in problemas if p["idx"] not in hechos]
print(f"[GEN_AIME] hechos={len(hechos)} pendientes={len(pend)}", flush=True)

if not pend:
    print("[GEN_AIME] ya completado — nada que generar", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Generación con vLLM
# ---------------------------------------------------------------------------
from vllm import LLM, SamplingParams

llm = LLM(model=MODEL_ID, dtype="float16", max_model_len=MAX_TOKENS + 2048,
          gpu_memory_utilization=0.90, trust_remote_code=True)
sp = SamplingParams(n=K, temperature=TEMPERATURE, top_p=TOP_P,
                    max_tokens=MAX_TOKENS, logprobs=1)


def construir_prompt(p):
    return p + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


CHUNK = 4  # problemas por lote (K=64 × 4 = 256 peticiones simultáneas)

with open(SALIDA, "a", encoding="utf-8") as f:
    for ci, s in enumerate(range(0, len(pend), CHUNK)):
        lote = pend[s:s + CHUNK]
        convs = [[{"role": "user", "content": construir_prompt(p["problema"])}] for p in lote]
        outs = llm.chat(convs, sp)
        for p, out in zip(lote, outs):
            for oi, o in enumerate(out.outputs):
                nt = len(o.token_ids)
                cert = (o.cumulative_logprob / nt) if (o.cumulative_logprob is not None and nt > 0) else 0.0
                pred = extraer_num(o.text)
                correcto = comparar(pred, p["gold"])
                f.write(json.dumps({
                    "idx": p["idx"],
                    "sol_idx": oi,
                    "problema": p["problema"],
                    "gold": p["gold"],
                    "fuente": p["fuente"],
                    "texto": o.text,
                    "pred": pred,
                    "certeza": cert,
                    "correcto": correcto,
                    "truncado": (o.finish_reason == "length"),
                }, ensure_ascii=False) + "\n")
            hechos[p["idx"]] = True
        f.flush()
        os.fsync(f.fileno())
        print(f"[GEN_AIME] lote {ci+1}: {len(hechos)}/{len(problemas)} probs", flush=True)

# subir backup a HF
try:
    api.upload_file(path_or_fileobj=SALIDA, path_in_repo="aime_gen.jsonl",
                    repo_id=HF_REPO_DATA, repo_type="dataset",
                    commit_message=f"aime_gen: {len(problemas)} probs × K={K}")
    print(f"[GEN_AIME] subido aime_gen.jsonl a HF", flush=True)
except Exception as e:
    print(f"[GEN_AIME] aviso subida HF: {repr(e)[:120]}", flush=True)

# Resumen rápido
filas = [json.loads(l) for l in open(SALIDA, encoding="utf-8") if l.strip()]
probs_u = {r["idx"] for r in filas}
corr_p = sum(1 for r in filas if r["correcto"])
print(f"[GEN_AIME] {len(filas)} soluciones | {len(probs_u)} problemas | {corr_p} correctas ({100*corr_p/max(1,len(filas)):.1f}%)", flush=True)
print("[GEN_AIME] PASO 4 COMPLETADO", flush=True)
