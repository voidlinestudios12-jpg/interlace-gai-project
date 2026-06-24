"""VAST.AI — RFT PASO 2: construir el dataset DORADO (rejection sampling).

Idea (STaR / rejection-sampling fine-tuning): de las soluciones que el PROPIO
modelo ya generó sobre problemas de ENTRENAMIENTO, quedarse, por problema, con la
MEJOR solución CORRECTA — donde "mejor" la decide el verificador ORM. Eso da un
corpus (problema -> razonamiento de máxima calidad) con el que luego entrenamos el
modelo puro (N=1).

Fuentes (ya descontaminadas contra evaluación, y se re-descontaminan aquí contra
los 90 AIME de evaluación por seguridad):
  - verif_dataset_v1.jsonl     (MATH-train, 400 problemas x 8 soluciones)
  - verif_dataset_numina.jsonl (NuminaMath, 600 problemas x 8 soluciones)

Filtros de calidad de cada solución candidata:
  - etiqueta == 1 (correcta contra el gold)
  - contiene \\boxed{}  y  </think>  (traza bien formada)
  - no truncada
  - longitud dentro de límites (ni degenerada ni gigante)
  - sin bucles degenerados (n-gramas repetidos)

Salida: rft_dorado_v1.jsonl  (subido a HF). Variables: HF_TOKEN.
"""
import os, sys, json, re
from collections import defaultdict, Counter

HF_REPO_DATA = "Quantumadvancedai/nova-verif-data"
HF_REPO_MODEL = "Quantumadvancedai/nova-verificador-v1"
TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "Falta HF_TOKEN"
WORK = "/workspace"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
BASE_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Repo root (para leer nova/data/aime_eval_90.json)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Límites de calidad
MAX_LEN_CHARS = 24000   # ~7000 tokens; descarta trazas gigantes
MIN_LEN_CHARS = 200     # descarta soluciones triviales/vacías
DECON_UMBRAL = 0.6

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api = HfApi(token=TOKEN)


# --------------------------------------------------------------------------
# Descontaminación (idéntica a preparar_datos_verificador.py)
# --------------------------------------------------------------------------
def _norm_stmt(s):
    s = (s or "").lower()
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _shingle(norm, k=5):
    pal = norm.split()
    if len(pal) < k:
        return frozenset([" ".join(pal)]) if pal else frozenset()
    return frozenset(" ".join(pal[i:i + k]) for i in range(len(pal) - k + 1))


def _contaminado(norm_train, sh_train, eval_norms_set, eval_shingles, umbral=DECON_UMBRAL):
    if not norm_train:
        return False
    if norm_train in eval_norms_set:
        return True
    for ev in eval_norms_set:
        if len(norm_train) > 40 and len(ev) > 40 and (norm_train in ev or ev in norm_train):
            return True
    if not sh_train:
        return False
    for sh_ev in eval_shingles:
        if not sh_ev:
            continue
        inter = len(sh_train & sh_ev)
        if inter == 0:
            continue
        union = len(sh_train | sh_ev)
        if union and inter / union >= umbral:
            return True
    return False


def _degenerada(texto):
    """Detecta bucles degenerados: si algún 6-grama de palabras se repite mucho."""
    pal = texto.split()
    if len(pal) < 60:
        return False
    c = Counter(" ".join(pal[i:i + 6]) for i in range(len(pal) - 6 + 1))
    mas_comun = c.most_common(1)[0][1] if c else 0
    return mas_comun >= 12  # un mismo 6-grama 12+ veces = bucle


def _baja(fn):
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        return p
    return hf_hub_download(HF_REPO_DATA, fn, repo_type="dataset", token=TOKEN, local_dir=WORK)


# --------------------------------------------------------------------------
# 1) Cargar enunciados de evaluación (90 AIME) para descontaminar
# --------------------------------------------------------------------------
aime_path = os.path.join(REPO_ROOT, "nova", "data", "aime_eval_90.json")
eval_norms, eval_shingles = set(), []
if os.path.exists(aime_path):
    aime = json.load(open(aime_path, encoding="utf-8"))
    for p in aime:
        n = _norm_stmt(p["problema"])
        if n:
            eval_norms.add(n)
            eval_shingles.append(_shingle(n))
    print(f"[GOLD] descontaminación contra {len(eval_norms)} enunciados AIME eval", flush=True)
else:
    print(f"[GOLD] AVISO: no encuentro {aime_path}; sigo SIN descontaminar (peligroso)", flush=True)


# --------------------------------------------------------------------------
# 2) Cargar soluciones y filtrar candidatas correctas + bien formadas
# --------------------------------------------------------------------------
def cargar(fn):
    p = _baja(fn)
    filas = []
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if l:
            try:
                filas.append(json.loads(l))
            except Exception:
                pass
    return filas


todas = []
for fn in ["verif_dataset_v1.jsonl", "verif_dataset_numina.jsonl"]:
    f = cargar(fn)
    print(f"[GOLD] {fn}: {len(f)} soluciones", flush=True)
    todas += f

por_idx = defaultdict(list)
for r in todas:
    por_idx[r["idx"]].append(r)
print(f"[GOLD] {len(por_idx)} problemas, {len(todas)} soluciones totales", flush=True)


def candidata_valida(s):
    t = s.get("texto", "")
    if s.get("etiqueta", 0) != 1:
        return False
    if s.get("truncado", False):
        return False
    if "\\boxed{" not in t:
        return False
    if "</think>" not in t:
        return False
    if not (MIN_LEN_CHARS <= len(t) <= MAX_LEN_CHARS):
        return False
    if _degenerada(t):
        return False
    return True


# Filtrar problemas contaminados y construir lista de candidatas por problema
cand_por_idx = {}
n_contaminados = 0
for idx, sols in por_idx.items():
    problema = sols[0].get("problema", "")
    norm = _norm_stmt(problema)
    if _contaminado(norm, _shingle(norm), eval_norms, eval_shingles):
        n_contaminados += 1
        continue
    cands = [s for s in sols if candidata_valida(s)]
    if cands:
        cand_por_idx[idx] = cands

print(f"[GOLD] problemas contaminados descartados: {n_contaminados}", flush=True)
print(f"[GOLD] problemas con >=1 candidata válida: {len(cand_por_idx)}", flush=True)
total_cands = sum(len(v) for v in cand_por_idx.values())
print(f"[GOLD] candidatas correctas+bien formadas: {total_cands}", flush=True)

if not cand_por_idx:
    print("[GOLD] ERROR: 0 candidatas; abortando", flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------
# 3) Cargar verificador ORM y puntuar las candidatas
# --------------------------------------------------------------------------
print("[GOLD] cargando verificador ORM...", flush=True)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

verif_dir = snapshot_download(HF_REPO_MODEL, repo_type="model", token=TOKEN,
                              local_dir=os.path.join(WORK, "verificador_v1"))
MAX_LEN = 2048
vtok = AutoTokenizer.from_pretrained(verif_dir)
if vtok.pad_token is None:
    vtok.pad_token = vtok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
vmodel = AutoModelForSequenceClassification.from_pretrained(
    BASE_ID, num_labels=2, quantization_config=bnb, dtype=torch.bfloat16)
vmodel.config.pad_token_id = vtok.pad_token_id
vmodel = PeftModel.from_pretrained(vmodel, verif_dir)
vmodel.eval()
device = next(vmodel.parameters()).device
print("[GOLD] verificador cargado", flush=True)


def _tokenizar(tok, problema, solucion, max_len):
    """IDÉNTICA a la de entrenamiento del ORM (mismo template, crítico)."""
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


SCORE_BATCH = 12


@torch.no_grad()
def puntuar(rows):
    toks = [_tokenizar(vtok, r["problema"], r["texto"], MAX_LEN) for r in rows]
    max_l = max(len(t) for t in toks)
    pad = vtok.pad_token_id
    ids = torch.tensor([[pad] * (max_l - len(t)) + t for t in toks], dtype=torch.long).to(device)
    attn = (ids != pad).long()
    out = vmodel(input_ids=ids, attention_mask=attn)
    return torch.softmax(out.logits.float(), dim=-1)[:, 1].cpu().tolist()


# Aplanar para puntuar en lotes
flat = [(idx, j) for idx, cands in cand_por_idx.items() for j in range(len(cands))]
print(f"[GOLD] puntuando {len(flat)} candidatas...", flush=True)
for b in range(0, len(flat), SCORE_BATCH):
    lote = flat[b:b + SCORE_BATCH]
    rows = [cand_por_idx[idx][j] for idx, j in lote]
    scores = puntuar(rows)
    for (idx, j), sc in zip(lote, scores):
        cand_por_idx[idx][j]["prm_score"] = sc
    if (b // SCORE_BATCH + 1) % 50 == 0:
        print(f"[GOLD] {b + len(lote)}/{len(flat)} puntuadas", flush=True)


# --------------------------------------------------------------------------
# 4) Elegir la mejor candidata por problema y escribir el dataset dorado
# --------------------------------------------------------------------------
SUFIJO = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
dorado = []
for idx, cands in cand_por_idx.items():
    mejor = max(cands, key=lambda s: s.get("prm_score", 0.0))
    dorado.append({
        "idx": idx,
        "problema": mejor["problema"],
        "gold": mejor.get("gold"),
        "fuente": mejor.get("fuente"),
        "solucion": mejor["texto"],
        "prm_score": mejor.get("prm_score", 0.0),
        "n_candidatas": len(cands),
    })

dorado.sort(key=lambda r: r["idx"])
out_path = os.path.join(WORK, "rft_dorado_v1.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for r in dorado:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Estadísticas
scores = [r["prm_score"] for r in dorado]
longs = [len(r["solucion"]) for r in dorado]
por_fuente = Counter(r.get("fuente") for r in dorado)
print(f"\n[GOLD] === DATASET DORADO ===", flush=True)
print(f"[GOLD] problemas: {len(dorado)}", flush=True)
print(f"[GOLD] prm_score medio: {sum(scores)/len(scores):.3f} (min {min(scores):.3f} max {max(scores):.3f})", flush=True)
print(f"[GOLD] longitud media: {sum(longs)//len(longs)} chars (min {min(longs)} max {max(longs)})", flush=True)
print(f"[GOLD] por fuente: {dict(por_fuente)}", flush=True)

# Subir a HF
try:
    api.upload_file(path_or_fileobj=out_path, path_in_repo="rft_dorado_v1.jsonl",
                    repo_id=HF_REPO_DATA, repo_type="dataset",
                    commit_message=f"rft dorado v1: {len(dorado)} problemas")
    print(f"[GOLD] subido rft_dorado_v1.jsonl a HF", flush=True)
except Exception as e:
    print(f"[GOLD] aviso subida HF: {repr(e)[:160]}", flush=True)

# Meta
meta = {"problemas": len(dorado), "candidatas_totales": total_cands,
        "contaminados_descartados": n_contaminados,
        "prm_score_medio": sum(scores) / len(scores),
        "long_media_chars": sum(longs) // len(longs), "por_fuente": dict(por_fuente)}
json.dump(meta, open(os.path.join(WORK, "rft_dorado_v1_meta.json"), "w"), indent=2)
print("[GOLD] PASO 2 COMPLETADO", flush=True)
