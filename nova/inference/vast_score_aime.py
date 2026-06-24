"""VAST.AI — Paso 5: puntúa las soluciones AIME con el ORM y genera la tabla final.

Lee /workspace/aime_gen.jsonl (K=64 soluciones por problema).
Para cada N en SWEEP y cada selector, elige la respuesta y evalúa.
Escribe /workspace/report_verif_eval.md y lo sube a HF + GitHub (via git).

Variables: HF_TOKEN, GITHUB_TOKEN (para push a repo)
"""
import os, sys, json, re, math
from collections import defaultdict

HF_REPO_DATA = "Quantumadvancedai/nova-verif-data"
HF_REPO_MODEL = "Quantumadvancedai/nova-verificador-v1"
TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "Falta HF_TOKEN"
WORK = "/workspace"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

SWEEP = [8, 16, 32, 64]
SELECTORES = ["mayoria", "autocerteza", "verificador_prm", "verificador_prm_pesado"]

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api = HfApi(token=TOKEN)

# ---------------------------------------------------------------------------
# Cargar soluciones generadas
# ---------------------------------------------------------------------------
gen_path = os.path.join(WORK, "aime_gen.jsonl")
if not os.path.exists(gen_path):
    try:
        gen_path = hf_hub_download(HF_REPO_DATA, "aime_gen.jsonl", repo_type="dataset",
                                   token=TOKEN, local_dir=WORK)
    except Exception as e:
        print(f"[SCORE] no se pudo bajar aime_gen.jsonl: {repr(e)}", flush=True)
        sys.exit(1)

filas = [json.loads(l) for l in open(gen_path, encoding="utf-8") if l.strip()]
print(f"[SCORE] {len(filas)} soluciones cargadas", flush=True)

# Agrupar por problema
por_idx = defaultdict(list)
for r in filas:
    por_idx[r["idx"]].append(r)
probs = sorted(por_idx.keys())
print(f"[SCORE] {len(probs)} problemas únicos", flush=True)

# Verificar K disponible
k_min = min(len(v) for v in por_idx.values())
k_max = max(len(v) for v in por_idx.values())
print(f"[SCORE] soluciones por problema: min={k_min} max={k_max}", flush=True)
if k_min < max(SWEEP):
    print(f"[SCORE] aviso: K disponible ({k_min}) < max sweep ({max(SWEEP)}), ajustando sweep", flush=True)
    SWEEP = [n for n in SWEEP if n <= k_min]

# ---------------------------------------------------------------------------
# Cargar verificador ORM
# ---------------------------------------------------------------------------
print("[SCORE] cargando verificador ORM...", flush=True)
verif_dir = os.path.join(WORK, "verificador_v1")
if not os.path.exists(verif_dir):
    try:
        verif_dir = snapshot_download(HF_REPO_MODEL, repo_type="model", token=TOKEN,
                                      local_dir=os.path.join(WORK, "verificador_v1"))
    except Exception as e:
        print(f"[SCORE] no se pudo bajar verificador: {repr(e)}", flush=True)
        sys.exit(1)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MAX_LEN = 2048

base_tok = AutoTokenizer.from_pretrained(verif_dir)
if base_tok.pad_token is None:
    base_tok.pad_token = base_tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

# Intentar cargar como PEFT sobre base, o directamente si ya fue merged
try:
    base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    verif_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, num_labels=2, quantization_config=bnb, torch_dtype=torch.bfloat16)
    verif_model.config.pad_token_id = base_tok.pad_token_id
    verif_model = PeftModel.from_pretrained(verif_model, verif_dir)
    print("[SCORE] verificador cargado (PEFT sobre base)", flush=True)
except Exception as e:
    print(f"[SCORE] fallo PEFT load, intento directo: {repr(e)[:120]}", flush=True)
    verif_model = AutoModelForSequenceClassification.from_pretrained(
        verif_dir, num_labels=2, quantization_config=bnb, torch_dtype=torch.bfloat16)
    verif_model.config.pad_token_id = base_tok.pad_token_id
    print("[SCORE] verificador cargado (directo)", flush=True)

verif_model.eval()
device = next(verif_model.parameters()).device


def _tokenizar(tok, problema, solucion, max_len):
    """Idéntica a la de entrenamiento (crítico: mismo template)."""
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


SCORE_BATCH = 8


@torch.no_grad()
def puntuar_lote(rows):
    """Devuelve lista de p(correcto) ∈ [0,1] para cada fila."""
    tokenizados = [_tokenizar(base_tok, r["problema"], r["texto"], MAX_LEN) for r in rows]
    max_l = max(len(t) for t in tokenizados)
    pad_id = base_tok.pad_token_id
    input_ids = torch.tensor([[pad_id] * (max_l - len(t)) + t for t in tokenizados], dtype=torch.long).to(device)
    attn = (input_ids != pad_id).long()
    out = verif_model(input_ids=input_ids, attention_mask=attn)
    probs = torch.softmax(out.logits.float(), dim=-1)[:, 1].cpu().tolist()
    return probs


print("[SCORE] puntuando soluciones con ORM...", flush=True)
# Puntuar todas las soluciones de una vez en lotes
prm_scores = {}  # (idx, sol_idx) -> prob
all_rows = [(r["idx"], r.get("sol_idx", idx), r) for idx, r in enumerate(filas)]
# Recalcular sol_idx si no está
sol_idx_counters = defaultdict(int)
indexed_rows = []
for r in filas:
    si = r.get("sol_idx")
    if si is None:
        si = sol_idx_counters[r["idx"]]
        sol_idx_counters[r["idx"]] += 1
    indexed_rows.append((r["idx"], si, r))

for b in range(0, len(indexed_rows), SCORE_BATCH):
    lote_info = indexed_rows[b:b + SCORE_BATCH]
    lote_rows = [x[2] for x in lote_info]
    scores = puntuar_lote(lote_rows)
    for (idx, si, _), sc in zip(lote_info, scores):
        prm_scores[(idx, si)] = sc
    if (b // SCORE_BATCH + 1) % 50 == 0:
        print(f"[SCORE] {b + SCORE_BATCH}/{len(indexed_rows)} puntuadas", flush=True)

print(f"[SCORE] {len(prm_scores)} soluciones puntuadas", flush=True)


# ---------------------------------------------------------------------------
# Selectores
# ---------------------------------------------------------------------------
def _normalizar(s):
    s = str(s).strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s


def mayoria(sols):
    from collections import Counter
    c = Counter(_normalizar(s.get("pred", "")) for s in sols)
    return c.most_common(1)[0][0]


def autocerteza(sols):
    best = max(sols, key=lambda s: s.get("certeza", float("-inf")))
    return _normalizar(best.get("pred", ""))


def verificador_prm(sols):
    best = max(sols, key=lambda s: s.get("prm_score", float("-inf")))
    return _normalizar(best.get("pred", ""))


def verificador_prm_pesado(sols):
    from collections import defaultdict as dd
    votos = dd(float)
    for s in sols:
        votos[_normalizar(s.get("pred", ""))] += s.get("prm_score", 0.0)
    return max(votos, key=votos.__getitem__)


def oracle(sols, gold):
    """Límite superior: acierta si alguna solución es correcta."""
    norm_gold = _normalizar(gold)
    for s in sols:
        if _normalizar(s.get("pred", "")) == norm_gold:
            return norm_gold
    return ""


# Añadir prm_scores a las filas
si_counters = defaultdict(int)
for r in filas:
    si = r.get("sol_idx")
    if si is None:
        si = si_counters[r["idx"]]
        si_counters[r["idx"]] += 1
    r["prm_score"] = prm_scores.get((r["idx"], si), 0.0)
    r["sol_idx_computed"] = si

# ---------------------------------------------------------------------------
# Evaluar
# ---------------------------------------------------------------------------
print("[SCORE] evaluando selectores...", flush=True)
resultados = {}  # (selector, N) -> {"correct": int, "total": int}

for N in SWEEP:
    for idx in probs:
        # Tomar las primeras N soluciones (orden de generación)
        sols_n = por_idx[idx][:N]
        gold = sols_n[0]["gold"]
        norm_gold = _normalizar(gold)

        for sel in SELECTORES:
            if sel == "mayoria":
                pred = mayoria(sols_n)
            elif sel == "autocerteza":
                pred = autocerteza(sols_n)
            elif sel == "verificador_prm":
                pred = verificador_prm(sols_n)
            elif sel == "verificador_prm_pesado":
                pred = verificador_prm_pesado(sols_n)
            else:
                continue
            key = (sel, N)
            resultados.setdefault(key, {"correct": 0, "total": 0})
            resultados[key]["total"] += 1
            if pred == norm_gold:
                resultados[key]["correct"] += 1

        # oracle
        key_or = ("oracle", N)
        resultados.setdefault(key_or, {"correct": 0, "total": 0})
        resultados[key_or]["total"] += 1
        if oracle(sols_n, gold) == norm_gold:
            resultados[key_or]["correct"] += 1


def pct(v):
    return f"{100*v['correct']/max(1,v['total']):.1f}%"


def ci95(v):
    p = v["correct"] / max(1, v["total"])
    n = v["total"]
    err = 1.96 * math.sqrt(p * (1 - p) / max(1, n))
    return f"±{100*err:.1f}pp"


# ---------------------------------------------------------------------------
# Reporte Markdown
# ---------------------------------------------------------------------------
lines = ["# Nova Fase 3 — Evaluación Verificador ORM\n",
         f"**Modelo base:** DeepSeek-R1-Distill-Qwen-1.5B  ",
         f"**Eval set:** AIME 2023+2024+2025 ({len(probs)} problemas)  ",
         f"**Sweep N:** {SWEEP}\n",
         "## Resultados por selector\n",
         "| Selector | N=8 | N=16 | N=32 | N=64 |",
         "|---|---|---|---|---|"]

all_sels = SELECTORES + ["oracle"]
for sel in all_sels:
    row = f"| {sel} |"
    for N in SWEEP:
        if (sel, N) in resultados:
            v = resultados[(sel, N)]
            row += f" {pct(v)} ({v['correct']}/{v['total']}) {ci95(v)} |"
        else:
            row += " — |"
    lines.append(row)

lines += ["",
          "## Interpretación",
          "- **oracle**: límite superior teórico (si alguna solución entre N es correcta).",
          "- **verificador_prm**: selecciona la solución con mayor P(correcta) según ORM.",
          "- **verificador_prm_pesado**: voto ponderado por P(correcta) del ORM.",
          "- **mayoria**: voto mayoritario (baseline fuerte).",
          "- **autocerteza**: solución con mayor log-probabilidad media (baseline ligero).",
          ""]

# Determinar ganador
print("\n[SCORE] === TABLA FINAL ===", flush=True)
for line in lines:
    print(line, flush=True)

report_path = os.path.join(WORK, "report_verif_eval.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[SCORE] reporte guardado en {report_path}", flush=True)

# Guardar también JSON con los números crudos
json_path = os.path.join(WORK, "report_verif_eval.json")
json_data = {f"{sel}_N{N}": resultados.get((sel, N), {}) for sel in all_sels for N in SWEEP}
json.dump(json_data, open(json_path, "w"), indent=2)

# ---------------------------------------------------------------------------
# Subir a HF
# ---------------------------------------------------------------------------
try:
    api.upload_file(path_or_fileobj=report_path, path_in_repo="report_verif_eval.md",
                    repo_id=HF_REPO_DATA, repo_type="dataset",
                    commit_message="report: fase3 ORM evaluation AIME")
    api.upload_file(path_or_fileobj=json_path, path_in_repo="report_verif_eval.json",
                    repo_id=HF_REPO_DATA, repo_type="dataset",
                    commit_message="report json: fase3 ORM evaluation AIME")
    print(f"[SCORE] reporte subido a HF", flush=True)
except Exception as e:
    print(f"[SCORE] aviso subida HF: {repr(e)[:120]}", flush=True)

print("[SCORE] PASO 5 COMPLETADO", flush=True)
