"""VAST.AI — Paso 2: construye verif_dataset_v2.jsonl (CPU, sin GPU).

v2 = v1 (MATH-train, 400 problemas) + problemas MIXTOS de NuminaMath
(aquellos donde el modelo a veces acierta y a veces falla: 1-7/8 correctas).
Los triviales (8/8) y los imposibles (0/8) se descartan — no aportan al verificador.

Descarga v1 + numina de HF si no existen localmente. Sube v2 a HF al terminar.

Variables: HF_TOKEN, HF_HOME (opcional)
"""
import os, sys, json
from collections import defaultdict

HF_REPO = "Quantumadvancedai/nova-verif-data"
TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "Falta HF_TOKEN"
WORK = "/workspace"
K = 8  # soluciones generadas por problema
MIN_C, MAX_C = 1, K - 1  # "mixto": entre 1 y 7 correctas de 8

from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=TOKEN)


def _baja(fn):
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        print(f"  ya existe: {p}", flush=True)
        return p
    try:
        return hf_hub_download(HF_REPO, fn, repo_type="dataset", token=TOKEN, local_dir=WORK)
    except Exception as e:
        print(f"  no se pudo bajar {fn}: {repr(e)[:120]}", flush=True)
        return None


v1_path = _baja("verif_dataset_v1.jsonl")
numina_path = _baja("verif_dataset_numina.jsonl")
assert v1_path and os.path.exists(v1_path), "Falta verif_dataset_v1.jsonl"
assert numina_path and os.path.exists(numina_path), "Falta verif_dataset_numina.jsonl (ejecuta paso 1 primero)"

# Cargar numina y filtrar mixtos (MIN_C <= correctas <= MAX_C)
por_idx = defaultdict(list)
for l in open(numina_path, encoding="utf-8"):
    l = l.strip()
    if l:
        try: por_idx[json.loads(l)["idx"]].append(json.loads(l))
        except Exception: pass
total_numina = len(por_idx)
mixtos = {idx: sols for idx, sols in por_idx.items()
          if MIN_C <= sum(s.get("etiqueta",0) for s in sols) <= MAX_C}
triviales = sum(1 for idx, sols in por_idx.items() if sum(s.get("etiqueta",0) for s in sols) == K)
imposibles = sum(1 for idx, sols in por_idx.items() if sum(s.get("etiqueta",0) for s in sols) == 0)
print(f"[V2] numina: {total_numina} problemas | mixtos={len(mixtos)} triviales={triviales} imposibles={imposibles}", flush=True)

v2_path = os.path.join(WORK, "verif_dataset_v2.jsonl")
n_v1, n_numina = 0, 0
with open(v2_path, "w", encoding="utf-8") as f:
    # v1 intacto
    for l in open(v1_path, encoding="utf-8"):
        l = l.strip()
        if l: f.write(l + "\n"); n_v1 += 1
    # mixtos de numina
    for idx in sorted(mixtos):
        for sol in mixtos[idx]:
            f.write(json.dumps(sol, ensure_ascii=False) + "\n"); n_numina += 1

filas = n_v1 + n_numina
pos = sum(1 for l in open(v2_path, encoding="utf-8") if l.strip() and json.loads(l.strip()).get("etiqueta")==1)
print(f"[V2] v2: {filas} soluciones ({n_v1} v1 + {n_numina} numina mixtos)", flush=True)
print(f"[V2] correctas={pos} ({100*pos/filas:.1f}%) incorrectas={filas-pos}", flush=True)

# subir v2 a HF
try:
    api.upload_file(path_or_fileobj=v2_path, path_in_repo="verif_dataset_v2.jsonl",
                    repo_id=HF_REPO, repo_type="dataset", commit_message=f"v2: {filas} soluciones")
    print(f"[V2] subido verif_dataset_v2.jsonl a HF", flush=True)
except Exception as e:
    print(f"[V2] aviso subida HF: {repr(e)[:120]}", flush=True)
print("[V2] PASO 2 COMPLETADO", flush=True)
