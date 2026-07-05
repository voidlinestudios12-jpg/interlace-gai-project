"""
build_rl_dataset.py — PASO 2 de la fase RL: dataset de dificultad intermedia.

Clave de GRPO: si el modelo siempre acierta o nunca acierta un problema, todas
las recompensas del grupo son iguales -> ventaja 0 -> gradiente 0. Solo hay
señal donde acierta A VECES.

Qué hace:
  1. Lee verif_dataset_v1.jsonl (MATH, 400 problemas) y
     verif_dataset_numina.jsonl (Numina, 600 problemas), ambos K=8
     generaciones/problema, y calcula la fracción de aciertos por problema
     (media de `etiqueta` agrupando por `idx`).
  2. Filtra dificultad intermedia: fracción en [0.15, 0.85]. Con K=8 el plan
     lo glosa como 1/8..7/8 -> aplicamos 1 <= aciertos <= 7 (se guarda la
     fracción exacta para poder endurecer el corte después).
  3. Descontaminación (innegociable) por exact-match normalizado del
     enunciado contra: AIME 23/24/25 (local), MATH500, GSM8K test completo y
     GPQA-Diamond. También avisa de casi-duplicados por prefijo normalizado.
  4. Comprueba que el gold es numéricamente comparable (el reward de GRPO usa
     extraer_num/comparar_num del arnés; un gold no parseable = reward
     siempre 0 = hueco muerto).
  5. Escribe rl_dataset_v1.jsonl con {problema, gold, fraccion_base} (+ idx,
     fuente, level para trazabilidad).

Uso:  python nova/rl/build_rl_dataset.py
"""

import hashlib
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nova" / "eval"))
from run_benchmark import limpiar, _descargar, _token_hf  # noqa: E402

VERIF_DIR = REPO_ROOT / "data" / "verif"
ENTRADAS = ["verif_dataset_v1.jsonl", "verif_dataset_numina.jsonl"]
SALIDA = VERIF_DIR / "rl_dataset_v1.jsonl"
AIME90 = REPO_ROOT / "nova" / "data" / "aime_eval_90.json"

MATH500_URL = ("https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
               "resolve/main/test.jsonl")
GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/test.jsonl")
GPQA_URL = "https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv"

FRAC_MIN, FRAC_MAX = 0.15, 0.85  # con K=8: 1/8..7/8 (ver docstring)


def normalizar(texto):
    """Normaliza un enunciado para comparar: minúsculas, sin espacios ni
    signos de puntuación/LaTeX cosmético. Robusto a reformateos triviales."""
    t = texto.lower()
    t = re.sub(r"\\(left|right|,|;|!|quad|qquad|\s)", "", t)
    t = re.sub(r"[\s$`'\"*_~]+", "", t)
    return t


def hash_norm(texto):
    return hashlib.sha256(normalizar(texto).encode()).hexdigest()


def cargar_eval_sets():
    """Devuelve {nombre: set(hashes)} + índice de prefijos para casi-dups."""
    sets = {}
    # AIME 23/24/25 (local)
    aime = json.loads(AIME90.read_text(encoding="utf-8"))
    sets["aime_23_24_25"] = [d["problema"] for d in aime]
    # MATH500
    print("Descargando MATH500 ...", flush=True)
    r = _descargar(MATH500_URL, token=_token_hf())
    sets["math500"] = [json.loads(l)["problem"] for l in r.text.splitlines() if l.strip()]
    # GSM8K test completo (1319; incluye los 250 de eval)
    print("Descargando GSM8K test ...", flush=True)
    r = _descargar(GSM8K_URL)
    sets["gsm8k_test"] = [json.loads(l)["question"] for l in r.text.splitlines() if l.strip()]
    # GPQA-Diamond (gated; requiere token con acceso)
    print("Descargando GPQA-Diamond ...", flush=True)
    import csv
    r = _descargar(GPQA_URL, token=_token_hf())
    lector = csv.DictReader(io.StringIO(r.text))
    col = {c.lower().strip(): c for c in (lector.fieldnames or [])}
    sets["gpqa"] = [fila[col["question"]] for fila in lector if fila.get(col["question"])]
    return sets


def main():
    # ---- 1. fracciones por problema ----
    problemas = {}          # idx -> {problema, gold, fuente, level}
    etiquetas = defaultdict(list)   # idx -> [0/1, ...]
    for nombre in ENTRADAS:
        ruta = VERIF_DIR / nombre
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                d = json.loads(linea)
                idx = d["idx"]
                if idx not in problemas:
                    problemas[idx] = {"idx": idx, "problema": d["problema"],
                                      "gold": str(d["gold"]).strip(),
                                      "fuente": d.get("fuente", ""),
                                      "level": d.get("level", "")}
                elif problemas[idx]["gold"] != str(d["gold"]).strip():
                    print(f"AVISO: gold inconsistente en idx={idx}", flush=True)
                etiquetas[idx].append(int(d["etiqueta"]))
    ks = defaultdict(int)
    for idx, es in etiquetas.items():
        ks[len(es)] += 1
    print(f"Problemas totales: {len(problemas)} | K por problema: {dict(ks)}", flush=True)

    # ---- 2. filtro de dificultad intermedia ----
    candidatos = []
    dist = defaultdict(int)
    for idx, es in etiquetas.items():
        frac = sum(es) / len(es)
        dist[round(frac, 3)] += 1
        # con K=8 el plan glosa [0.15,0.85] como 1/8..7/8 aciertos
        if 1 <= sum(es) <= len(es) - 1:
            p = dict(problemas[idx])
            p["fraccion_base"] = round(frac, 4)
            candidatos.append(p)
    print("Distribución de fracciones:", dict(sorted(dist.items())), flush=True)
    print(f"Intermedios (1..K-1 aciertos): {len(candidatos)}", flush=True)
    estricto = sum(1 for p in candidatos if FRAC_MIN <= p["fraccion_base"] <= FRAC_MAX)
    print(f"  (de ellos, en [0.15, 0.85] estricto: {estricto})", flush=True)

    # ---- 3. gold numéricamente comparable ----
    con_gold = []
    for p in candidatos:
        try:
            float(limpiar(p["gold"]) or p["gold"])
            con_gold.append(p)
        except ValueError:
            print(f"  descartado idx={p['idx']}: gold no numérico {p['gold']!r}", flush=True)
    print(f"Con gold numérico: {len(con_gold)}", flush=True)

    # ---- 4. descontaminación ----
    sets = cargar_eval_sets()
    hashes_eval = {}
    prefijos_eval = {}
    for nombre, textos in sets.items():
        for t in textos:
            hashes_eval[hash_norm(t)] = nombre
            prefijos_eval[normalizar(t)[:150]] = nombre
        print(f"  set de eval {nombre}: {len(textos)} problemas", flush=True)

    limpios, contaminados, sospechosos = [], [], []
    for p in con_gold:
        h = hash_norm(p["problema"])
        pref = normalizar(p["problema"])[:150]
        if h in hashes_eval:
            contaminados.append((p["idx"], hashes_eval[h]))
        elif pref and pref in prefijos_eval:
            sospechosos.append((p["idx"], prefijos_eval[pref]))
        else:
            limpios.append(p)
    print(f"Contaminados (exact-match): {len(contaminados)} {contaminados[:10]}", flush=True)
    print(f"Casi-duplicados (prefijo 150): {len(sospechosos)} {sospechosos[:10]}", flush=True)
    print(f"LIMPIOS: {len(limpios)}", flush=True)
    # los casi-duplicados también fuera: mejor perder 1 problema que contaminar
    del con_gold, candidatos

    # ---- 5. salida ----
    limpios.sort(key=lambda p: p["idx"])
    with open(SALIDA, "w", encoding="utf-8") as f:
        for p in limpios:
            f.write(json.dumps({
                "idx": p["idx"], "problema": p["problema"], "gold": p["gold"],
                "fraccion_base": p["fraccion_base"],
                "fuente": p["fuente"], "level": p["level"],
            }, ensure_ascii=False) + "\n")
    print(f"\nEscrito {SALIDA} con {len(limpios)} problemas.", flush=True)
    if len(limpios) < 300:
        print(f"FALTAN {300 - len(limpios)} para el objetivo de 300: ampliar con "
              "NuminaMath (K=8, temp 0.6) en tandas nocturnas.", flush=True)


if __name__ == "__main__":
    main()
