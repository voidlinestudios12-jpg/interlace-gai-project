"""FASE 3 — PASO 1: dataset del VERIFICADOR (separado y descontaminado de la evaluacion).

Construye el conjunto de entrenamiento del verificador (ORM) con DOS funciones Modal:

  paso 'preparar'  (CPU, barato):
     - Fuente de ENTRENO: MATH train (EleutherAI/hendrycks_math) -> disjunto de MATH500
       (que es el split de TEST). Gold = contenido de \\boxed{} de la solucion de referencia,
       PERO solo nos quedamos con problemas cuyo gold es un ENTERO/DECIMAL limpio (para que
       el etiquetado por comparacion numerica sea fiable, igual que en AIME).
     - DESCONTAMINA por enunciado normalizado contra los benchmarks de EVALUACION:
       MATH500, AIME 2023/2024/2025, GSM8K-test (y GPQA si el token lo permite).
       Chequeo: igualdad normalizada, contencion de subcadena, y solape de palabras (Jaccard).
     - Selecciona `target` problemas limpios y los guarda en /data/verif_problemas.jsonl.

  paso 'generar'  (A100):
     - Para cada problema genera K soluciones con Nova-v0 (temp 0.6, top_p 0.95), GUARDANDO
       el texto completo de cada una.
     - Etiqueta cada solucion correcto/incorrecto comparando su respuesta con el gold
       (extractor robusto del arnes). Reanudable y con commit por lote.
     - Salida FLAT (una linea por solucion) en /data/verif_dataset_v1.jsonl:
         {idx, problema, gold, fuente, level, texto, pred, certeza, etiqueta, truncado}
     - Escribe un informe con el balance correcto/incorrecto y la descontaminacion.

El verificador se entrena (PASO 2) con ESTE dataset; AIME/GPQA/MATH500/GSM8K son SOLO
evaluacion. El gold aqui se usa unicamente para ETIQUETAR el dataset de entrenamiento.

Uso:
  modal run nova/forge/preparar_datos_verificador.py --paso preparar --target 400
  modal run nova/forge/preparar_datos_verificador.py --paso generar  --k 8
Descargar:
  modal volume get nova-data verif_dataset_v1.jsonl ./resultados/
"""
import modal

app = modal.App("nova-verif-datos")
vol = modal.Volume.from_name("nova-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

_archivos = (
    lambda img: img
    .add_local_file("nova/eval/run_benchmark.py", "/root/run_benchmark.py")
    .add_local_file("shared/modelo_base.py", "/root/modelo_base.py")
    .add_local_file("nova/inference/verificadores.py", "/root/verificadores.py")
)
image_cpu = _archivos(
    modal.Image.debian_slim(python_version="3.11").pip_install("requests", "pandas", "pyarrow", "sympy")
)
image_gpu = _archivos(
    modal.Image.debian_slim(python_version="3.11").pip_install("vllm", "requests", "pandas", "sympy")
)

PROBLEMAS_FILE = "/data/verif_problemas.jsonl"
DATASET_FILE = "/data/verif_dataset_v1.jsonl"
REPORT_FILE = "/data/verif_dataset_report.md"
DS_SERVER = "https://datasets-server.huggingface.co"
MATH_TRAIN_DS = "EleutherAI/hendrycks_math"


# ============================ DESCONTAMINACION =============================

def _norm_stmt(s):
    """Forma canonica de un enunciado para comparar: minusculas, sin comandos LaTeX,
    solo alfanumericos colapsados en palabras separadas por un espacio."""
    import re
    s = (s or "").lower()
    s = re.sub(r"\\[a-zA-Z]+", " ", s)     # comandos LaTeX (\frac, \boxed, ...)
    s = re.sub(r"[^a-z0-9]+", " ", s)      # quita simbolos/puntuacion/$
    return " ".join(s.split())


def _shingle(norm, k=5):
    """Conjunto de k-gramas de palabras (para solape Jaccard)."""
    pal = norm.split()
    if len(pal) < k:
        return frozenset([" ".join(pal)]) if pal else frozenset()
    return frozenset(" ".join(pal[i:i + k]) for i in range(len(pal) - k + 1))


def _contaminado(norm_train, sh_train, eval_norms_set, eval_shingles, umbral=0.6):
    """True si el problema de entreno coincide con algun problema de evaluacion:
    (a) enunciado normalizado identico, (b) uno contiene al otro como subcadena
    (si es largo), o (c) solape Jaccard de 5-gramas >= umbral."""
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


# =========================== CARGA DE FUENTES ==============================

def _rows_api(ds, config=None, split=None, limite=0):
    """Lee filas de un dataset por la API publica datasets-server (sin pyarrow)."""
    import run_benchmark as rb
    if config is None or split is None:
        info = rb._descargar(f"{DS_SERVER}/splits?dataset={ds}").json()
        pr = info["splits"][0]
        config = config or pr["config"]
        split = split or pr["split"]
    filas, offset = [], 0
    while True:
        j = rb._descargar(f"{DS_SERVER}/rows?dataset={ds}&config={config}"
                          f"&split={split}&offset={offset}&length=100").json()
        nuevas = [f["row"] for f in j.get("rows", [])]
        filas += nuevas
        total = j.get("num_rows_total", len(filas))
        if not nuevas or len(filas) >= total or (limite and len(filas) >= limite):
            return filas[:limite] if limite else filas
        offset += 100


def _cargar_eval_statements():
    """Enunciados de TODOS los benchmarks de evaluacion (para descontaminar)."""
    import run_benchmark as rb
    fuentes = {}
    # MATH500 (split test) -> problem
    try:
        filas = _rows_api("HuggingFaceH4/MATH-500", "default", "test")
        fuentes["MATH500"] = [str(f.get("problem", "")) for f in filas]
    except Exception as e:
        print(f"  aviso: MATH500 no cargado: {e}", flush=True)
    # AIME 2023 + 2024 (gneubig 1983-2024) -> Question, filtrando Year
    try:
        filas = _rows_api("gneubig/aime-1983-2024", "default", "train")
        aime2324 = [str(f.get("Question", "")) for f in filas if int(f.get("Year", 0)) in (2023, 2024)]
        fuentes["AIME_2023_2024"] = aime2324
    except Exception as e:
        print(f"  aviso: AIME 2023/2024 no cargado: {e}", flush=True)
    # AIME 2025 -> problem
    try:
        filas = _rows_api("math-ai/aime25", "default", "test")
        fuentes["AIME_2025"] = [str(f.get("problem", "")) for f in filas]
    except Exception as e:
        print(f"  aviso: AIME 2025 no cargado: {e}", flush=True)
    # GSM8K test -> pregunta (repo oficial OpenAI)
    try:
        fuentes["GSM8K_test"] = [d["pregunta"] for d in rb.cargar_gsm8k()]
    except Exception as e:
        print(f"  aviso: GSM8K-test no cargado: {e}", flush=True)
    # GPQA-Diamond -> pregunta (necesita token; ciencia, disjunto de mates, best-effort)
    try:
        fuentes["GPQA"] = [d["pregunta"] for d in rb.cargar_gpqa()]
    except SystemExit:
        print("  aviso: GPQA omitido (sin token); es ciencia, disjunto de mates por dominio.", flush=True)
    except Exception as e:
        print(f"  aviso: GPQA no cargado ({e}); disjunto de mates por dominio.", flush=True)
    return fuentes


def _gold_limpio(solucion):
    """Devuelve el gold SOLO si el \\boxed de la solucion de referencia es un entero o
    decimal limpio (etiquetado numerico fiable). None en otro caso (fracciones, expresiones)."""
    import re
    import run_benchmark as rb
    bx = rb.extraer_boxed(solucion)
    if bx is None:
        return None
    s = bx.strip()
    for tok in ("\\$", "$", ",", "\\!", "\\,", "\\ ", "\\left", "\\right", " "):
        s = s.replace(tok, "")
    if re.fullmatch(r"-?\d+\.?\d*", s):
        try:
            float(s)
            return s
        except ValueError:
            return None
    return None


def _cargar_math_train():
    """MATH train (todas las materias) via parquet auto-convertido de HF."""
    import io
    import pandas as pd
    import run_benchmark as rb
    arbol = rb._descargar(f"https://huggingface.co/api/datasets/{MATH_TRAIN_DS}/parquet").json()
    filas = []
    # estructura {config: {split: [urls]}}
    for config, splits in arbol.items():
        urls = splits.get("train", [])
        for url in urls:
            df = pd.read_parquet(io.BytesIO(rb._descargar(url).content))
            for r in df.to_dict("records"):
                filas.append({"problem": str(r.get("problem", "")),
                              "solution": str(r.get("solution", "")),
                              "level": str(r.get("level", "")),
                              "type": str(r.get("type", config))})
    print(f"  MATH train: {len(filas)} problemas en bruto ({len(arbol)} materias)", flush=True)
    return filas


# =============================== PASO PREPARAR =============================

@app.function(image=image_cpu, volumes={"/data": vol}, timeout=60 * 40)
def preparar(target, niveles_csv, seed):
    import json
    import random
    import sys
    sys.path.insert(0, "/root")
    import run_benchmark as rb  # noqa: F401  (lo usan las funciones de arriba)

    niveles = set(x.strip() for x in niveles_csv.split(",") if x.strip())
    print(f"[PREP] niveles={sorted(niveles)} target={target} seed={seed}", flush=True)

    # 1) enunciados de evaluacion -> conjuntos normalizados + shingles
    fuentes = _cargar_eval_statements()
    eval_norms, eval_shingles = set(), []
    for nombre, lst in fuentes.items():
        print(f"[PREP] eval '{nombre}': {len(lst)} enunciados", flush=True)
        for s in lst:
            n = _norm_stmt(s)
            if n:
                eval_norms.add(n)
                eval_shingles.append(_shingle(n))
    print(f"[PREP] descontaminacion contra {len(eval_norms)} enunciados de evaluacion", flush=True)

    # 2) MATH train -> filtrar por nivel + gold entero/decimal limpio
    bruto = _cargar_math_train()
    pool = []
    sin_gold = 0
    for r in bruto:
        nivel_ok = (not niveles) or any(nv in r["level"] for nv in niveles)
        if not nivel_ok:
            continue
        gold = _gold_limpio(r["solution"])
        if gold is None:
            sin_gold += 1
            continue
        pool.append({"problema": r["problem"], "gold": gold, "level": r["level"], "fuente": r["type"]})
    print(f"[PREP] candidatos con gold limpio y nivel valido: {len(pool)} (descartados sin gold limpio: {sin_gold})", flush=True)

    # 3) descontaminar (chequeo profundo solo sobre los seleccionados, hasta llegar a target)
    random.Random(seed).shuffle(pool)
    elegidos, contaminados = [], 0
    for c in pool:
        norm = _norm_stmt(c["problema"])
        if norm in eval_norms:
            contaminados += 1
            continue
        if _contaminado(norm, _shingle(norm), eval_norms, eval_shingles):
            contaminados += 1
            continue
        elegidos.append(c)
        if len(elegidos) >= target:
            break
    print(f"[PREP] elegidos={len(elegidos)} contaminados_descartados={contaminados}", flush=True)

    # 4) guardar problemas + meta de descontaminacion
    with open(PROBLEMAS_FILE, "w", encoding="utf-8") as f:
        for i, c in enumerate(elegidos):
            f.write(json.dumps({"idx": i, **c}, ensure_ascii=False) + "\n")
    meta = {"target": target, "elegidos": len(elegidos), "contaminados_descartados": contaminados,
            "candidatos_pool": len(pool), "niveles": sorted(niveles),
            "eval_enunciados": {k: len(v) for k, v in fuentes.items()}, "seed": seed}
    with open("/data/verif_problemas_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    vol.commit()
    print(f"[PREP] guardados {len(elegidos)} problemas -> {PROBLEMAS_FILE}", flush=True)
    return meta


# =============================== PASO GENERAR =============================

@app.function(gpu="A100", image=image_gpu, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 6)
def generar(k, max_tokens, chunk_p):
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache"
    os.environ["BENCHMARK"] = "math"
    import modelo_base as mb
    import run_benchmark as rb
    import verificadores as V
    from vllm import SamplingParams

    if not os.path.exists(PROBLEMAS_FILE):
        raise RuntimeError(f"No existe {PROBLEMAS_FILE}; ejecuta primero --paso preparar")
    problemas = []
    for linea in open(PROBLEMAS_FILE, encoding="utf-8"):
        linea = linea.strip()
        if linea:
            problemas.append(json.loads(linea))
    print(f"[GEN] {len(problemas)} problemas, K={k}, max_tokens={max_tokens}", flush=True)

    # reanudar: idx ya presentes en el dataset
    hechos = set()
    if os.path.exists(DATASET_FILE):
        for linea in open(DATASET_FILE, encoding="utf-8"):
            linea = linea.strip()
            if not linea:
                continue
            try:
                hechos.add(json.loads(linea)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    pend = [p for p in problemas if p["idx"] not in hechos]
    print(f"[GEN] ya={len(hechos)} pendientes={len(pend)}", flush=True)

    if pend:
        llm = mb.crear_llm(max_model_len=max_tokens + 2048)
        sp = SamplingParams(n=k, temperature=mb.TEMPERATURE, top_p=mb.TOP_P,
                            max_tokens=max_tokens, logprobs=1)
        with open(DATASET_FILE, "a", encoding="utf-8") as f:
            for s in range(0, len(pend), chunk_p):
                lote = pend[s:s + chunk_p]
                conv = [mb.mensajes(rb.construir_prompt("math", p["problema"])) for p in lote]
                outs = llm.chat(conv, sp)
                for p, out in zip(lote, outs):
                    for o in out.outputs:
                        nt = len(o.token_ids)
                        cert = (o.cumulative_logprob / nt) if (o.cumulative_logprob is not None and nt) else 0.0
                        pred = V.extraer_respuesta("math", o.text)
                        etiqueta = 1 if V.es_correcto("math", pred, p["gold"]) else 0
                        f.write(json.dumps({
                            "idx": p["idx"], "problema": p["problema"], "gold": p["gold"],
                            "fuente": p["fuente"], "level": p["level"],
                            "texto": o.text, "pred": pred, "certeza": cert,
                            "etiqueta": etiqueta, "truncado": (o.finish_reason == "length"),
                        }, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                vol.commit()
                print(f"[GEN] procesados {min(s + chunk_p, len(pend))}/{len(pend)} problemas", flush=True)

    # --- balance + informe ---
    filas = [json.loads(l) for l in open(DATASET_FILE, encoding="utf-8") if l.strip()]
    pos = sum(1 for r in filas if r["etiqueta"] == 1)
    neg = len(filas) - pos
    trunc = sum(1 for r in filas if r.get("truncado"))
    n_prob = len({r["idx"] for r in filas})
    meta = {}
    if os.path.exists("/data/verif_problemas_meta.json"):
        meta = json.load(open("/data/verif_problemas_meta.json", encoding="utf-8"))

    P = ["# Dataset del verificador (v1) — Fase 3", "",
         f"- Problemas (MATH train, descontaminados): **{n_prob}**",
         f"- Soluciones generadas (Nova-v0, K={k}): **{len(filas)}**",
         f"- Correctas (etiqueta=1): **{pos}** ({100 * pos / max(1, len(filas)):.1f}%)",
         f"- Incorrectas (etiqueta=0): **{neg}** ({100 * neg / max(1, len(filas)):.1f}%)",
         f"- Truncadas (sin EOS): {trunc}",
         f"- Generacion: temp {mb.TEMPERATURE} / top_p {mb.TOP_P} / max_tokens {max_tokens}", ""]
    if meta:
        P += ["## Descontaminacion (PASO 1)",
              f"- Fuente de entreno: MATH train ({MATH_TRAIN_DS}), niveles {meta.get('niveles')}.",
              f"- Candidatos con gold limpio: {meta.get('candidatos_pool')}.",
              f"- Descartados por contaminacion con evaluacion: {meta.get('contaminados_descartados')}.",
              "- Enunciados de evaluacion usados para descontaminar:"]
        for kk, vv in meta.get("eval_enunciados", {}).items():
            P.append(f"  - {kk}: {vv}")
        P += ["", "El verificador se entrena con ESTE dataset; AIME/GPQA/MATH500/GSM8K son solo evaluacion."]
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    vol.commit()
    print(f"[GEN] dataset: {len(filas)} soluciones, {pos} correctas / {neg} incorrectas, {n_prob} problemas", flush=True)
    print(f"[GEN] informe -> {REPORT_FILE}", flush=True)
    return {"soluciones": len(filas), "correctas": pos, "incorrectas": neg, "problemas": n_prob, "truncadas": trunc}


@app.local_entrypoint()
def main(paso: str = "preparar", target: int = 400, niveles: str = "Level 3,Level 4,Level 5",
         seed: int = 1234, k: int = 8, max_tokens: int = 12288, chunk_p: int = 8):
    if paso == "preparar":
        print(preparar.remote(target, niveles, seed))
    elif paso == "generar":
        print(generar.remote(k, max_tokens, chunk_p))
    else:
        raise SystemExit("paso debe ser 'preparar' o 'generar'")
