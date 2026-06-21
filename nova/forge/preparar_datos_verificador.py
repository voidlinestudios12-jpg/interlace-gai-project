"""FASE 3 — PASO 1: dataset del VERIFICADOR (separado y descontaminado de la evaluacion).

Construye el conjunto de entrenamiento del verificador (ORM) con funciones Modal:

  paso 'preparar'  (CPU, barato):
     - Fuente de ENTRENO: MATH train (EleutherAI/hendrycks_math) -> disjunto de MATH500
       (que es el split de TEST). Gold = contenido de \\boxed{} de la solucion de referencia,
       PERO solo nos quedamos con problemas cuyo gold es un ENTERO/DECIMAL limpio (para que
       el etiquetado por comparacion numerica sea fiable, igual que en AIME).
     - DESCONTAMINA por enunciado normalizado contra los benchmarks de EVALUACION:
       MATH500, AIME 2023/2024/2025, GSM8K-test (y GPQA si el token lo permite).
       Chequeo: igualdad normalizada, contencion de subcadena, y solape de palabras (Jaccard).
     - (NUEVO) Puede EXCLUIR enunciados ya usados (p.ej. el v1) para no repetir problemas,
       escribir a un archivo de salida distinto y numerar con un idx_base propio.
     - Selecciona `target` problemas limpios y los guarda en /data/verif_problemas.jsonl.

  paso 'generar'  (A100):
     - Para cada problema genera K soluciones con Nova-v0 (temp 0.6, top_p 0.95), GUARDANDO
       el texto completo de cada una.
     - Etiqueta cada solucion correcto/incorrecto comparando su respuesta con el gold
       (extractor robusto del arnes). Reanudable y con commit por lote.
     - (NUEVO) Acepta archivo de problemas / dataset de salida propios y un `limite` (para
       pilotos baratos: generar solo los primeros N problemas pendientes).
     - Salida FLAT (una linea por solucion):
         {idx, problema, gold, fuente, level, texto, pred, certeza, etiqueta, truncado}

  paso 'filtrar'  (CPU, barato) [NUEVO — enriquecimiento]:
     - Construye v2 = v1 (intacto) + los problemas DIFICILES que resultaron MIXTOS
       (entre min_c y max_c correctas de K). Los triviales/imposibles se descartan.

El verificador se entrena (PASO 2) con ESTE dataset; AIME/GPQA/MATH500/GSM8K son SOLO
evaluacion. El gold aqui se usa unicamente para ETIQUETAR el dataset de entrenamiento.

Uso (v1 original):
  modal run nova/forge/preparar_datos_verificador.py --paso preparar --target 400
  modal run nova/forge/preparar_datos_verificador.py --paso generar  --k 8
Uso (enriquecimiento con problemas dificiles, Nivel 5, sin repetir el v1):
  modal run nova/forge/preparar_datos_verificador.py --paso preparar --niveles "Level 5" \
      --target 600 --seed 7 --excluir /data/verif_problemas.jsonl \
      --out /data/verif_problemas_hard.jsonl --idx-base 400
  modal run nova/forge/preparar_datos_verificador.py --paso generar --k 8 \
      --problemas /data/verif_problemas_hard.jsonl --dataset /data/verif_dataset_hard.jsonl --limite 40
  modal run nova/forge/preparar_datos_verificador.py --paso filtrar
Descargar:
  modal volume get nova-data verif_dataset_v2.jsonl ./resultados/
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
NUMINA_DS = "AI-MO/NuminaMath-1.5"


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


def _gold_limpio_answer(a):
    """Gold para NuminaMath: usa el campo 'answer' DIRECTO, quedandonos SOLO con
    enteros/decimales limpios (etiquetado numerico fiable). None si es 'proof',
    fraccion, expresion, etc."""
    import re
    if a is None:
        return None
    s = str(a).strip()
    for tok in ("\\boxed{", "}", "\\$", "$", ",", "\\!", "\\,", "\\ ", "\\left", "\\right", " "):
        s = s.replace(tok, "")
    if re.fullmatch(r"-?\d+\.?\d*", s):
        try:
            float(s)
            return s
        except ValueError:
            return None
    return None


def _cargar_numina(sources):
    """NuminaMath-1.5 (competicion/olimpiada, mas parecido a AIME): descarga el parquet
    por shards, lee SOLO las columnas necesarias (memoria acotada) y filtra a las fuentes
    pedidas, no-sinteticas, validas y con 'answer' entero/decimal limpio."""
    import io
    import pandas as pd
    import run_benchmark as rb
    arbol = rb._descargar(f"https://huggingface.co/api/datasets/{NUMINA_DS}/parquet").json()
    cols = ["problem", "answer", "source", "synthetic", "problem_is_valid"]
    filas = []
    n_shards = 0
    for config, splits in arbol.items():
        for split, urls in splits.items():
            if split != "train":
                continue
            for url in urls:
                n_shards += 1
                df = pd.read_parquet(io.BytesIO(rb._descargar(url).content), columns=cols)
                for r in df.to_dict("records"):
                    if str(r.get("source")) not in sources:
                        continue
                    if r.get("synthetic") in (True, "True", "true"):
                        continue
                    if str(r.get("problem_is_valid")).strip().lower() not in ("yes", "true", "1"):
                        continue
                    gold = _gold_limpio_answer(r.get("answer"))
                    if gold is None:
                        continue
                    filas.append({"problema": str(r.get("problem", "")), "gold": gold,
                                  "level": str(r.get("source")), "fuente": str(r.get("source"))})
                del df
    print(f"  NuminaMath: {len(filas)} candidatos limpios de {sorted(sources)} ({n_shards} shards)", flush=True)
    return filas


# =============================== PASO PREPARAR =============================

@app.function(image=image_cpu, volumes={"/data": vol}, timeout=60 * 40, memory=16384)
def preparar(target, niveles_csv, seed, excluir, out_file, idx_base, fuente, sources_csv):
    import json
    import os
    import random
    import sys
    sys.path.insert(0, "/root")
    import run_benchmark as rb  # noqa: F401  (lo usan las funciones de arriba)

    out_file = out_file or PROBLEMAS_FILE
    niveles = set(x.strip() for x in niveles_csv.split(",") if x.strip())
    print(f"[PREP] fuente={fuente} niveles={sorted(niveles)} sources={sources_csv} "
          f"target={target} seed={seed} salida={out_file} idx_base={idx_base}", flush=True)

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

    # 1b) (NUEVO) enunciados a EXCLUIR (p.ej. el v1): se tratan como contaminacion
    #     para no volver a elegir problemas ya usados.
    n_excluidos = 0
    for ruta in [x.strip() for x in (excluir or "").split(",") if x.strip()]:
        if not os.path.exists(ruta):
            print(f"[PREP] aviso: excluir '{ruta}' no existe; ignorado", flush=True)
            continue
        for linea in open(ruta, encoding="utf-8"):
            linea = linea.strip()
            if not linea:
                continue
            pr = json.loads(linea).get("problema", "")
            n = _norm_stmt(pr)
            if n:
                eval_norms.add(n)
                eval_shingles.append(_shingle(n))
                n_excluidos += 1
    print(f"[PREP] excluidos anyadidos al filtro (no se repetiran): {n_excluidos}", flush=True)
    print(f"[PREP] descontaminacion/exclusion contra {len(eval_norms)} enunciados", flush=True)

    # 2) cargar pool de candidatos segun la fuente
    if fuente == "numina":
        sources = set(x.strip() for x in sources_csv.split(",") if x.strip())
        pool = _cargar_numina(sources)
        print(f"[PREP] candidatos NuminaMath con answer limpio: {len(pool)}", flush=True)
    else:
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
        print(f"[PREP] candidatos MATH con gold limpio y nivel valido: {len(pool)} "
              f"(descartados sin gold limpio: {sin_gold})", flush=True)

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
    print(f"[PREP] elegidos={len(elegidos)} contaminados/excluidos_descartados={contaminados}", flush=True)

    # 4) guardar problemas + meta de descontaminacion
    with open(out_file, "w", encoding="utf-8") as f:
        for i, c in enumerate(elegidos):
            f.write(json.dumps({"idx": idx_base + i, **c}, ensure_ascii=False) + "\n")
    meta = {"target": target, "elegidos": len(elegidos), "contaminados_descartados": contaminados,
            "excluidos": n_excluidos, "idx_base": idx_base,
            "candidatos_pool": len(pool), "niveles": sorted(niveles),
            "eval_enunciados": {k: len(v) for k, v in fuentes.items()}, "seed": seed}
    meta_file = out_file.rsplit(".", 1)[0] + "_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    vol.commit()
    print(f"[PREP] guardados {len(elegidos)} problemas -> {out_file}", flush=True)
    return meta


# =============================== PASO GENERAR =============================

@app.function(gpu="A100", image=image_gpu, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 6)
def generar(k, max_tokens, chunk_p, problemas_file, dataset_file, limite):
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

    prob_file = problemas_file or PROBLEMAS_FILE
    ds_file = dataset_file or DATASET_FILE
    report_file = ds_file.rsplit(".", 1)[0] + "_report.md"

    if not os.path.exists(prob_file):
        raise RuntimeError(f"No existe {prob_file}; ejecuta primero --paso preparar")
    problemas = []
    for linea in open(prob_file, encoding="utf-8"):
        linea = linea.strip()
        if linea:
            problemas.append(json.loads(linea))
    print(f"[GEN] {len(problemas)} problemas, K={k}, max_tokens={max_tokens} "
          f"(salida={ds_file}, limite={limite})", flush=True)

    # reanudar: idx ya presentes en el dataset
    hechos = set()
    if os.path.exists(ds_file):
        for linea in open(ds_file, encoding="utf-8"):
            linea = linea.strip()
            if not linea:
                continue
            try:
                hechos.add(json.loads(linea)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    pend = [p for p in problemas if p["idx"] not in hechos]
    if limite and limite > 0:
        pend = pend[:limite]
    print(f"[GEN] ya={len(hechos)} pendientes={len(pend)}", flush=True)

    if pend:
        llm = mb.crear_llm(max_model_len=max_tokens + 2048)
        sp = SamplingParams(n=k, temperature=mb.TEMPERATURE, top_p=mb.TOP_P,
                            max_tokens=max_tokens, logprobs=1)
        with open(ds_file, "a", encoding="utf-8") as f:
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
    filas = [json.loads(l) for l in open(ds_file, encoding="utf-8") if l.strip()]
    pos = sum(1 for r in filas if r["etiqueta"] == 1)
    neg = len(filas) - pos
    trunc = sum(1 for r in filas if r.get("truncado"))
    n_prob = len({r["idx"] for r in filas})
    meta = {}
    meta_p = prob_file.rsplit(".", 1)[0] + "_meta.json"
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p, encoding="utf-8"))

    P = ["# Dataset del verificador — Fase 3", "",
         f"- Archivo: {ds_file}",
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
              f"- Descartados por contaminacion/exclusion con evaluacion: {meta.get('contaminados_descartados')}.",
              "- Enunciados de evaluacion usados para descontaminar:"]
        for kk, vv in meta.get("eval_enunciados", {}).items():
            P.append(f"  - {kk}: {vv}")
        P += ["", "El verificador se entrena con ESTE dataset; AIME/GPQA/MATH500/GSM8K son solo evaluacion."]
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    vol.commit()
    print(f"[GEN] dataset: {len(filas)} soluciones, {pos} correctas / {neg} incorrectas, {n_prob} problemas", flush=True)
    print(f"[GEN] informe -> {report_file}", flush=True)
    return {"soluciones": len(filas), "correctas": pos, "incorrectas": neg, "problemas": n_prob, "truncadas": trunc}


# ============================== PASO FILTRAR ===============================

@app.function(image=image_cpu, volumes={"/data": vol}, timeout=60 * 20)
def filtrar(v1_file, hard_file, out_file, min_c, max_c):
    """Enriquecimiento: v2 = v1 (intacto) + problemas DIFICILES que salieron MIXTOS
    (entre min_c y max_c correctas de K). Descarta triviales (todas correctas) e
    imposibles (ninguna correcta), que no ayudan a discriminar."""
    import json
    import os
    from collections import defaultdict

    def cargar(ruta):
        if not os.path.exists(ruta):
            return []
        return [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]

    v1 = cargar(v1_file)
    hard = cargar(hard_file)
    por = defaultdict(list)
    for r in hard:
        por[r["idx"]].append(r["etiqueta"])
    mixtos = {i for i, et in por.items() if min_c <= sum(et) <= max_c}
    triviales = {i for i, et in por.items() if sum(et) == len(et)}
    imposibles = {i for i, et in por.items() if sum(et) == 0}
    hard_mix = [r for r in hard if r["idx"] in mixtos]
    salida = v1 + hard_mix

    with open(out_file, "w", encoding="utf-8") as f:
        for r in salida:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()

    pos = sum(1 for r in salida if r["etiqueta"] == 1)
    n_v1_prob = len({r["idx"] for r in v1})
    P = ["# Dataset v2 (enriquecido) — Fase 3", "",
         f"- v1: {len(v1)} soluciones / {n_v1_prob} problemas.",
         f"- Dificiles generados: {len(hard)} soluciones / {len(por)} problemas.",
         f"  - MIXTOS (1..{max_c} correctas, utiles): **{len(mixtos)}** problemas -> {len(hard_mix)} soluciones AÑADIDAS.",
         f"  - triviales (todas correctas, descartados): {len(triviales)}.",
         f"  - imposibles (0 correctas, descartados): {len(imposibles)}.",
         f"- **v2 = {len(salida)} soluciones** ({pos} correctas / {len(salida) - pos} incorrectas), "
         f"{n_v1_prob + len(mixtos)} problemas.", ""]
    with open("/data/verif_dataset_v2_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    vol.commit()
    print(f"[FILT] v1={len(v1)} sol; dificiles={len(hard)} sol ({len(por)} problemas: "
          f"{len(mixtos)} mixtos, {len(triviales)} triviales, {len(imposibles)} imposibles)", flush=True)
    print(f"[FILT] v2={len(salida)} sol ({pos} correctas / {len(salida) - pos} incorrectas) -> {out_file}", flush=True)
    return {"v1_sol": len(v1), "hard_sol": len(hard), "hard_problemas": len(por),
            "mixtos": len(mixtos), "triviales": len(triviales), "imposibles": len(imposibles),
            "v2_sol": len(salida), "v2_correctas": pos}


@app.local_entrypoint()
def main(paso: str = "preparar", target: int = 400, niveles: str = "Level 3,Level 4,Level 5",
         seed: int = 1234, k: int = 8, max_tokens: int = 12288, chunk_p: int = 8,
         excluir: str = "", out: str = "", idx_base: int = 0,
         fuente: str = "math", sources: str = "olympiads,aops_forum",
         problemas: str = "", dataset: str = "", limite: int = 0,
         v1: str = "/data/verif_dataset_v1.jsonl", hard: str = "/data/verif_dataset_hard.jsonl",
         salida: str = "/data/verif_dataset_v2.jsonl", min_c: int = 1, max_c: int = 7,
         spawn: bool = False):
    if paso == "preparar":
        print(preparar.remote(target, niveles, seed, excluir, out, idx_base, fuente, sources))
    elif paso == "generar":
        if spawn:
            # Fire-and-forget: con `modal run --detach` la funcion sigue en la nube
            # AUNQUE el cliente local se desconecte (apagar el PC, cerrar la terminal).
            fc = generar.spawn(k, max_tokens, chunk_p, problemas, dataset, limite)
            print(f"[SPAWN] generar lanzado en la nube. FunctionCall id: {fc.object_id}")
        else:
            print(generar.remote(k, max_tokens, chunk_p, problemas, dataset, limite))
    elif paso == "filtrar":
        print(filtrar.remote(v1, hard, salida, min_c, max_c))
    else:
        raise SystemExit("paso debe ser 'preparar', 'generar' o 'filtrar'")
