"""
run_benchmark.py — Evaluación del modelo base en GSM8K / AIME 2024 / GPQA-Diamond.

Self-contained: pega este archivo ENTERO en una celda de un notebook de Kaggle
(con GPU e Internet activados) y ejecuta "Save & Run All (Commit)".
También funciona en local: python run_benchmark.py

Qué hace:
  1. Descarga el benchmark elegido por HTTP directo (sin la librería `datasets`,
     que da problemas de compatibilidad con pyarrow en Kaggle).
  2. Genera una respuesta por problema con DeepSeek-R1-Distill-Qwen-1.5B usando
     la configuración oficial de DeepSeek (sin system prompt, temp 0.6, top_p
     0.95). Si la generación agota el presupuesto de tokens SIN emitir EOS,
     CONTINÚA generando hasta un tope total para evitar truncamientos.
  3. Corrige cada respuesta con una extracción robusta de \\boxed{...}.
  4. Guarda cada resultado al momento en results_{benchmark}.jsonl. Es
     REANUDABLE: si el archivo ya existe, continúa donde se quedó.
  5. Al final imprime la precisión y escribe un informe report_{benchmark}.md.

Las funciones de extracción son importables sin dependencias pesadas:
    from run_benchmark import extraer_boxed, extraer_num, extraer_letra
"""

# ============================== CONFIGURACIÓN ===============================
import os  # arriba para poder leer BENCHMARK y N desde variables de entorno

BENCHMARK = os.environ.get("BENCHMARK", "gsm8k")   # "gsm8k", "aime", "gpqa"
N = int(os.environ.get("N", "1000"))               # nº de problemas a evaluar

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEMPERATURE = 0.6         # recomendación oficial de DeepSeek-R1
TOP_P = 0.95
MAX_NEW_TOKENS = 4096     # ARREGLADO: antes 32768 -> causaba CUDA out-of-memory y horas/problema en T4
MAX_TOTAL_TOKENS = 4096   # ARREGLADO: antes 49152 -> sin continuaciones gigantes que revientan los 15GB

RESULTS_FILE = f"results_{BENCHMARK}.jsonl"
REPORT_FILE = f"report_{BENCHMARK}.md"

# Aquí solo stdlib. torch / transformers / requests / pandas se importan dentro
# de las funciones que los usan, para que las funciones de extracción puedan
# importarse desde cualquier máquina sin instalar nada.
import csv
import datetime
import io
import json
import os
import random
import re
import sys
import time

# Evita la fragmentación de memoria de la GPU (recomendado por PyTorch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ===================== EXTRACCIÓN DE RESPUESTAS (importable) ================

def extraer_boxed(t):
    """Devuelve el contenido del ÚLTIMO \\boxed{...} del texto, manejando
    llaves anidadas de LaTeX (p.ej. \\boxed{\\frac{1}{2}}). None si no hay."""
    idx = t.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + 7  # len("\\boxed{")
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
    """Quita LaTeX y formato de una respuesta numérica ("\\text{km}", "$",
    separadores de miles...) y devuelve el primer número que quede ("" si no hay)."""
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mathrm\{[^}]*\}", "", s)
    for tok in ["\\$", "\\%", "\\,", "\\!", "\\;", "\\ ", "\\left", "\\right", "$", "%", ","]:
        s = s.replace(tok, "")
    s = s.replace("\\", "").strip()
    m = re.search(r"-?\d+\.?\d*", s)
    return m.group(0) if m else ""


def extraer_num(t):
    """Extrae la respuesta numérica de una generación: primero del último
    \\boxed{}; si no lo hay (o está vacío), el último número del texto."""
    bx = extraer_boxed(t)
    if bx is not None:
        n = limpiar(bx)
        if n:
            return n
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "") if nums else ""


def extraer_letra(t):
    """Extrae la letra (A-D) de una respuesta de opción múltiple: primero del
    último \\boxed{}; si no, la última letra mayúscula A-D suelta del texto."""
    bx = extraer_boxed(t)
    if bx is not None:
        m = re.search(r"[A-Da-d]", bx)
        if m:
            return m.group(0).upper()
    sueltas = re.findall(r"\b([A-D])\b", t)
    return sueltas[-1] if sueltas else ""


def comparar_num(pred, gold):
    """True si ambos se interpretan como número y coinciden (tolerancia 1e-6)."""
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False


def corregir(benchmark, texto, correcta):
    """Devuelve (prediccion, acierto) para una generación del modelo."""
    if benchmark == "gpqa":
        pred = extraer_letra(texto)
        return pred, (pred != "" and pred == correcta)
    pred = extraer_num(texto)
    return pred, comparar_num(pred, correcta)


# ============================ CARGA DE DATOS ================================
# Cada loader devuelve una lista de dicts {"pregunta": str, "correcta": str}.
# Todo por HTTP directo con `requests`: evitamos la librería `datasets`.

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/test.jsonl")
AIME_DATASETS = ["Maxwell-Jia/AIME_2024", "HuggingFaceH4/aime_2024"]  # NO gated
GPQA_URL = "https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv"
DS_SERVER = "https://datasets-server.huggingface.co"


def _descargar(url, token=None, intentos=4, timeout=120):
    """GET con reintentos y errores claros. 401/403/404 no se reintentan."""
    import requests
    headers = {"User-Agent": "nova-eval-benchmark"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    ultimo = None
    for k in range(intentos):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (401, 403):
                raise PermissionError(f"HTTP {r.status_code} (sin acceso) en {url}")
            if r.status_code == 404:
                raise FileNotFoundError(f"HTTP 404 (no existe) en {url}")
            r.raise_for_status()
            return r
        except (PermissionError, FileNotFoundError):
            raise  # estos errores no se arreglan reintentando
        except Exception as e:
            ultimo = e
            print(f"  aviso: descarga fallida (intento {k + 1}/{intentos}): {e}", flush=True)
            time.sleep(2 * (k + 1))
    raise RuntimeError(
        f"No se pudo descargar {url} tras {intentos} intentos: {ultimo}\n"
        "¿Está activado Internet en los ajustes del notebook de Kaggle?"
    )


def cargar_gsm8k():
    """GSM8K test (1319 problemas), directo del repo oficial de OpenAI.
    El número correcto va tras '####' en el campo 'answer'."""
    print(f"Descargando GSM8K de {GSM8K_URL} ...", flush=True)
    r = _descargar(GSM8K_URL)
    items = []
    for linea in r.text.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            d = json.loads(linea)
            gold = d["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"pregunta": d["question"].strip(), "correcta": gold})
        except (KeyError, json.JSONDecodeError) as e:
            print(f"  aviso: línea de GSM8K ignorada ({e})", flush=True)
    if not items:
        raise RuntimeError("GSM8K descargado pero sin items válidos")
    return items


def _elegir_campo(claves, candidatos):
    """Devuelve la clave real que coincide (sin distinguir mayúsculas) con algún
    candidato; si no hay coincidencia exacta, la primera que contenga alguno."""
    por_minusculas = {k.lower(): k for k in claves}
    for c in candidatos:
        if c in por_minusculas:
            return por_minusculas[c]
    for k in claves:
        if any(c in k.lower() for c in candidatos):
            return k
    return None


def _normalizar_entero(v):
    """AIME: la respuesta es un entero 0-999; tolera 204, '204' o '204.0'."""
    try:
        return str(int(float(str(v).strip())))
    except (TypeError, ValueError):
        return str(v).strip()


def _aime_via_api(ds):
    """Lee las filas con la API pública datasets-server de HF (JSON puro, sin
    pyarrow). Detecta automáticamente la config y el split publicados."""
    info = _descargar(f"{DS_SERVER}/splits?dataset={ds}").json()
    primera = info["splits"][0]
    config, split = primera["config"], primera["split"]
    filas, offset = [], 0
    while True:
        j = _descargar(f"{DS_SERVER}/rows?dataset={ds}&config={config}"
                       f"&split={split}&offset={offset}&length=100").json()
        filas += [f["row"] for f in j.get("rows", [])]
        total = j.get("num_rows_total", len(filas))
        if not j.get("rows") or len(filas) >= total:
            return filas
        offset += 100


def _aime_via_parquet(ds):
    """Plan B: descarga el parquet auto-convertido de HF y lo lee con pandas
    (sin pasar por la librería `datasets`)."""
    import pandas as pd
    arbol = _descargar(f"https://huggingface.co/api/datasets/{ds}/parquet").json()
    url = None  # estructura {config: {split: [urls]}}: cogemos la primera URL
    for config in arbol.values():
        for urls in config.values():
            if urls:
                url = urls[0]
                break
        if url:
            break
    if not url:
        raise RuntimeError("el dataset no tiene archivos parquet publicados")
    df = pd.read_parquet(io.BytesIO(_descargar(url).content))
    return df.to_dict("records")


def cargar_aime():
    """AIME 2024 (30 problemas) desde fuentes NO restringidas de HuggingFace.
    Inspecciona los nombres de campo reales y se adapta a ellos."""
    ultimo = None
    for ds in AIME_DATASETS:
        for metodo, fn in (("datasets-server", _aime_via_api), ("parquet", _aime_via_parquet)):
            try:
                print(f"Descargando AIME 2024 de {ds} ({metodo}) ...", flush=True)
                filas = fn(ds)
                if not filas:
                    raise RuntimeError("0 filas")
                claves = list(filas[0].keys())
                print(f"  campos detectados: {claves}", flush=True)
                c_problema = _elegir_campo(claves, ["problem", "question", "prompt"])
                c_respuesta = _elegir_campo(claves, ["answer", "expected_answer", "final_answer"])
                if not c_problema or not c_respuesta:
                    raise RuntimeError(f"no reconozco los campos: {claves}")
                print(f"  usando problema={c_problema!r}, respuesta={c_respuesta!r}", flush=True)
                return [{"pregunta": str(f[c_problema]).strip(),
                         "correcta": _normalizar_entero(f[c_respuesta])}
                        for f in filas]
            except Exception as e:
                ultimo = e
                print(f"  aviso: {ds} vía {metodo} falló: {e}", flush=True)
    raise RuntimeError(f"No se pudo cargar AIME 2024 de ninguna fuente. Último error: {ultimo}")


def _token_hf():
    """Busca un token de HuggingFace en: variables de entorno, secretos de
    Kaggle (Add-ons -> Secrets) y el login local de huggingface_hub."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        cliente = UserSecretsClient()
        for nombre in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                v = cliente.get_secret(nombre)
                if v and v.strip():
                    return v.strip()
            except Exception:
                pass
    except Exception:
        pass
    try:
        from huggingface_hub import get_token
        v = get_token()
        if v:
            return v
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder  # API antigua (< 1.0)
        v = HfFolder.get_token()
        if v:
            return v
    except Exception:
        pass
    return None


GPQA_AYUDA = """
================================================================================
ERROR: GPQA-Diamond es un dataset GATED en HuggingFace y no se encontró ningún
token de acceso. Para poder ejecutarlo:

  1. Entra con tu cuenta en https://huggingface.co/datasets/Idavidrein/gpqa
     y acepta las condiciones de acceso del dataset.
  2. Crea un token de lectura en https://huggingface.co/settings/tokens
  3. Dale el token al script:
       - En Kaggle: Add-ons -> Secrets -> crea un secreto llamado HF_TOKEN
         y actívalo (checkbox) para este notebook.
       - En local: define la variable de entorno HF_TOKEN.
================================================================================
"""


def cargar_gpqa():
    """GPQA-Diamond (198 preguntas). Requiere token HF con acceso aceptado:
    si no lo hay, lo dice claramente y para (sin fallo silencioso). Baraja las
    4 opciones de forma determinista por pregunta, así las letras no cambian
    al reanudar una ejecución."""
    token = _token_hf()
    if not token:
        print(GPQA_AYUDA, flush=True)
        sys.exit(1)
    print(f"Descargando GPQA-Diamond de {GPQA_URL} ...", flush=True)
    try:
        r = _descargar(GPQA_URL, token=token)
    except PermissionError as e:
        print(f"\nERROR: el token HF encontrado NO tiene acceso a Idavidrein/gpqa ({e}).\n"
              "Acepta las condiciones del dataset con la cuenta dueña del token en\n"
              "https://huggingface.co/datasets/Idavidrein/gpqa y vuelve a ejecutar.", flush=True)
        sys.exit(1)
    lector = csv.DictReader(io.StringIO(r.text))
    columnas = {c.lower().strip(): c for c in (lector.fieldnames or [])}

    def col(nombre):
        c = columnas.get(nombre.lower())
        if c is None:
            raise RuntimeError(f"columna {nombre!r} no encontrada; hay: {lector.fieldnames}")
        return c

    items = []
    for i, fila in enumerate(lector):
        pregunta = (fila[col("Question")] or "").strip()
        buena = (fila[col("Correct Answer")] or "").strip()
        malas = [(fila[col(f"Incorrect Answer {k}")] or "").strip() for k in (1, 2, 3)]
        if not pregunta or not buena or not all(malas):
            continue
        opciones = [buena] + malas
        # Barajado determinista por índice: reanudar no cambia las letras.
        random.Random(1234 + i).shuffle(opciones)
        letra = "ABCD"[opciones.index(buena)]
        texto_opciones = "\n".join(f"{l}) {o}" for l, o in zip("ABCD", opciones))
        items.append({"pregunta": f"{pregunta}\n\n{texto_opciones}", "correcta": letra})
    if not items:
        raise RuntimeError("GPQA descargado pero sin items válidos")
    return items


def cargar_datos(benchmark):
    if benchmark == "gsm8k":
        return cargar_gsm8k()
    if benchmark == "aime":
        return cargar_aime()
    if benchmark == "gpqa":
        return cargar_gpqa()
    raise ValueError(f"BENCHMARK desconocido: {benchmark!r} (usa 'gsm8k', 'aime' o 'gpqa')")


# ========================== PROMPT Y GENERACIÓN =============================

def construir_prompt(benchmark, pregunta):
    """Contenido del mensaje de usuario. Siguiendo la recomendación oficial de
    DeepSeek-R1, NO se usa system prompt: la instrucción acompaña a la pregunta
    (en gpqa la pregunta ya incluye las opciones A) B) C) D))."""
    if benchmark == "gpqa":
        return (pregunta + "\n\nReason step by step, then put the letter of the "
                "correct option within \\boxed{}.")
    return (pregunta + "\n\nPlease reason step by step, and put your final "
            "answer within \\boxed{}.")


def generar_respuesta(model, tokenizer, contenido, eos_ids):
    """Genera la respuesta completa para un problema.

    Anti-truncamiento: si generate() agota max_new_tokens SIN emitir EOS, se
    vuelve a llamar a generate() pasando todo lo ya generado como contexto,
    hasta que el modelo termine solo o se alcance MAX_TOTAL_TOKENS.

    Devuelve (texto, truncado, n_tokens_generados).
    """
    import torch
    mensajes = [{"role": "user", "content": contenido}]  # sin system prompt
    entrada = tokenizer.apply_chat_template(
        mensajes, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    longitud_prompt = entrada["input_ids"].shape[1]
    salida = entrada["input_ids"]
    generados = 0
    terminado = False
    while not terminado and generados < MAX_TOTAL_TOKENS:
        paso = min(MAX_NEW_TOKENS, MAX_TOTAL_TOKENS - generados)
        salida = model.generate(
            **entrada,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=paso,
            pad_token_id=tokenizer.eos_token_id,
        )
        nuevos = salida.shape[1] - entrada["input_ids"].shape[1]
        generados += nuevos
        # Con batch=1 no hay padding: si el modelo terminó, el último token es EOS;
        # si no lo es, generate() paró por agotar max_new_tokens (truncamiento).
        terminado = salida[0, -1].item() in eos_ids
        if nuevos == 0:  # salvaguarda: no debería ocurrir
            break
        if not terminado and generados < MAX_TOTAL_TOKENS:
            print(f"    sin EOS tras {generados} tokens; continuando la generación...", flush=True)
            entrada = {"input_ids": salida, "attention_mask": torch.ones_like(salida)}
    texto = tokenizer.decode(salida[0, longitud_prompt:], skip_special_tokens=True)
    return texto, (not terminado), generados


# ================== RESULTADOS (incrementales y reanudables) ================

def leer_resultados(ruta):
    """Lee el JSONL ignorando líneas corruptas (p.ej. cortadas a media escritura
    si Kaggle mató la sesión). Si hay índices repetidos, gana la última versión.
    Devuelve la lista ordenada por índice."""
    if not os.path.exists(ruta):
        return []
    por_indice = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                d = json.loads(linea)
                if "i" in d and "acierto" in d:
                    por_indice[d["i"]] = d
            except json.JSONDecodeError:
                continue
    return [por_indice[k] for k in sorted(por_indice)]


def _bloque(texto):
    """Bloque de código markdown con una valla que no choque con el contenido."""
    valla = "```"
    while valla in texto:
        valla += "`"
    return f"{valla}\n{texto}\n{valla}"


def escribir_informe(resultados, n_ok, n_trunc, precision):
    """Escribe report_{BENCHMARK}.md: cabecera con la configuración y las
    métricas, y después el detalle completo de cada problema."""
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    partes = [
        f"# Evaluación: {MODEL_ID} en {BENCHMARK}",
        "",
        f"- **Modelo:** {MODEL_ID} (float16, cuda)",
        f"- **Benchmark:** {BENCHMARK}",
        f"- **Problemas evaluados (N):** {len(resultados)}",
        f"- **Fecha:** {fecha}",
        f"- **Generación:** temperature={TEMPERATURE}, top_p={TOP_P}, "
        f"max_new_tokens={MAX_NEW_TOKENS} (tope total con continuaciones: {MAX_TOTAL_TOKENS})",
        f"- **Precisión:** {n_ok}/{len(resultados)} = {precision:.2f}%",
        f"- **Truncados (sin EOS al agotar el tope total):** {n_trunc}",
        "",
        "---",
        "",
    ]
    for r in resultados:
        estado = "TRUNCADO" if r.get("truncado") else ("CORRECTO" if r["acierto"] else "INCORRECTO")
        partes += [
            f"## Problema {r['i'] + 1} — {estado}",
            "",
            "**Pregunta:**",
            "",
            _bloque(r["pregunta"]),
            "",
            "**Respuesta completa del modelo:**",
            "",
            _bloque(r["respuesta"]),
            "",
            f"**Predicción:** `{r['prediccion'] or '(vacía)'}` · "
            f"**Correcta:** `{r['correcta']}` · **Resultado:** {estado}",
            "",
            "---",
            "",
        ]
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(partes))


# ================================ PRINCIPAL =================================

def main():
    print("=" * 78, flush=True)
    print(f"Benchmark: {BENCHMARK} | N={N} | modelo: {MODEL_ID}", flush=True)
    print(f"Salidas: {RESULTS_FILE} y {REPORT_FILE}", flush=True)
    print("=" * 78, flush=True)

    import torch
    if not torch.cuda.is_available():
        sys.exit("ERROR: no hay GPU CUDA disponible. En Kaggle: Settings -> "
                 "Accelerator -> GPU y vuelve a ejecutar.")

    # ---- datos ----
    datos = cargar_datos(BENCHMARK)
    total = min(N, len(datos))
    if total < N:
        print(f"Aviso: el benchmark solo tiene {len(datos)} problemas; N se ajusta a {total}.",
              flush=True)

    # ---- reanudación: contar lo ya hecho y continuar desde ahí ----
    previos = leer_resultados(RESULTS_FILE)
    hechos = len(previos)
    aciertos = sum(1 for r in previos if r["acierto"])
    if hechos:
        print(f"Reanudando: {hechos} resultados válidos ya guardados en {RESULTS_FILE}.",
              flush=True)

    if hechos < total:
        # ---- modelo (solo se carga si queda trabajo pendiente) ----
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Cargando {MODEL_ID} en float16 sobre cuda ...", flush=True)
        t_carga = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
        model.eval()
        # Ids de fin de generación: los del tokenizer y los del generation_config.
        eos_ids = {tokenizer.eos_token_id}
        eos_cfg = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos_cfg, int):
            eos_ids.add(eos_cfg)
        elif eos_cfg:
            eos_ids.update(eos_cfg)
        eos_ids.discard(None)
        print(f"Modelo cargado en {time.time() - t_carga:.0f}s.", flush=True)

        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            for i in range(hechos, total):
                item = datos[i]
                contenido = construir_prompt(BENCHMARK, item["pregunta"])
                t0 = time.time()
                try:
                    texto, truncado, n_tok = generar_respuesta(model, tokenizer, contenido, eos_ids)
                except Exception as e:
                    # Un fallo puntual (p.ej. OOM) no debe tirar una ejecución de
                    # horas: limpiamos memoria, reintentamos una vez y, si vuelve
                    # a fallar, registramos el error y seguimos.
                    print(f"  aviso: fallo generando el problema {i}: {e}; reintentando...",
                          flush=True)
                    torch.cuda.empty_cache()
                    try:
                        texto, truncado, n_tok = generar_respuesta(model, tokenizer,
                                                                   contenido, eos_ids)
                    except Exception as e2:
                        texto, truncado, n_tok = f"[ERROR DE GENERACIÓN: {e2}]", True, 0
                pred, ok = corregir(BENCHMARK, texto, item["correcta"])
                registro = {
                    "i": i,
                    "pregunta": item["pregunta"],
                    "respuesta": texto,
                    "prediccion": pred,
                    "correcta": item["correcta"],
                    "acierto": ok,
                    "truncado": truncado,
                }
                # Escritura incremental con flush inmediato: si Kaggle corta la
                # sesión, no se pierde nada y la siguiente ejecución reanuda.
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                aciertos += int(ok)
                print(f"[{i + 1}/{total}] {'OK ' if ok else 'MAL'} "
                      f"pred={pred!r} correcta={item['correcta']!r} tokens={n_tok}"
                      f"{' TRUNCADO' if truncado else ''} ({time.time() - t0:.0f}s) "
                      f"| precisión acumulada: {100.0 * aciertos / (i + 1):.1f}%",
                      flush=True)
    else:
        print("No queda nada por evaluar; se regenera solo el informe.", flush=True)

    # ---- resumen final e informe ----
    resultados = leer_resultados(RESULTS_FILE)[:total]
    n_eval = len(resultados)
    n_ok = sum(1 for r in resultados if r["acierto"])
    n_trunc = sum(1 for r in resultados if r.get("truncado"))
    precision = 100.0 * n_ok / n_eval if n_eval else 0.0
    print("\n" + "=" * 78, flush=True)
    print(f"RESULTADO {BENCHMARK}: {n_ok}/{n_eval} = {precision:.2f}% de precisión "
          f"| truncados: {n_trunc}", flush=True)
    print("=" * 78, flush=True)
    escribir_informe(resultados, n_ok, n_trunc, precision)
    print(f"Informe guardado en {REPORT_FILE}", flush=True)


if __name__ == "__main__":
    main()
