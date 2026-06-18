"""FASE 1 — Motor de computo en inferencia, sobre Modal (plantilla de la §5.2, adaptada a vLLM).

Para cada problema genera N_max muestras (vLLM, temp 0.6 / top_p 0.95), guarda las
muestras crudas en el volumen persistente `nova-data` (reanudable, con flush), y traza
la curva PRECISION vs N para los tres selectores (mayoria / autocerteza / verificador).
Genera report_ttc_{benchmark}.md en el volumen.

Reutiliza el arnes validado (nova/eval/run_benchmark.py): carga de datos, prompt y grading.

Uso (desde la raiz del repo):
  # prueba rapida (5 problemas de AIME, N=4, GPU T4):
  modal run nova/inference/run_ttc_modal.py --benchmark aime --n-max 4 --num-problemas 5 --max-tokens 8192 --gpu T4
  # barrido completo (en 2o plano):
  modal run --detach nova/inference/run_ttc_modal.py --benchmark aime --n-max 32 --gpu A100
Descargar:  modal volume get nova-data report_ttc_aime.md ./
"""
import modal

app = modal.App("nova-ttc")

# Volumenes persistentes: resultados (nova-data, §5.2) y cache del modelo (para no re-descargar).
vol = modal.Volume.from_name("nova-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Imagen: vLLM (trae torch+transformers) + sympy (verificadores) + requests/pandas (carga de datos del arnes).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "requests", "pandas")  # mismas deps que el run anterior -> capa cacheada
    .pip_install("sympy")                         # extra pequeno para los verificadores
    .add_local_file("nova/eval/run_benchmark.py", "/root/run_benchmark.py")
    .add_local_file("shared/modelo_base.py", "/root/modelo_base.py")
    .add_local_file("nova/inference/verificadores.py", "/root/verificadores.py")
    .add_local_file("nova/inference/motor.py", "/root/motor.py")
)

BASELINES = {"gsm8k": 85.70, "aime": 23.33, "gpqa": 37.37}  # de docs/benchmarks/RESUMEN.md
SWEEP = [1, 4, 8, 16, 32]
SELECTORES = ["mayoria", "autocerteza", "verificador"]


def _escribir_report(benchmark, tabla, registros, n_max, max_tokens, Ns, tag=""):
    """Escribe report_ttc_{benchmark}.md con la tabla precision-vs-N y ejemplos (auditoria)."""
    import datetime
    import modelo_base as mb
    import motor
    import verificadores as V

    n_trunc = sum(1 for reg in registros for m in reg["muestras"] if m.get("truncado"))
    base = BASELINES.get(benchmark)
    P = [
        f"# Fase 1 — Computo en inferencia (TTC) — {benchmark}",
        "",
        f"- Modelo: {mb.MODEL_ID}",
        f"- Problemas: {len(registros)}  |  N_max: {n_max}  |  max_tokens: {max_tokens}",
        f"- Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    if base is not None:
        P.append(f"- Baseline (muestreo unico N=1): {base:.2f}%")
    P += [f"- Muestras truncadas (de todas las generadas): {n_trunc}", "",
          "## Precision vs N por selector", "",
          "| Selector | " + " | ".join(f"N={N}" for N in Ns) + " |",
          "|" + "---|" * (len(Ns) + 1)]
    for metodo in SELECTORES:
        celdas = []
        for N in Ns:
            ok, tot = tabla[metodo][N]
            celdas.append(f"{100.0 * ok / tot:.2f}% ({ok}/{tot})" if tot else "-")
        P.append(f"| {metodo} | " + " | ".join(celdas) + " |")
    P += ["", "## Ejemplos (primeros 3 problemas, para auditar)", ""]
    for reg in registros[:3]:
        respuestas = [m["respuesta"] for m in reg["muestras"]]
        P.append(f"- Problema {reg['i']}: gold=`{reg['correcta']}` | respuestas extraidas: `{respuestas}`")
        for metodo in SELECTORES:
            pred = motor.seleccionar(benchmark, reg["muestras"], metodo)
            estado = "OK" if V.es_correcto(benchmark, pred, reg["correcta"]) else "MAL"
            P.append(f"    - {metodo}: `{pred}` ({estado})")
    ruta = f"/data/report_ttc_{benchmark}{('_' + tag) if tag else ''}.md"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    print(f"[TTC] report escrito en {ruta}", flush=True)


@app.function(gpu="L4", image=image, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 9)
def correr(benchmark: str, n_max: int, num_problemas: int, max_tokens: int, hf_token: str = "", gpu: str = "T4", adaptador: str = "", tag: str = ""):
    import json
    import os
    import sys

    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache"
    os.environ["BENCHMARK"] = benchmark
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token  # solo necesario para GPQA (dataset restringido)
    # La T4 (compute 7.5) no soporta FlashAttention-2; vLLM caeria a FlashInfer, que
    # compila kernels con nvcc (ausente en la imagen). Usamos el backend Triton, que
    # compila por su cuenta sin nvcc. En GPUs >= sm80 (L4/A10G/A100) no hace falta.
    if str(gpu).upper().startswith("T4"):
        os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

    import motor
    import modelo_base as mb
    import run_benchmark as rb
    import verificadores as V

    datos = rb.cargar_datos(benchmark)
    total = len(datos) if num_problemas <= 0 else min(num_problemas, len(datos))
    print(f"[TTC] {benchmark}: problemas={total} n_max={n_max} max_tokens={max_tokens}", flush=True)

    sufijo = f"_{tag}" if tag else ""
    ruta = f"/data/ttc_samples_{benchmark}{sufijo}.jsonl"
    # Reanudar: cargar problemas que ya tienen al menos n_max muestras.
    hechos = {}
    if os.path.exists(ruta):
        for linea in open(ruta, encoding="utf-8"):
            linea = linea.strip()
            if not linea:
                continue
            try:
                d = json.loads(linea)
                if d.get("i") is not None and len(d.get("muestras", [])) >= n_max:
                    hechos[d["i"]] = d
            except json.JSONDecodeError:
                continue
    pend = [i for i in range(total) if i not in hechos]
    print(f"[TTC] ya generados={len(hechos)} pendientes={len(pend)}", flush=True)

    if pend:
        from vllm import SamplingParams
        llm = mb.crear_llm(max_model_len=max_tokens + 4096, enable_lora=bool(adaptador))
        sp = SamplingParams(n=n_max, temperature=mb.TEMPERATURE, top_p=mb.TOP_P,
                            max_tokens=max_tokens, logprobs=1)
        chat_kw = {}
        if adaptador:
            from vllm.lora.request import LoRARequest
            chat_kw["lora_request"] = LoRARequest("nova", 1, adaptador)
        CHUNK_P = 16  # problemas por lote: guardamos + commit tras cada lote (reanudable)
        with open(ruta, "a", encoding="utf-8") as f:
            for s in range(0, len(pend), CHUNK_P):
                lote = pend[s:s + CHUNK_P]
                conversaciones = [mb.mensajes(rb.construir_prompt(benchmark, datos[i]["pregunta"])) for i in lote]
                salidas = llm.chat(conversaciones, sp, **chat_kw)
                for i, out in zip(lote, salidas):
                    muestras = []
                    for o in out.outputs:
                        nt = len(o.token_ids)
                        cert = (o.cumulative_logprob / nt) if (o.cumulative_logprob is not None and nt) else 0.0
                        muestras.append({
                            "respuesta": V.extraer_respuesta(benchmark, o.text),
                            "certeza": cert,
                            "truncado": (o.finish_reason == "length"),
                            "n_tok": nt,
                        })
                    reg = {"i": i, "correcta": datos[i]["correcta"], "muestras": muestras}
                    f.write(json.dumps(reg, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    hechos[i] = reg
                vol.commit()
                print(f"[TTC] guardado: {len(hechos)}/{total} problemas", flush=True)

    # Analisis: precision vs N por selector (subconjuntos de las muestras ya generadas).
    registros = [hechos[i] for i in range(total) if i in hechos]
    Ns = [n for n in SWEEP if n <= n_max]
    tabla = {metodo: {} for metodo in SELECTORES}
    for metodo in SELECTORES:
        for N in Ns:
            ok = sum(
                1 for reg in registros
                if V.es_correcto(benchmark, motor.seleccionar(benchmark, reg["muestras"][:N], metodo), reg["correcta"])
            )
            tabla[metodo][N] = (ok, len(registros))

    _escribir_report(benchmark, tabla, registros, n_max, max_tokens, Ns, tag)
    vol.commit()
    for metodo in SELECTORES:
        resumen = "  ".join(f"N{N}:{tabla[metodo][N][0]}/{tabla[metodo][N][1]}" for N in Ns)
        print(f"[TTC] {benchmark} {metodo}: {resumen}", flush=True)
    return {m: {str(N): tabla[m][N] for N in tabla[m]} for m in tabla}


@app.local_entrypoint()
def main(benchmark: str = "aime", n_max: int = 32, num_problemas: int = 0,
         max_tokens: int = 32768, gpu: str = "L4", hf_token: str = "", adaptador: str = "", tag: str = ""):
    print(f"[TTC] lanzando {benchmark} | n_max={n_max} | problemas={num_problemas or 'todos'} | gpu={gpu} | tag={tag or 'v0'}")
    fn = correr if gpu == "L4" else correr.with_options(gpu=gpu)
    fn.remote(benchmark, n_max, num_problemas, max_tokens, hf_token, gpu, adaptador, tag)
    print("[TTC] enviado. Resultados en el volumen nova-data (report_ttc_*.md).")
