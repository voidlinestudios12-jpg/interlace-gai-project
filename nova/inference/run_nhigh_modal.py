"""FASE 3 — PASO 0: experimento de N alto en AIME (barato, sin entrenar, sin tocar la base).

Para AIME (30 problemas) genera N_max soluciones por problema (temp 0.6, top_p 0.95),
GUARDANDO EL TEXTO COMPLETO de cada solucion (lo necesitaremos para el reranking del
verificador en la Fase 3 — NO para entrenarlo: el verificador se entrena con un conjunto
SEPARADO y descontaminado; AIME es solo evaluacion).

Para cada N en {8,16,32,64,128} reporta DOS curvas:
  - mayoria : precision del voto mayoritario (lo que se usaria en produccion).
  - oracle  : pass@N = acierto si ALGUNA de las N muestras tiene la respuesta correcta
              (el gold se usa SOLO para medir el techo; nunca para elegir en produccion).

Reanudable (salta problemas que ya tienen >= n_max muestras) y con commit por lote.

Uso:  modal run nova/inference/run_nhigh_modal.py --n-max 128
Descargar: modal volume get nova-data ttc_samples_aime_nhigh.jsonl ./resultados/
"""
import modal

app = modal.App("nova-nhigh")
vol = modal.Volume.from_name("nova-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "requests", "pandas")
    .pip_install("sympy")
    .add_local_file("nova/eval/run_benchmark.py", "/root/run_benchmark.py")
    .add_local_file("shared/modelo_base.py", "/root/modelo_base.py")
    .add_local_file("nova/inference/verificadores.py", "/root/verificadores.py")
    .add_local_file("nova/inference/motor.py", "/root/motor.py")
)

BENCHMARK = "aime"
SWEEP = [8, 16, 32, 64, 128]


def _escribir_report(tabla, n_reg, n_max, max_tokens, Ns, n_trunc):
    import datetime
    P = [f"# Fase 3 — Paso 0: N alto en AIME ({n_reg} problemas, N_max={n_max})", "",
         f"Fecha: {datetime.datetime.now():%Y-%m-%d %H:%M}",
         f"Generacion: temp 0.6 / top_p 0.95 / max_tokens {max_tokens}.",
         f"Muestras truncadas (sin EOS al agotar tokens): {n_trunc} (de {n_reg * n_max}).", "",
         "## Mayoria vs Oracle (pass@N) por N", "",
         "| N | mayoria | oracle (pass@N) | hueco de seleccion |",
         "|---|---|---|---|"]
    for N in Ns:
        maj, tot = tabla["mayoria"][N]
        orc, _ = tabla["oracle"][N]
        P.append(f"| {N} | {100 * maj / tot:.1f}% ({maj}/{tot}) | {100 * orc / tot:.1f}% ({orc}/{tot}) | "
                 f"+{100 * (orc - maj) / tot:.1f} pts |")
    P += ["", "- **mayoria**: lo que se elegiria en produccion (voto mayoritario).",
          "- **oracle**: techo de cualquier selector (si alguna muestra acierta). El gold solo mide el techo.",
          "- **hueco**: lo maximo que un verificador perfecto podria rescatar a cada N."]
    with open(f"/data/report_nhigh_{BENCHMARK}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    print(f"[NHIGH] report -> /data/report_nhigh_{BENCHMARK}.md", flush=True)


@app.function(gpu="A100", image=image, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 9)
def correr(n_max, num_problemas, max_tokens, chunk_p):
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache"
    os.environ["BENCHMARK"] = BENCHMARK
    import modelo_base as mb
    import motor
    import run_benchmark as rb
    import verificadores as V
    from vllm import SamplingParams

    datos = rb.cargar_datos(BENCHMARK)
    total = len(datos) if num_problemas <= 0 else min(num_problemas, len(datos))
    ruta = f"/data/ttc_samples_{BENCHMARK}_nhigh.jsonl"

    # --- reanudar: cargar problemas que ya tienen >= n_max muestras ---
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
    print(f"[NHIGH] total={total} ya={len(hechos)} pendientes={len(pend)} n_max={n_max}", flush=True)

    # --- generacion ---
    if pend:
        llm = mb.crear_llm(max_model_len=max_tokens + 4096)
        sp = SamplingParams(n=n_max, temperature=mb.TEMPERATURE, top_p=mb.TOP_P,
                            max_tokens=max_tokens, logprobs=1)
        with open(ruta, "a", encoding="utf-8") as f:
            for s in range(0, len(pend), chunk_p):
                lote = pend[s:s + chunk_p]
                conv = [mb.mensajes(rb.construir_prompt(BENCHMARK, datos[i]["pregunta"])) for i in lote]
                outs = llm.chat(conv, sp)
                for i, out in zip(lote, outs):
                    muestras = []
                    for o in out.outputs:
                        nt = len(o.token_ids)
                        cert = (o.cumulative_logprob / nt) if (o.cumulative_logprob is not None and nt) else 0.0
                        muestras.append({
                            "respuesta": V.extraer_respuesta(BENCHMARK, o.text),
                            "certeza": cert,
                            "truncado": (o.finish_reason == "length"),
                            "texto": o.text,
                        })
                    reg = {"i": i, "correcta": datos[i]["correcta"], "muestras": muestras}
                    f.write(json.dumps(reg, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    hechos[i] = reg
                vol.commit()
                print(f"[NHIGH] guardado {len(hechos)}/{total} problemas", flush=True)

    # --- analisis: mayoria + oracle por N ---
    registros = [hechos[i] for i in range(total) if i in hechos]
    n_reg = len(registros)
    Ns = [n for n in SWEEP if n <= n_max]
    n_trunc = sum(1 for r in registros for m in r["muestras"] if m.get("truncado"))
    tabla = {"mayoria": {}, "oracle": {}}
    for N in Ns:
        maj = sum(1 for r in registros
                  if V.es_correcto(BENCHMARK, motor.seleccionar(BENCHMARK, r["muestras"][:N], "mayoria"), r["correcta"]))
        orc = sum(1 for r in registros
                  if any(V.es_correcto(BENCHMARK, m["respuesta"], r["correcta"]) for m in r["muestras"][:N]))
        tabla["mayoria"][N] = (maj, n_reg)
        tabla["oracle"][N] = (orc, n_reg)

    _escribir_report(tabla, n_reg, n_max, max_tokens, Ns, n_trunc)
    vol.commit()
    for cur in ("mayoria", "oracle"):
        print(f"[NHIGH] {cur}: " + "  ".join(f"N{N}:{tabla[cur][N][0]}/{n_reg}" for N in Ns), flush=True)
    return {cur: {str(N): tabla[cur][N] for N in Ns} for cur in tabla}


@app.local_entrypoint()
def main(n_max: int = 128, num_problemas: int = 0, max_tokens: int = 32768, chunk_p: int = 5, gpu: str = "A100"):
    fn = correr if gpu == "A100" else correr.with_options(gpu=gpu)
    print(fn.remote(n_max, num_problemas, max_tokens, chunk_p))
