"""EXPERIMENTO DE AUTOVERIFICACION (Fase 1+, sin entrenar, sin tocar la base).

Para cada problema: (1) genera N soluciones con v0; (2) el PROPIO modelo v0 verifica
cada solucion (¿correcta? -> SI/NO -> v_score); (3) selecciona ponderando por v_score.
Compara 'autoverificacion' contra 'mayoria' y 'autocerteza' en AIME (N=8,16,32).
Logging completo en el volumen nova-data.

Uso:  modal run nova/inference/run_autoverif_modal.py --benchmark aime --n-max 32
"""
import modal

app = modal.App("nova-autoverif")
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

SWEEP = [8, 16, 32]
SELECTORES = ["mayoria", "autocerteza", "autoverificacion"]


def _parse_si_no(t):
    import re
    m = re.search(r"\\boxed\{\s*(SI|SÍ|YES|NO)\s*\}", t, re.IGNORECASE)
    if m:
        return "SI" if m.group(1).upper() in ("SI", "SÍ", "YES") else "NO"
    ult = None
    for mm in re.finditer(r"\b(SI|SÍ|YES|NO)\b", t, re.IGNORECASE):
        ult = mm.group(1).upper()
    return ("SI" if ult in ("SI", "SÍ", "YES") else "NO") if ult else "NO"


def _verif_prompt(pregunta, solucion, max_chars):
    sol = solucion if len(solucion) <= max_chars else "...(recortado)...\n" + solucion[-max_chars:]
    return (
        "Eres un verificador riguroso. Juzga si la SOLUCION resuelve correctamente el PROBLEMA.\n\n"
        f"PROBLEMA:\n{pregunta}\n\nSOLUCION PROPUESTA:\n{sol}\n\n"
        "¿El razonamiento es correcto y la respuesta final es valida? Razona muy brevemente y "
        "termina EXACTAMENTE con \\boxed{SI} o \\boxed{NO}."
    )


def _report(benchmark, tabla, total, n_max, aprob, ntot):
    import datetime
    P = [f"# Experimento de autoverificacion — {benchmark} ({total} problemas, N_max={n_max})", "",
         f"Fecha: {datetime.datetime.now():%Y-%m-%d %H:%M}",
         f"Soluciones aprobadas por el autoverificador (SI): {aprob}/{ntot}", "",
         "## Precision por selector y N", "",
         "| Selector | " + " | ".join(f"N={N}" for N in SWEEP) + " |",
         "|" + "---|" * (len(SWEEP) + 1)]
    for m in SELECTORES:
        P.append(f"| {m} | " + " | ".join(f"{100 * tabla[m][N][0] / total:.1f}% ({tabla[m][N][0]}/{total})" for N in SWEEP) + " |")
    with open(f"/data/report_autoverif_{benchmark}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    print(f"[AV] report -> /data/report_autoverif_{benchmark}.md", flush=True)


@app.function(gpu="A100", image=image, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 9)
def correr(benchmark, n_max, num_problemas, max_tokens, max_verif_tokens, max_sol_chars):
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache"
    os.environ["BENCHMARK"] = benchmark
    import modelo_base as mb
    import motor
    import run_benchmark as rb
    import verificadores as V
    from vllm import SamplingParams

    datos = rb.cargar_datos(benchmark)
    total = len(datos) if num_problemas <= 0 else min(num_problemas, len(datos))
    llm = mb.crear_llm(max_model_len=max_tokens + 4096)

    # (1) generar N soluciones por problema (texto completo)
    sp_gen = SamplingParams(n=n_max, temperature=mb.TEMPERATURE, top_p=mb.TOP_P, max_tokens=max_tokens, logprobs=1)
    conv = [mb.mensajes(rb.construir_prompt(benchmark, datos[i]["pregunta"])) for i in range(total)]
    print(f"[AV] generando {total} x {n_max} soluciones...", flush=True)
    gens = llm.chat(conv, sp_gen)

    cands, vprompts, mapa = [], [], []
    for i, g in enumerate(gens):
        lst = []
        for j, o in enumerate(g.outputs):
            nt = len(o.token_ids)
            cert = (o.cumulative_logprob / nt) if (o.cumulative_logprob is not None and nt) else 0.0
            lst.append({"respuesta": V.extraer_respuesta(benchmark, o.text), "certeza": cert})
            vprompts.append(mb.mensajes(_verif_prompt(datos[i]["pregunta"], o.text, max_sol_chars)))
            mapa.append((i, j))
        cands.append(lst)

    # (2) autoverificacion: el modelo juzga cada solucion (veredicto determinista)
    print(f"[AV] verificando {len(vprompts)} soluciones...", flush=True)
    sp_v = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=max_verif_tokens)
    verifs = llm.chat(vprompts, sp_v)
    aprob = 0
    for (i, j), vo in zip(mapa, verifs):
        ver = _parse_si_no(vo.outputs[0].text)
        cands[i][j]["v_score"] = 1.0 if ver == "SI" else 0.0
        cands[i][j]["veredicto"] = ver
        aprob += (ver == "SI")
    print(f"[AV] aprobadas (SI): {aprob}/{len(mapa)}", flush=True)

    # (3) logging completo
    ruta = f"/data/ttc_samples_{benchmark}_autoverif.jsonl"
    with open(ruta, "w", encoding="utf-8") as f:
        for i in range(total):
            f.write(json.dumps({"i": i, "correcta": datos[i]["correcta"], "muestras": cands[i]}, ensure_ascii=False) + "\n")
    vol.commit()

    # (4) analisis: 3 selectores x N
    tabla = {m: {} for m in SELECTORES}
    for m in SELECTORES:
        for N in SWEEP:
            ok = sum(1 for i in range(total)
                     if V.es_correcto(benchmark, motor.seleccionar(benchmark, cands[i][:N], m), datos[i]["correcta"]))
            tabla[m][N] = (ok, total)

    _report(benchmark, tabla, total, n_max, aprob, len(mapa))
    vol.commit()
    for m in SELECTORES:
        print(f"[AV] {benchmark} {m}: " + "  ".join(f"N{N}:{tabla[m][N][0]}/{total}" for N in SWEEP), flush=True)
    return {m: {str(N): tabla[m][N] for N in SWEEP} for m in tabla}


@app.local_entrypoint()
def main(benchmark: str = "aime", n_max: int = 32, num_problemas: int = 0,
         max_tokens: int = 32768, max_verif_tokens: int = 1024, max_sol_chars: int = 6000, gpu: str = "A100"):
    fn = correr if gpu == "A100" else correr.with_options(gpu=gpu)
    print(fn.remote(benchmark, n_max, num_problemas, max_tokens, max_verif_tokens, max_sol_chars))
