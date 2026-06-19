"""FASE 3 — PASO 4: medir el VERIFICADOR entrenado en AIME 2023+2024+2025 (~90 problemas).

Compara, a cada N in {8,16,32,64}, los selectores:
  - mayoria               (voto mayoritario / self-consistency)
  - autocerteza           (voto ponderado por logprob)
  - verificador_prm       (best-of-N por la puntuacion del verificador ENTRENADO)
  - verificador_prm_pesado(voto ponderado por la puntuacion del verificador)
  - oracle / pass@N       (techo: acierta si ALGUNA muestra es correcta; gold solo mide el techo)

Busca el N donde el verificador rinde mas. Criterio: que supere a 'mayoria' por mas que el ruido.

Dos funciones (imagenes separadas para no mezclar vLLM con bitsandbytes):
  paso 'generar' (A100, vLLM): genera N muestras por problema de AIME 2023+2025 (con texto),
     y REUSA las muestras ya generadas de AIME 2024 (ttc_samples_aime_nhigh.jsonl, primeras N).
     Salida: /data/aime_eval_samples.jsonl  (por problema: year, problema, gold, muestras[]).
  paso 'medir' (A100, transformers+peft): carga el verificador (base 4-bit + adaptador), puntua
     cada solucion (prm_score = P(correcta)), y calcula la tabla de selectores x N. Resumible:
     guarda los prm_score en el propio archivo.

Uso:
  modal run nova/inference/run_verif_eval_modal.py --paso generar --n-eval 64
  modal run nova/inference/run_verif_eval_modal.py --paso medir   --n-eval 64
"""
import modal

app = modal.App("nova-verif-eval")
vol = modal.Volume.from_name("nova-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

_arch = (
    lambda img: img
    .add_local_file("nova/eval/run_benchmark.py", "/root/run_benchmark.py")
    .add_local_file("shared/modelo_base.py", "/root/modelo_base.py")
    .add_local_file("nova/inference/verificadores.py", "/root/verificadores.py")
    .add_local_file("nova/inference/motor.py", "/root/motor.py")
)
image_gen = _arch(modal.Image.debian_slim(python_version="3.11").pip_install("vllm", "requests", "pandas", "sympy"))
image_med = _arch(modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers==4.55.0", "peft", "bitsandbytes", "accelerate", "sympy", "requests", "pandas"))

SAMPLES = "/data/aime_eval_samples.jsonl"
NHIGH = "/data/ttc_samples_aime_nhigh.jsonl"
VERIF = "/data/verificador_v1"
SWEEP = [8, 16, 32, 64]
SELECTORES = ["mayoria", "autocerteza", "verificador_prm", "verificador_prm_pesado"]
DS_SERVER = "https://datasets-server.huggingface.co"


def _tokenizar(tok, problema, solucion, max_len):
    """IDENTICO al de sft_verificador.py: prefijo del problema + COLA de la solucion + cierre."""
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


def _cargar_aime_year(year):
    """AIME de un anyo concreto con gold, via datasets-server. 2025 -> math-ai/aime25;
    2023/2024 -> gneubig/aime-1983-2024 filtrado por Year."""
    import run_benchmark as rb
    items = []
    if year == 2025:
        info = rb._descargar(f"{DS_SERVER}/splits?dataset=math-ai/aime25").json()["splits"][0]
        off, filas = 0, []
        while True:
            j = rb._descargar(f"{DS_SERVER}/rows?dataset=math-ai/aime25&config={info['config']}"
                              f"&split={info['split']}&offset={off}&length=100").json()
            nv = [f["row"] for f in j.get("rows", [])]
            filas += nv
            if not nv or len(filas) >= j.get("num_rows_total", len(filas)):
                break
            off += 100
        for f in filas:
            items.append({"problema": str(f.get("problem", "")).strip(),
                          "gold": rb._normalizar_entero(f.get("answer"))})
    else:
        off, filas = 0, []
        while True:
            j = rb._descargar(f"{DS_SERVER}/rows?dataset=gneubig/aime-1983-2024&config=default"
                              f"&split=train&offset={off}&length=100").json()
            nv = [f["row"] for f in j.get("rows", [])]
            filas += nv
            if not nv or len(filas) >= j.get("num_rows_total", len(filas)):
                break
            off += 100
        for f in filas:
            if int(f.get("Year", 0)) == year:
                items.append({"problema": str(f.get("Question", "")).strip(),
                              "gold": rb._normalizar_entero(f.get("Answer"))})
    return items


# =============================== GENERAR ===================================

@app.function(gpu="A100", image=image_gen, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 9)
def generar(n_eval, max_tokens, chunk_p):
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache"
    os.environ["BENCHMARK"] = "aime"
    import modelo_base as mb
    import run_benchmark as rb
    import verificadores as V
    from vllm import SamplingParams

    # 1) construir el conjunto de 90 problemas (key estable = year+indice)
    problemas = []
    # 2024 desde Maxwell-Jia (mismo orden que nhigh)
    aime24 = rb.cargar_aime()
    for i, it in enumerate(aime24):
        problemas.append({"year": 2024, "iy": i, "problema": it["pregunta"], "gold": it["correcta"]})
    for year in (2023, 2025):
        for i, it in enumerate(_cargar_aime_year(year)):
            problemas.append({"year": year, "iy": i, "problema": it["problema"], "gold": it["gold"]})
    for k, p in enumerate(problemas):
        p["i"] = k
    print(f"[GEN] problemas: total={len(problemas)} "
          f"(2024={sum(p['year']==2024 for p in problemas)}, "
          f"2023={sum(p['year']==2023 for p in problemas)}, "
          f"2025={sum(p['year']==2025 for p in problemas)})", flush=True)

    # 2) reusar muestras de AIME 2024 (nhigh): primeras n_eval por problema
    nhigh = {}
    if os.path.exists(NHIGH):
        for linea in open(NHIGH, encoding="utf-8"):
            linea = linea.strip()
            if linea:
                d = json.loads(linea)
                nhigh[d["i"]] = d
        print(f"[GEN] nhigh AIME2024 disponible: {len(nhigh)} problemas", flush=True)

    # 3) cargar lo ya guardado (reanudar)
    hechos = {}
    if os.path.exists(SAMPLES):
        for linea in open(SAMPLES, encoding="utf-8"):
            linea = linea.strip()
            if linea:
                d = json.loads(linea)
                if len(d.get("muestras", [])) >= n_eval:
                    hechos[d["i"]] = d

    # pre-sembrar 2024 desde nhigh (sin generar)
    nuevos_2024 = []
    for p in problemas:
        if p["year"] == 2024 and p["i"] not in hechos and p["iy"] in nhigh:
            ms = nhigh[p["iy"]]["muestras"][:n_eval]
            ms = [{"texto": m.get("texto", ""), "respuesta": m.get("respuesta", ""),
                   "certeza": m.get("certeza", 0.0)} for m in ms]
            reg = {"i": p["i"], "year": 2024, "problema": p["problema"], "gold": p["gold"], "muestras": ms}
            hechos[p["i"]] = reg
            nuevos_2024.append(reg)
    if nuevos_2024:
        with open(SAMPLES, "a", encoding="utf-8") as f:
            for reg in nuevos_2024:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        vol.commit()
        print(f"[GEN] reusadas {len(nuevos_2024)} problemas de AIME2024 desde nhigh", flush=True)

    # 4) generar 2023+2025 (y 2024 si faltara nhigh)
    pend = [p for p in problemas if p["i"] not in hechos]
    print(f"[GEN] pendientes de generar: {len(pend)}", flush=True)
    if pend:
        llm = mb.crear_llm(max_model_len=max_tokens + 4096)
        sp = SamplingParams(n=n_eval, temperature=mb.TEMPERATURE, top_p=mb.TOP_P,
                            max_tokens=max_tokens, logprobs=1)
        with open(SAMPLES, "a", encoding="utf-8") as f:
            for s in range(0, len(pend), chunk_p):
                lote = pend[s:s + chunk_p]
                conv = [mb.mensajes(rb.construir_prompt("aime", p["problema"])) for p in lote]
                outs = llm.chat(conv, sp)
                for p, out in zip(lote, outs):
                    ms = []
                    for o in out.outputs:
                        nt = len(o.token_ids)
                        cert = (o.cumulative_logprob / nt) if (o.cumulative_logprob is not None and nt) else 0.0
                        ms.append({"texto": o.text, "respuesta": V.extraer_respuesta("aime", o.text), "certeza": cert})
                    reg = {"i": p["i"], "year": p["year"], "problema": p["problema"], "gold": p["gold"], "muestras": ms}
                    f.write(json.dumps(reg, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    hechos[p["i"]] = reg
                vol.commit()
                print(f"[GEN] generadas {min(s + chunk_p, len(pend))}/{len(pend)}", flush=True)

    n = len({d["i"] for d in hechos.values()})
    print(f"[GEN] listo: {n} problemas con muestras en {SAMPLES}", flush=True)
    return {"problemas": n}


# =============================== MEDIR =====================================

def _escribir_report(tabla, oracle, por_year, n, mejor_n, ruido):
    import datetime
    P = [f"# Fase 3 — Paso 4: verificador entrenado en AIME 2023+2024+2025 ({n} problemas)", "",
         f"Fecha: {datetime.datetime.now():%Y-%m-%d %H:%M}",
         "Selectores comparados a cada N (best-of-N o voto). El gold solo mide oracle/aciertos.", "",
         "## Precision por selector y N", "",
         "| Selector | " + " | ".join(f"N={N}" for N in SWEEP) + " |",
         "|" + "---|" * (len(SWEEP) + 1)]
    for m in SELECTORES:
        P.append(f"| {m} | " + " | ".join(f"{100 * tabla[m][N] / n:.1f}% ({tabla[m][N]}/{n})" for N in SWEEP) + " |")
    P.append(f"| **oracle (pass@N)** | " + " | ".join(f"{100 * oracle[N] / n:.1f}% ({oracle[N]}/{n})" for N in SWEEP) + " |")
    P += ["", f"- **Mejor N para verificador_prm:** {mejor_n} "
              f"({100 * tabla['verificador_prm'][mejor_n] / n:.1f}%).",
          f"- **Ruido estimado** (~±{ruido:.1f} pts con {n} problemas).",
          "- **Criterio:** el verificador debe superar a 'mayoria' por mas que el ruido.", "",
          "## Por anyo (en el mejor N del verificador)", "",
          "| Anyo | n | mayoria | verificador_prm | oracle |", "|---|---|---|---|---|"]
    for y in sorted(por_year):
        d = por_year[y]
        P.append(f"| {y} | {d['n']} | {100 * d['mayoria'] / d['n']:.1f}% | "
                 f"{100 * d['verificador_prm'] / d['n']:.1f}% | {100 * d['oracle'] / d['n']:.1f}% |")
    with open("/data/report_verif_eval.md", "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    print("[MED] report -> /data/report_verif_eval.md", flush=True)


@app.function(gpu="A100", image=image_med, volumes={"/data": vol, "/cache": hf_cache}, timeout=60 * 60 * 5)
def medir(n_eval, max_len, batch):
    import json
    import math
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache"
    os.environ["BENCHMARK"] = "aime"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import modelo_base as mb
    import motor
    import verificadores as V

    if not os.path.exists(SAMPLES):
        raise RuntimeError(f"No existe {SAMPLES}; ejecuta primero --paso generar")
    if not os.path.exists(VERIF):
        raise RuntimeError(f"No existe {VERIF}; entrena primero el verificador (sft_verificador.py)")
    registros = [json.loads(l) for l in open(SAMPLES, encoding="utf-8") if l.strip()]
    registros = sorted({r["i"]: r for r in registros}.values(), key=lambda r: r["i"])
    print(f"[MED] {len(registros)} problemas cargados", flush=True)

    # --- cargar verificador (base 4-bit + adaptador) ---
    tok = AutoTokenizer.from_pretrained(VERIF)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        mb.MODEL_ID, num_labels=2, quantization_config=bnb, torch_dtype=torch.bfloat16)
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, VERIF)
    model.eval()
    print("[MED] verificador cargado", flush=True)

    # --- puntuar cada solucion: prm_score = P(correcta) ---
    pares = []  # (ri, mi, problema, texto)
    for ri, r in enumerate(registros):
        for mi, m in enumerate(r["muestras"][:n_eval]):
            pares.append((ri, mi, r["problema"], m.get("texto", "")))
    print(f"[MED] puntuando {len(pares)} soluciones...", flush=True)

    @torch.no_grad()
    def puntuar(lote):
        ids = [_tokenizar(tok, p, t, max_len) for (_, _, p, t) in lote]
        maxl = max(len(x) for x in ids)
        pad = tok.pad_token_id
        input_ids = torch.tensor([x + [pad] * (maxl - len(x)) for x in ids])
        attn = torch.tensor([[1] * len(x) + [0] * (maxl - len(x)) for x in ids])
        out = model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device))
        return torch.softmax(out.logits.float(), dim=-1)[:, 1].cpu().tolist()

    for s in range(0, len(pares), batch):
        lote = pares[s:s + batch]
        scores = puntuar(lote)
        for (ri, mi, _, _), sc in zip(lote, scores):
            registros[ri]["muestras"][mi]["prm_score"] = float(sc)
        if (s // batch) % 20 == 0:
            print(f"[MED] {min(s + batch, len(pares))}/{len(pares)}", flush=True)

    # guardar prm_score (resumible / auditable)
    with open(SAMPLES, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()

    # --- analisis: selectores x N + oracle, global y por anyo ---
    n = len(registros)
    tabla = {m: {} for m in SELECTORES}
    oracle = {}
    for N in SWEEP:
        for m in SELECTORES:
            tabla[m][N] = sum(
                1 for r in registros
                if V.es_correcto("aime", motor.seleccionar("aime", r["muestras"][:N], m), r["gold"]))
        oracle[N] = sum(1 for r in registros
                        if any(V.es_correcto("aime", x["respuesta"], r["gold"]) for x in r["muestras"][:N]))

    mejor_n = max(SWEEP, key=lambda N: tabla["verificador_prm"][N])
    por_year = {}
    for y in sorted({r["year"] for r in registros}):
        sub = [r for r in registros if r["year"] == y]
        por_year[y] = {
            "n": len(sub),
            "mayoria": sum(1 for r in sub if V.es_correcto("aime", motor.seleccionar("aime", r["muestras"][:mejor_n], "mayoria"), r["gold"])),
            "verificador_prm": sum(1 for r in sub if V.es_correcto("aime", motor.seleccionar("aime", r["muestras"][:mejor_n], "verificador_prm"), r["gold"])),
            "oracle": sum(1 for r in sub if any(V.es_correcto("aime", x["respuesta"], r["gold"]) for x in r["muestras"][:mejor_n])),
        }
    p_maj = tabla["mayoria"][mejor_n] / n
    ruido = 100 * 1.96 * math.sqrt(max(p_maj * (1 - p_maj), 0.01) / n)

    _escribir_report(tabla, oracle, por_year, n, mejor_n, ruido)
    for m in SELECTORES:
        print(f"[MED] {m}: " + "  ".join(f"N{N}:{tabla[m][N]}/{n}" for N in SWEEP), flush=True)
    print(f"[MED] oracle: " + "  ".join(f"N{N}:{oracle[N]}/{n}" for N in SWEEP), flush=True)
    print(f"[MED] mejor_n(verif)={mejor_n} ruido=±{ruido:.1f}pts", flush=True)
    return {"tabla": {m: {str(N): tabla[m][N] for N in SWEEP} for m in SELECTORES},
            "oracle": {str(N): oracle[N] for N in SWEEP}, "n": n, "mejor_n": mejor_n, "ruido": ruido}


@app.local_entrypoint()
def main(paso: str = "generar", n_eval: int = 64, max_tokens: int = 32768,
         chunk_p: int = 5, max_len: int = 2048, batch: int = 16):
    if paso == "generar":
        print(generar.remote(n_eval, max_tokens, chunk_p))
    elif paso == "medir":
        print(medir.remote(n_eval, max_len, batch))
    else:
        raise SystemExit("paso debe ser 'generar' o 'medir'")
