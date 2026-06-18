"""DIAGNOSTICO del SFT fallido (Nova-v1). Solo para aprender. No entrena nada.

(a) Cuenta cuantas trazas de entrenamiento superaban max_seq (se truncaron, perdiendo
    su \\boxed{} final -> el modelo aprendio a NO concluir).
(b) Comprueba si la plantilla de chat añade un <think> sola (hipotesis del doble <think>).
(c) Genera 1 respuesta de AIME y 1 de GSM8K con el adaptador v1 (texto completo).
(d) Archiva el adaptador v1 (NO se usa; Nova = v0).
"""
import modal

app = modal.App("nova-diag")
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


@app.function(gpu="L4", image=image, volumes={"/data": vol, "/cache": hf_cache}, timeout=3600)
def diag(adaptador, max_seq_entren):
    import json
    import os
    import sys
    os.environ["HF_HOME"] = "/cache"
    sys.path.insert(0, "/root")
    import modelo_base as mb
    import run_benchmark as rb
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mb.MODEL_ID)

    # (a) trazas de entrenamiento truncadas
    total = trunc = sin_boxed = 0
    p = "/data/sft/light_r1_n1000.jsonl"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total += 1
            full = tok.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=False)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(ids) > max_seq_entren:
                trunc += 1
                if "\\boxed{" not in tok.decode(ids[:max_seq_entren]):
                    sin_boxed += 1
    print(f"[DIAG] trazas={total} truncadas(>{max_seq_entren}tok)={trunc} de_ellas_sin_boxed={sin_boxed}", flush=True)

    # (b) ¿la plantilla añade <think> sola?
    aime_q = rb.construir_prompt("aime", rb.cargar_datos("aime")[0]["pregunta"])
    prompt_tail = tok.apply_chat_template(mb.mensajes(aime_q), tokenize=False, add_generation_prompt=True)[-140:]
    print(f"[DIAG] PROMPT_TAIL={prompt_tail!r}", flush=True)

    # (c) generar 1 AIME + 1 GSM8K con el adaptador v1
    from vllm import LLM, SamplingParams  # noqa: F401
    from vllm.lora.request import LoRARequest
    llm = mb.crear_llm(max_model_len=36864, enable_lora=True)
    lreq = LoRARequest("nova", 1, adaptador)
    gsm_q = rb.construir_prompt("gsm8k", rb.cargar_datos("gsm8k")[0]["pregunta"])
    res = {}
    for nom, q, mt in [("aime", aime_q, 32768), ("gsm8k", gsm_q, 8192)]:
        sp = SamplingParams(n=1, temperature=mb.TEMPERATURE, top_p=mb.TOP_P, max_tokens=mt)
        o = llm.chat([mb.mensajes(q)], sp, lora_request=lreq)[0].outputs[0]
        t = o.text
        res[nom] = {"texto": t, "chars": len(t), "tokens": len(o.token_ids),
                    "finish": o.finish_reason, "boxed": "\\boxed{" in t,
                    "think_open": t.count("<think>"), "think_close": t.count("</think>")}
        print(f"[DIAG] {nom}: tokens={res[nom]['tokens']} finish={o.finish_reason} "
              f"boxed={res[nom]['boxed']} <think>={res[nom]['think_open']} </think>={res[nom]['think_close']}", flush=True)

    # (d) archivar el adaptador v1
    arch = "/data/adapters/ARCHIVADO_nova-v1-sft-FALLIDO"
    if os.path.exists(adaptador) and not os.path.exists(arch):
        os.rename(adaptador, arch)
        vol.commit()
        print(f"[DIAG] adaptador archivado -> {arch}", flush=True)

    res["trunc"] = {"total": total, "truncadas": trunc, "sin_boxed": sin_boxed}
    res["prompt_tail"] = prompt_tail
    return res


@app.local_entrypoint()
def main(adaptador: str = "/data/adapters/nova-v1-sft", gpu: str = "L4", max_seq_entren: int = 8192):
    import os
    fn = diag if gpu == "L4" else diag.with_options(gpu=gpu)
    r = fn.remote(adaptador, max_seq_entren)
    d = "C:/Users/aleja/Desktop/_bench_work/repo/docs/benchmarks/leccion_fase2"
    os.makedirs(d, exist_ok=True)
    for nom in ["aime", "gsm8k"]:
        with open(f"{d}/v1_{nom}_respuesta.txt", "w", encoding="utf-8") as f:
            f.write(r[nom]["texto"])
    print("\n===== DIAGNOSTICO Nova-v1 =====")
    print("Trazas entren. truncadas:", r["trunc"])
    print("Prompt tail:", repr(r["prompt_tail"]))
    for nom in ["aime", "gsm8k"]:
        x = r[nom]
        print(f"\n--- {nom.upper()} v1 | tokens={x['tokens']} finish={x['finish']} "
              f"boxed={x['boxed']} <think>={x['think_open']} </think>={x['think_close']} ---")
        print(x["texto"][:2800])
        if x["chars"] > 4000:
            print("\n...[recortado; completo en docs/benchmarks/leccion_fase2/]...\n")
            print(x["texto"][-1000:])
