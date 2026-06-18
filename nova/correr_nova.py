"""Nova — PUNTO DE ENTRADA UNICO ("usar Nova").

Carga la base CONGELADA + el ultimo adaptador de Nova (si existe; si no, base = Nova-v0)
y corre el motor de la Fase 1: muestrea N soluciones y elige la mejor con el selector.

Como no hay GPU local, corre en Modal:
    modal run nova/correr_nova.py --pregunta "¿por que se mueven los coches?" --n 32
    modal run nova/correr_nova.py --pregunta "..." --n 32 --selector autocerteza --adaptador /data/adapters/nova-v1-sft

`--adaptador` es una ruta DENTRO del volumen nova-data (p. ej. /data/adapters/nova-v1-sft).
Si se omite, se usa solo la base (Nova-v0).
"""
import modal

app = modal.App("nova-correr")
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
def responder(pregunta: str, n: int, selector: str, adaptador: str):
    import os
    import sys

    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache"
    import modelo_base as mb
    import motor
    import verificadores as V
    from vllm import LLM, SamplingParams

    usar_lora = bool(adaptador) and os.path.exists(adaptador)
    comun = dict(model=mb.MODEL_ID, dtype="float16", max_model_len=36864,
                 gpu_memory_utilization=0.92, trust_remote_code=True)
    if usar_lora:
        llm = LLM(enable_lora=True, max_lora_rank=32, **comun)
    else:
        llm = LLM(**comun)

    sp = SamplingParams(n=n, temperature=mb.TEMPERATURE, top_p=mb.TOP_P,
                        max_tokens=32768, logprobs=1)
    chat_kw = {}
    if usar_lora:
        from vllm.lora.request import LoRARequest
        chat_kw["lora_request"] = LoRARequest("nova", 1, adaptador)

    out = llm.chat([mb.mensajes(pregunta)], sp, **chat_kw)[0]
    muestras = []
    for o in out.outputs:
        nt = len(o.token_ids)
        cert = (o.cumulative_logprob / nt) if (o.cumulative_logprob is not None and nt) else 0.0
        # extraccion estilo mates (\\boxed{}); si no hay, queda "" y se elige por certeza
        muestras.append({"respuesta": V.extraer_respuesta("aime", o.text), "certeza": cert, "texto": o.text})

    idx = motor.indice_elegido("aime", muestras, selector)
    return {
        "version": "Nova-v1 (base+SFT)" if usar_lora else "Nova-v0 (base)",
        "n": n, "selector": selector,
        "extraida": muestras[idx]["respuesta"],
        "texto": muestras[idx]["texto"],
    }


@app.local_entrypoint()
def main(pregunta: str, n: int = 32, selector: str = "autocerteza", adaptador: str = "", gpu: str = "L4"):
    fn = responder if gpu == "L4" else responder.with_options(gpu=gpu)
    r = fn.remote(pregunta, n, selector, adaptador)
    print("\n" + "=" * 72)
    print(f"{r['version']}  |  N={r['n']}  |  selector={r['selector']}")
    if r["extraida"]:
        print(f"Respuesta final extraida: {r['extraida']}")
    print("=" * 72)
    print(r["texto"])
