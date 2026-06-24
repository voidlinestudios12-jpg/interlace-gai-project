"""VAST.AI — RFT PASO 1/4: medir pass@1 a N=1 (modelo PURO, una muestra/problema).

Mide el MODELO PURO (sin best-of-N) generando K muestras por problema y tratando
cada muestra como una "semilla" de N=1: pass@1 = media de acierto sobre las K
semillas (± desviación). También reporta el pass@1 esperado (media sobre todas
las muestras).

Sirve para el baseline (modelo base) y para el modelo RFT (adaptador LoRA), y para
el control anti-regresión en GSM8K / GPQA.

Uso (por entorno):
  MODELO=base|rft   (rft = base + adaptador LoRA de HF nova-rft-v1)
  BENCHES=aime:16,gsm8k:4,gpqa:4   (bench:Ksemillas)
Escribe eval_{MODELO}_{bench}.jsonl (reanudable) y lo sube a HF. Variables: HF_TOKEN.
"""
import os, sys, json
from collections import defaultdict

HF_REPO_DATA = "Quantumadvancedai/nova-verif-data"
HF_REPO_RFT = "Quantumadvancedai/nova-rft-v1"
TOKEN = os.environ.get("HF_TOKEN")
WORK = "/workspace"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
BASE_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEMPERATURE, TOP_P = 0.6, 0.95

MODELO = os.environ.get("MODELO", "base").strip()
BENCHES_ENV = os.environ.get("BENCHES", "aime:16,gsm8k:4,gpqa:4")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "nova", "eval"))
import run_benchmark as rb  # extraer_num, extraer_letra, comparar_num, cargar_gsm8k, cargar_gpqa

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api = HfApi(token=TOKEN)

MAX_TOKENS = {"aime": 12288, "gsm8k": 3072, "gpqa": 8192}
GSM8K_SUBSET = int(os.environ.get("GSM8K_N", "200"))


def construir_prompt(bench, pregunta):
    if bench == "gpqa":
        return pregunta + "\n\nReason step by step, then put the letter of the correct option within \\boxed{}."
    return pregunta + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def cargar_bench(bench):
    """Devuelve lista [{idx, pregunta, gold}]."""
    if bench == "aime":
        path = os.path.join(REPO_ROOT, "nova", "data", "aime_eval_90.json")
        data = json.load(open(path, encoding="utf-8"))
        return [{"idx": p["idx"], "pregunta": p["problema"], "gold": str(p["gold"])} for p in data]
    if bench == "gsm8k":
        items = rb.cargar_gsm8k()[:GSM8K_SUBSET]
        return [{"idx": i, "pregunta": it["pregunta"], "gold": it["correcta"]} for i, it in enumerate(items)]
    if bench == "gpqa":
        items = rb.cargar_gpqa()  # requiere token con acceso
        return [{"idx": i, "pregunta": it["pregunta"], "gold": it["correcta"]} for i, it in enumerate(items)]
    raise ValueError(bench)


def corregir(bench, texto, gold):
    if bench == "gpqa":
        pred = rb.extraer_letra(texto)
        return pred, (pred != "" and pred == gold)
    pred = rb.extraer_num(texto)
    return pred, rb.comparar_num(pred, gold)


# Parsear benches solicitados
solicitados = []
for parte in BENCHES_ENV.split(","):
    parte = parte.strip()
    if not parte:
        continue
    if ":" in parte:
        b, k = parte.split(":")
        solicitados.append((b.strip(), int(k)))
    else:
        solicitados.append((parte, 8))

print(f"[EVAL] MODELO={MODELO} BENCHES={solicitados}", flush=True)

# Cargar datos (antes de la GPU, por si GPQA no tiene acceso y hay que avisar)
datos_bench = {}
for b, k in solicitados:
    try:
        datos_bench[b] = cargar_bench(b)
        print(f"[EVAL] {b}: {len(datos_bench[b])} problemas (K={k})", flush=True)
    except SystemExit:
        print(f"[EVAL] AVISO: {b} no accesible (token sin permiso); se OMITE", flush=True)
    except Exception as e:
        print(f"[EVAL] AVISO: no se pudo cargar {b}: {repr(e)[:140]}; se OMITE", flush=True)

if not datos_bench:
    print("[EVAL] no hay benches cargables; saliendo", flush=True)
    sys.exit(0)

# --------------------------------------------------------------------------
# Cargar vLLM (con o sin LoRA)
# --------------------------------------------------------------------------
from vllm import LLM, SamplingParams
lora_req = None
if MODELO == "rft":
    adapter_dir = snapshot_download(HF_REPO_RFT, repo_type="model", token=TOKEN,
                                    local_dir=os.path.join(WORK, "nova_rft_v1"))
    from vllm.lora.request import LoRARequest
    lora_req = LoRARequest("rft", 1, adapter_dir)
    llm = LLM(model=BASE_ID, dtype="float16", max_model_len=14336,
              gpu_memory_utilization=0.90, trust_remote_code=True,
              enable_lora=True, max_lora_rank=32)
    print(f"[EVAL] modelo RFT (LoRA desde {adapter_dir})", flush=True)
else:
    llm = LLM(model=BASE_ID, dtype="float16", max_model_len=14336,
              gpu_memory_utilization=0.90, trust_remote_code=True)
    print(f"[EVAL] modelo BASE", flush=True)


def generar(bench, k):
    datos = datos_bench[bench]
    out_path = os.path.join(WORK, f"eval_{MODELO}_{bench}.jsonl")
    # Reanudar: contar (idx,seed) ya hechos
    hechos = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    r = json.loads(l)
                    hechos.add((r["idx"], r["seed"]))
                except Exception:
                    pass
    sp = SamplingParams(n=k, temperature=TEMPERATURE, top_p=TOP_P,
                        max_tokens=MAX_TOKENS.get(bench, 8192), seed=None)
    # Generar por lotes de problemas
    CHUNK = 8 if bench == "aime" else 24
    pend = [d for d in datos if any((d["idx"], s) not in hechos for s in range(k))]
    if not pend:
        print(f"[EVAL] {bench}: ya completo", flush=True)
        return out_path
    with open(out_path, "a", encoding="utf-8") as f:
        for s in range(0, len(pend), CHUNK):
            lote = pend[s:s + CHUNK]
            convs = [[{"role": "user", "content": construir_prompt(bench, d["pregunta"])}] for d in lote]
            kw = {"lora_request": lora_req} if lora_req else {}
            outs = llm.chat(convs, sp, **kw)
            for d, out in zip(lote, outs):
                for seed, o in enumerate(out.outputs):
                    pred, ok = corregir(bench, o.text, d["gold"])
                    f.write(json.dumps({
                        "bench": bench, "modelo": MODELO, "idx": d["idx"], "seed": seed,
                        "gold": d["gold"], "pred": pred, "correcto": bool(ok),
                        "truncado": (o.finish_reason == "length"),
                    }, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
            print(f"[EVAL] {bench} {MODELO}: lote {s//CHUNK+1} ({s+len(lote)}/{len(pend)} probs)", flush=True)
    return out_path


import statistics
for b, k in solicitados:
    if b not in datos_bench:
        continue
    path = generar(b, k)
    # Resumen pass@1
    por_idx = defaultdict(dict)
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if l:
            r = json.loads(l)
            por_idx[r["idx"]][r["seed"]] = r["correcto"]
    P = len(por_idx)
    # pass@1 por semilla
    accs = []
    for seed in range(k):
        c = sum(1 for idx in por_idx if por_idx[idx].get(seed))
        accs.append(100 * c / P)
    todas = [v for d in por_idx.values() for v in d.values()]
    esperado = 100 * sum(todas) / len(todas) if todas else 0
    media = statistics.mean(accs) if accs else 0
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f"[EVAL] === {b} {MODELO}: pass@1 = {media:.2f}% ± {sd:.2f}pp "
          f"(esperado {esperado:.2f}% sobre {len(todas)} muestras, {P} probs) ===", flush=True)
    # subir
    try:
        api.upload_file(path_or_fileobj=path, path_in_repo=f"eval_{MODELO}_{b}.jsonl",
                        repo_id=HF_REPO_DATA, repo_type="dataset",
                        commit_message=f"eval N=1 {MODELO} {b}")
    except Exception as e:
        print(f"[EVAL] aviso subida {b}: {repr(e)[:120]}", flush=True)

print(f"[EVAL] COMPLETADO modelo={MODELO}", flush=True)
