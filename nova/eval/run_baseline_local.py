"""
run_baseline_local.py — Baseline pass@1 a N=1 EN LOCAL (RTX 3060) con vLLM.

Fase RL (GRPO/RLVR). Re-medimos la referencia en ESTE entorno: solo se puede
comparar contra números del mismo arnés y entorno.

Uso (una semilla por ejecución; lanzar 3-5 semillas y agregar):
    python nova/eval/run_baseline_local.py --seed 101
    python nova/eval/run_baseline_local.py --seed 101 --dataset nova/data/aime_eval_90.json

Qué hace:
  1. Carga el dataset local (json o jsonl con {idx, problema, gold}).
  2. Genera UNA respuesta por problema con vLLM (enforce_eager=True, validado
     en esta máquina), config oficial DeepSeek: sin system prompt, temp 0.6,
     top_p 0.95. max_tokens generoso (16K): en evaluación SÍ se deja razonar.
  3. Corrige con los extractores del arnés (run_benchmark.py).
  4. REANUDABLE: jsonl por pregunta con flush+fsync; si existe, continúa.
  5. Semilla determinista POR PROBLEMA (seed*100000+idx): reanudar una tirada
     a medias produce las mismas generaciones que una tirada entera.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import extraer_num, extraer_letra, comparar_num  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEMPERATURE = 0.6   # config oficial DeepSeek-R1
TOP_P = 0.95


def parsear_args():
    p = argparse.ArgumentParser(description="Baseline N=1 local con vLLM")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset", default="nova/data/aime_eval_90.json",
                   help="ruta (relativa al repo o absoluta) a json/jsonl con {idx, problema, gold}")
    p.add_argument("--tipo", choices=["num", "letra"], default="num",
                   help="extracción: num (AIME/GSM8K) o letra (GPQA)")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--gpu-mem", type=float, default=0.85,
                   help="gpu_memory_utilization de vLLM (el escritorio de Windows come ~0.85 GB)")
    p.add_argument("--lote", type=int, default=12,
                   help="problemas por llamada a generate (flush incremental entre lotes)")
    p.add_argument("--n", type=int, default=0, help="limitar nº de problemas (0 = todos)")
    p.add_argument("--out-dir", default="results/rl_local")
    p.add_argument("--modelo", default=MODEL_ID,
                   help="modelo o ruta local (para evaluar checkpoints fusionados más adelante)")
    p.add_argument("--lora", default="",
                   help="ruta a un adaptador LoRA (se aplica sobre --modelo)")
    p.add_argument("--etiqueta", default="",
                   help="sufijo para el archivo de resultados (p.ej. 'piloto'); vacío = baseline")
    return p.parse_args()


def resolver(ruta):
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else REPO_ROOT / ruta


def cargar_dataset(ruta):
    """json (lista) o jsonl con {idx, problema, gold}. idx se genera si falta."""
    ruta = resolver(ruta)
    if ruta.suffix == ".jsonl":
        filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        filas = json.loads(ruta.read_text(encoding="utf-8"))
    datos = []
    for i, d in enumerate(filas):
        datos.append({
            "idx": d.get("idx", i),
            "problema": d["problema"],
            "gold": str(d["gold"]).strip(),
        })
    return datos


def construir_prompt(tipo, problema):
    """Sin system prompt (recomendación oficial DeepSeek-R1)."""
    if tipo == "letra":
        return (problema + "\n\nReason step by step, then put the letter of the "
                "correct option within \\boxed{}.")
    return (problema + "\n\nPlease reason step by step, and put your final "
            "answer within \\boxed{}.")


def corregir(tipo, texto, gold):
    if tipo == "letra":
        pred = extraer_letra(texto)
        return pred, (pred != "" and pred == gold)
    pred = extraer_num(texto)
    return pred, comparar_num(pred, gold)


def leer_hechos(ruta):
    """Índices ya evaluados (ignora líneas corruptas; si hay repetidos gana la última)."""
    hechos = {}
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    d = json.loads(linea)
                    if "idx" in d and "acierto" in d:
                        hechos[d["idx"]] = d
                except json.JSONDecodeError:
                    continue
    return hechos


def main():
    args = parsear_args()
    datos = cargar_dataset(args.dataset)
    if args.n:
        datos = datos[:args.n]
    nombre_ds = Path(args.dataset).stem
    sufijo = f"_{args.etiqueta}" if args.etiqueta else ""
    out_dir = resolver(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_res = out_dir / f"baseline_n1_{nombre_ds}{sufijo}_seed{args.seed}.jsonl"

    hechos = leer_hechos(ruta_res)
    pendientes = [d for d in datos if d["idx"] not in hechos]
    print("=" * 78, flush=True)
    print(f"Baseline N=1 | dataset={nombre_ds} ({len(datos)}) | seed={args.seed} | "
          f"tipo={args.tipo} | max_tokens={args.max_tokens}", flush=True)
    print(f"Resultados: {ruta_res} | ya hechos: {len(hechos)} | pendientes: {len(pendientes)}",
          flush=True)
    print("=" * 78, flush=True)

    if pendientes:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            LoRARequest = None

        tokenizer = AutoTokenizer.from_pretrained(args.modelo)
        t0 = time.time()
        kwargs_lora = {}
        if args.lora:
            kwargs_lora = {"enable_lora": True, "max_lora_rank": 32}
        llm = LLM(
            model=args.modelo,
            dtype="bfloat16",
            enforce_eager=True,          # único modo validado en esta máquina
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            seed=args.seed,
            **kwargs_lora,
        )
        lora_req = None
        if args.lora:
            lora_req = LoRARequest("adaptador", 1, str(resolver(args.lora)))
        print(f"vLLM cargado en {time.time() - t0:.0f}s.", flush=True)

        aciertos = sum(1 for d in hechos.values() if d["acierto"])
        n_hechos = len(hechos)
        with open(ruta_res, "a", encoding="utf-8") as f:
            for ini in range(0, len(pendientes), args.lote):
                lote = pendientes[ini:ini + args.lote]
                prompts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": construir_prompt(args.tipo, d["problema"])}],
                    tokenize=False, add_generation_prompt=True) for d in lote]
                # semilla determinista por problema: reanudable sin cambiar generaciones
                sampling = [SamplingParams(
                    temperature=TEMPERATURE, top_p=TOP_P, max_tokens=args.max_tokens,
                    seed=args.seed * 100000 + d["idx"]) for d in lote]
                t_lote = time.time()
                salidas = llm.generate(prompts, sampling, lora_request=lora_req) \
                    if lora_req else llm.generate(prompts, sampling)
                for d, salida in zip(lote, salidas):
                    gen = salida.outputs[0]
                    texto = gen.text
                    truncado = gen.finish_reason == "length"
                    pred, ok = corregir(args.tipo, texto, d["gold"])
                    if truncado:
                        ok = False  # truncado cuenta como fallo (disciplina N=1)
                    registro = {
                        "idx": d["idx"], "seed": args.seed,
                        "pregunta": d["problema"], "respuesta": texto,
                        "prediccion": pred, "correcta": d["gold"],
                        "acierto": ok, "truncado": truncado,
                        "n_tokens": len(gen.token_ids),
                    }
                    f.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    aciertos += int(ok)
                    n_hechos += 1
                    print(f"[{n_hechos}/{len(datos)}] {'OK ' if ok else 'MAL'} "
                          f"idx={d['idx']} pred={pred!r} gold={d['gold']!r} "
                          f"tokens={len(gen.token_ids)}{' TRUNCADO' if truncado else ''} "
                          f"| acc acumulada: {100.0 * aciertos / n_hechos:.1f}%", flush=True)
                print(f"  lote de {len(lote)} en {time.time() - t_lote:.0f}s", flush=True)

    # ---- resumen final ----
    hechos = leer_hechos(ruta_res)
    resultados = [hechos[d["idx"]] for d in datos if d["idx"] in hechos]
    n = len(resultados)
    n_ok = sum(1 for r in resultados if r["acierto"])
    n_trunc = sum(1 for r in resultados if r.get("truncado"))
    acc = 100.0 * n_ok / n if n else 0.0
    resumen = {
        "dataset": nombre_ds, "seed": args.seed, "modelo": args.modelo,
        "lora": args.lora or None, "n": n, "aciertos": n_ok, "acc": round(acc, 2),
        "truncados": n_trunc, "max_tokens": args.max_tokens,
        "temperatura": TEMPERATURE, "top_p": TOP_P,
        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    ruta_resumen = out_dir / f"resumen_{nombre_ds}{sufijo}.jsonl"
    with open(ruta_resumen, "a", encoding="utf-8") as f:
        f.write(json.dumps(resumen, ensure_ascii=False) + "\n")
    print("\n" + "=" * 78, flush=True)
    print(f"RESULTADO {nombre_ds} seed={args.seed}: {n_ok}/{n} = {acc:.2f}% "
          f"| truncados: {n_trunc}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
