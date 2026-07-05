"""
agregar_baseline.py — Agrega los resúmenes multi-semilla del baseline N=1 y
escribe la tabla oficial en docs/benchmarks/rl_local/.

Uso: python nova/eval/agregar_baseline.py [--dataset aime_eval_90] [--etiqueta ""]
"""

import argparse
import datetime
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="aime_eval_90")
    p.add_argument("--etiqueta", default="")
    p.add_argument("--res-dir", default="results/rl_local")
    args = p.parse_args()

    sufijo = f"_{args.etiqueta}" if args.etiqueta else ""
    res_dir = REPO_ROOT / args.res_dir
    ruta_resumen = res_dir / f"resumen_{args.dataset}{sufijo}.jsonl"
    resúmenes = {}
    with open(ruta_resumen, encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                d = json.loads(linea)
                if d["n"] > 0:
                    resúmenes[d["seed"]] = d  # si hay repetidos gana el último

    filas = sorted(resúmenes.values(), key=lambda d: d["seed"])
    accs = [d["acc"] for d in filas]
    media = statistics.mean(accs)
    desv = statistics.stdev(accs) if len(accs) > 1 else 0.0

    doc_dir = REPO_ROOT / "docs" / "benchmarks" / "rl_local"
    doc_dir.mkdir(parents=True, exist_ok=True)
    nombre = f"baseline_n1_{args.dataset}{sufijo}.md"
    d0 = filas[0]
    lineas = [
        f"# Baseline pass@1 a N=1 — {args.dataset} (local, RTX 3060)",
        "",
        f"- **Modelo:** {d0['modelo']}" + (f" + LoRA {d0['lora']}" if d0.get("lora") else ""),
        f"- **Arnés:** nova/eval/run_baseline_local.py (vLLM enforce_eager, bf16)",
        f"- **Generación:** temp {d0['temperatura']}, top_p {d0['top_p']}, "
        f"max_tokens {d0['max_tokens']}",
        f"- **Fecha:** {datetime.datetime.now().strftime('%Y-%m-%d')}",
        "",
        "| Semilla | Aciertos | N | pass@1 | Truncados |",
        "|---|---|---|---|---|",
    ]
    for d in filas:
        lineas.append(f"| {d['seed']} | {d['aciertos']} | {d['n']} | "
                      f"{d['acc']:.2f}% | {d['truncados']} |")
    lineas += [
        "",
        f"**Media ± desviación ({len(accs)} semillas): {media:.2f} ± {desv:.2f} pp**",
        "",
        "Referencia oficial de la fase RL: comparar SOLO contra números de este",
        "mismo arnés y entorno. Truncados cuentan como fallo.",
    ]
    (doc_dir / nombre).write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"{doc_dir / nombre}")
    print(f"media={media:.2f} desv={desv:.2f} semillas={[d['seed'] for d in filas]}")


if __name__ == "__main__":
    main()
