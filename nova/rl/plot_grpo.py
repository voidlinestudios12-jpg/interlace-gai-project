"""
plot_grpo.py — Gráficas de vigilancia del entrenamiento GRPO (piloto/run largo).

Lee el jsonl del LogJsonlCallback y pinta las 4 señales del plan:
reward media, KL, longitud media de completion y % truncados (+ % de grupos
sin señal). Guarda PNG en docs/benchmarks/rl_local/.

Uso: python nova/rl/plot_grpo.py [--log results/rl_local/grpo_train_log.jsonl]
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="results/rl_local/grpo_train_log.jsonl")
    p.add_argument("--out", default="docs/benchmarks/rl_local/grpo_curvas.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filas = []
    with open(REPO_ROOT / args.log, encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                try:
                    d = json.loads(linea)
                    if "reward" in d:
                        filas.append(d)
                except json.JSONDecodeError:
                    continue
    if not filas:
        raise SystemExit("sin filas con reward en el log")
    # si hay pasos repetidos (reanudaciones), gana el último
    por_paso = {d["step"]: d for d in filas}
    filas = [por_paso[s] for s in sorted(por_paso)]
    pasos = [d["step"] for d in filas]

    def serie(clave):
        return [d.get(clave) for d in filas]

    def media_movil(xs, k=10):
        out = []
        for i in range(len(xs)):
            v = [x for x in xs[max(0, i - k + 1):i + 1] if x is not None]
            out.append(sum(v) / len(v) if v else None)
        return out

    fig, ejes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GRPO — señales de vigilancia (plan fase RL)")

    ax = ejes[0][0]
    ax.plot(pasos, serie("reward"), alpha=0.3, color="tab:blue")
    ax.plot(pasos, media_movil(serie("reward")), color="tab:blue")
    ax.set_title("reward media (y móvil-10)")
    ax.set_xlabel("paso")

    ax = ejes[0][1]
    ax.plot(pasos, serie("kl"), color="tab:red")
    ax.set_title("KL a la base")
    ax.set_xlabel("paso")

    ax = ejes[1][0]
    ax.plot(pasos, serie("completions/mean_length"), color="tab:green")
    ax.set_title("longitud media de completion (tokens)")
    ax.set_xlabel("paso")

    ax = ejes[1][1]
    clipped = [None if v is None else 100 * v for v in serie("completions/clipped_ratio")]
    sin_senal = [None if v is None else 100 * v for v in serie("frac_reward_zero_std")]
    ax.plot(pasos, clipped, color="tab:orange", label="% truncados")
    ax.plot(pasos, sin_senal, color="tab:gray", alpha=0.6, label="% grupos sin señal")
    ax.axhline(30, color="tab:orange", ls="--", alpha=0.4)
    ax.set_title("% truncados (umbral 30) y % grupos sin señal")
    ax.set_xlabel("paso")
    ax.legend()

    fig.tight_layout()
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(out)
    # resumen rápido en texto para el reporte
    ult = filas[-1]
    print(f"último paso {ult['step']}: reward={ult.get('reward')} kl={ult.get('kl')} "
          f"len={ult.get('completions/mean_length')} "
          f"trunc={ult.get('completions/clipped_ratio')}")


if __name__ == "__main__":
    main()
