"""Grafica + tabla del PASO 4 (Fase 3): mayoria vs autocerteza vs verificador_prm
vs verificador_prm_pesado vs oracle, por N, en AIME 2023+2024+2025.

Recalcula en LOCAL (a prueba de infinitos) desde aime_eval_samples.jsonl (que ya trae
prm_score por solucion tras el paso 'medir'). Sirve para auditar que coincide con Modal.
"""
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pathlib
_repo = pathlib.Path(__file__).resolve().parent
RES = str(_repo.parent / "resultados")
OUT = str(_repo / "docs/benchmarks/fase3_prm")
SWEEP = [8, 16, 32, 64]


def norm(r):
    if r is None:
        return ""
    try:
        f = float(r)
        if not math.isfinite(f):
            return str(r).strip()
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError, OverflowError):
        return str(r).strip()


def correcto(p, g):
    if not p:
        return False
    try:
        fp, fg = float(p), float(g)
        return math.isfinite(fp) and math.isfinite(fg) and abs(fp - fg) < 1e-6
    except (TypeError, ValueError, OverflowError):
        return False


def _voto(muestras, peso):
    pesos = defaultdict(float)
    repro = {}
    for m in muestras:
        r = norm(m.get("respuesta"))
        if r == "":
            continue
        pesos[r] += peso(m)
        repro.setdefault(r, m.get("respuesta"))
    if not pesos or max(pesos.values()) <= 0:
        # fallback a mayoria simple
        return _voto(muestras, lambda m: 1.0) if peso(muestras[0]) != 1.0 else ""
    return repro[max(pesos.items(), key=lambda kv: kv[1])[0]]


def seleccionar(muestras, metodo):
    if not muestras:
        return ""
    if metodo == "mayoria":
        return _voto(muestras, lambda m: 1.0)
    if metodo == "autocerteza":
        return _voto(muestras, lambda m: math.exp(m.get("certeza", 0.0)))
    if metodo == "verificador_prm_pesado":
        return _voto(muestras, lambda m: float(m.get("prm_score", 0.0)))
    if metodo == "verificador_prm":
        validas = [m for m in muestras if norm(m.get("respuesta")) != ""]
        if not validas:
            return ""
        return max(validas, key=lambda m: m.get("prm_score", float("-inf"))).get("respuesta")
    raise ValueError(metodo)


SELECTORES = ["mayoria", "autocerteza", "verificador_prm", "verificador_prm_pesado"]


def main():
    path = f"{RES}/aime_eval_samples.jsonl"
    if not os.path.exists(path):
        print("No estan las muestras todavia:", path)
        return
    regs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    regs = list({r["i"]: r for r in regs}.values())
    n = len(regs)
    tiene_prm = any("prm_score" in m for r in regs for m in r["muestras"])
    print(f"AIME eval — {n} problemas | prm_score presente: {tiene_prm}")

    tabla = {m: [] for m in SELECTORES}
    orc = []
    print(f"{'N':>4} | " + " | ".join(f"{m:>22}" for m in SELECTORES) + " | oracle")
    for N in SWEEP:
        fila = {}
        for m in SELECTORES:
            ok = sum(1 for r in regs if correcto(seleccionar(r["muestras"][:N], m), r["gold"]))
            tabla[m].append(100 * ok / n)
            fila[m] = ok
        o = sum(1 for r in regs if any(correcto(x.get("respuesta"), r["gold"]) for x in r["muestras"][:N]))
        orc.append(100 * o / n)
        print(f"{N:>4} | " + " | ".join(f"{f'{fila[m]}/{n} ({100*fila[m]/n:.1f}%)':>22}" for m in SELECTORES) + f" | {o}/{n} ({100*o/n:.1f}%)")

    os.makedirs(OUT, exist_ok=True)
    plt.figure(figsize=(8, 5))
    estilos = {"mayoria": ("o", "-"), "autocerteza": ("^", "--"),
               "verificador_prm": ("s", "-"), "verificador_prm_pesado": ("D", "--")}
    for m in SELECTORES:
        mk, ls = estilos[m]
        plt.plot(SWEEP, tabla[m], marker=mk, linestyle=ls, linewidth=2, label=m)
    plt.plot(SWEEP, orc, marker="*", linewidth=2, color="black", label="oracle (techo)")
    plt.xscale("log", base=2)
    plt.xticks(SWEEP, [str(x) for x in SWEEP])
    plt.xlabel("N (muestras por problema)")
    plt.ylabel("Precision (%)")
    plt.title(f"AIME 2023+2024+2025 — selectores vs N ({n} problemas)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = f"{OUT}/verif_eval_aime.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("grafica ->", out)


if __name__ == "__main__":
    main()
