"""Grafica + tabla del Paso 0 (Fase 3): mayoria vs oracle (pass@N) en AIME, por N,
a partir de las muestras N-alto guardadas. A prueba de infinitos."""
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
SWEEP = [8, 16, 32, 64, 128]


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


def mayoria(muestras):
    cuenta = defaultdict(int)
    repro = {}
    for m in muestras:
        r = norm(m["respuesta"])
        if r == "":
            continue
        cuenta[r] += 1
        repro.setdefault(r, m["respuesta"])
    if not cuenta:
        return ""
    return repro[max(cuenta.items(), key=lambda kv: kv[1])[0]]


def main():
    path = f"{RES}/ttc_samples_aime_nhigh.jsonl"
    if not os.path.exists(path):
        print("No estan las muestras todavia:", path)
        return
    byi = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            byi[d["i"]] = d
    idxs = sorted(byi)
    nmax = max(len(byi[i]["muestras"]) for i in idxs)
    Ns = [n for n in SWEEP if n <= nmax]
    n = len(idxs)

    maj, orc = [], []
    print(f"AIME N-alto — {n} problemas (N_max={nmax})")
    print(f"{'N':>4} | {'mayoria':>16} | {'oracle (pass@N)':>16} | hueco")
    for N in Ns:
        m = sum(1 for i in idxs if correcto(mayoria(byi[i]["muestras"][:N]), byi[i]["correcta"]))
        o = sum(1 for i in idxs if any(correcto(x["respuesta"], byi[i]["correcta"]) for x in byi[i]["muestras"][:N]))
        maj.append(100 * m / n)
        orc.append(100 * o / n)
        print(f"{N:>4} | {f'{100*m/n:.1f}% ({m}/{n})':>16} | {f'{100*o/n:.1f}% ({o}/{n})':>16} | +{100*(o-m)/n:.1f} pts")

    os.makedirs(OUT, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(Ns, maj, marker="o", linewidth=2, label="mayoria (produccion)")
    plt.plot(Ns, orc, marker="s", linewidth=2, label="oracle / pass@N (techo)")
    plt.fill_between(Ns, maj, orc, alpha=0.12, label="hueco de seleccion (premio del PRM)")
    plt.xscale("log", base=2)
    plt.xticks(Ns, [str(x) for x in Ns])
    plt.xlabel("N (muestras por problema)")
    plt.ylabel("Precision (%)")
    plt.title(f"AIME 2024 — mayoria vs oracle vs N  ({n} problemas)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = f"{OUT}/nhigh_aime_mayoria_vs_oracle.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("grafica ->", out)


if __name__ == "__main__":
    main()
