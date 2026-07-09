"""
agrega_eval_final.py — PASO 6: agrega, audita y decide.

Lee los crudos de la eval final (RL y base re-medida, mismas cajas 4090),
re-gradea TODO con el comparador del arnés (auditoría), calcula medias ±
desviación por semilla, deltas pareados, anti-regresión, y aplica el criterio
del plan: promoción a Nova-v1 ⟺ AIME sube ≥ +3 pp de media (o mejora en
TODAS las semillas) Y GSM8K/GPQA no caen > 2 pp.

Uso: python nova/eval/agrega_eval_final.py [--dir results/rl_local]
"""

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nova" / "eval"))
from run_benchmark import extraer_num, extraer_letra, comparar_num  # noqa: E402


def carga_y_audita(ruta, tipo="num"):
    """Carga un jsonl de eval y re-gradea cada fila. Devuelve (n, ok, discrepancias)."""
    filas = [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]
    disc = 0
    for f in filas:
        if tipo == "letra":
            pred = extraer_letra(f["respuesta"]) if "respuesta" in f else f["prediccion"]
            rec = int(str(pred).strip().upper() == str(f["correcta"]).strip().upper())
        else:
            pred = extraer_num(f["respuesta"])
            rec = int(bool(comparar_num(pred, f["correcta"])))
        if rec != f["acierto"]:
            disc += 1
    return len(filas), sum(f["acierto"] for f in filas), disc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/rl_local")
    args = p.parse_args()
    d = REPO_ROOT / args.dir

    print("=" * 72)
    print("AIME-90 N=1 — RL vs BASE (misma caja, mismas semillas)")
    print("=" * 72)
    rl, base = {}, {}
    total_disc = 0
    for f in sorted(glob.glob(str(d / "baseline_n1_aime_eval_90_rl_final_nube_seed*.jsonl"))):
        seed = f.rsplit("seed", 1)[1].split(".")[0]
        n, ok, disc = carga_y_audita(f)
        rl[seed] = 100 * ok / n
        total_disc += disc
        print(f"  RL   seed {seed}: {ok}/{n} = {rl[seed]:.2f}%  (disc. auditoría: {disc})")
    for f in sorted(glob.glob(str(d / "baseline_n1_aime_eval_90_base_nube_seed*.jsonl"))):
        seed = f.rsplit("seed", 1)[1].split(".")[0]
        n, ok, disc = carga_y_audita(f)
        base[seed] = 100 * ok / n
        total_disc += disc
        print(f"  BASE seed {seed}: {ok}/{n} = {base[seed]:.2f}%  (disc. auditoría: {disc})")

    comunes = sorted(set(rl) & set(base))
    if not comunes:
        print("Sin pares completos todavía."); return
    m_rl = statistics.mean(rl[s] for s in comunes)
    m_b = statistics.mean(base[s] for s in comunes)
    sd_rl = statistics.stdev(rl[s] for s in comunes) if len(comunes) > 1 else 0
    sd_b = statistics.stdev(base[s] for s in comunes) if len(comunes) > 1 else 0
    deltas = {s: rl[s] - base[s] for s in comunes}
    print(f"\n  MEDIA RL   ({len(comunes)} semillas): {m_rl:.2f}% ± {sd_rl:.2f}")
    print(f"  MEDIA BASE ({len(comunes)} semillas): {m_b:.2f}% ± {sd_b:.2f}")
    print(f"  DELTA medio: {m_rl - m_b:+.2f} pp | por semilla: "
          + " ".join(f"{s}:{deltas[s]:+.1f}" for s in comunes))
    todas_mejoran = all(v > 0 for v in deltas.values())

    print("\n" + "=" * 72)
    print("ANTI-REGRESIÓN (semilla 101, base vs RL, misma caja)")
    print("=" * 72)
    regresion_ok = True
    for ds, tipo in [("gsm8k_eval_200", "num"), ("gpqa_eval_198", "letra")]:
        fb = glob.glob(str(d / f"baseline_n1_{ds}_antireg_base_nube_seed101.jsonl"))
        fr = glob.glob(str(d / f"baseline_n1_{ds}_antireg_rl_nube_seed101.jsonl"))
        if not (fb and fr):
            print(f"  {ds}: PENDIENTE"); regresion_ok = None; continue
        nb, okb, db = carga_y_audita(fb[0], tipo)
        nr, okr, dr = carga_y_audita(fr[0], tipo)
        total_disc += db + dr
        pb, pr = 100 * okb / nb, 100 * okr / nr
        print(f"  {ds}: base {pb:.2f}% ({okb}/{nb}) | rl {pr:.2f}% ({okr}/{nr}) | delta {pr-pb:+.2f} pp (disc: {db}+{dr})")
        if regresion_ok is not None and pr - pb < -2.0:
            regresion_ok = False

    print("\n" + "=" * 72)
    print(f"AUDITORÍA GLOBAL: {total_disc} discrepancias de regrading en total")
    delta = m_rl - m_b
    criterio_aime = delta >= 3.0 or todas_mejoran
    print(f"CRITERIO AIME (≥+3pp o mejora en todas): {'CUMPLE' if criterio_aime else 'NO CUMPLE'} "
          f"(delta {delta:+.2f} pp, todas mejoran: {todas_mejoran})")
    print(f"CRITERIO ANTI-REGRESIÓN (caída ≤2pp): "
          f"{'PENDIENTE' if regresion_ok is None else 'CUMPLE' if regresion_ok else 'NO CUMPLE'}")
    if criterio_aime and regresion_ok:
        print("VEREDICTO: PROMOCIONA a Nova-v1")
    elif regresion_ok is None:
        print("VEREDICTO: incompleto")
    else:
        print("VEREDICTO: NO promociona (documentar y rumbo a LARC)")


if __name__ == "__main__":
    main()
