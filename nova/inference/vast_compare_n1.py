"""VAST.AI — RFT PASO 4 (final): comparar BASE vs RFT a N=1 y dar veredicto.

Lee las muestras de evaluación (eval_base_*.jsonl, eval_rft_*.jsonl; para AIME del
base reutiliza aime_gen.jsonl) y calcula, por benchmark, el pass@1 a N=1
(media ± desviación sobre semillas, y el esperado sobre todas las muestras).

Veredicto:
  - AIME: ¿el RFT sube respecto al base por más que el ruido?
  - GSM8K / GPQA (anti-regresión): ¿NO bajan?
Escribe docs/benchmarks/fase_rft/report_rft_n1.md (+ .json) y sube a HF.
Variables: HF_TOKEN.
"""
import os, sys, json, statistics
from collections import defaultdict

HF_REPO_DATA = "Quantumadvancedai/nova-verif-data"
TOKEN = os.environ.get("HF_TOKEN")
WORK = "/workspace"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=TOKEN)

BENCHES = ["aime", "gsm8k", "gpqa"]


def _baja(fn):
    p = os.path.join(WORK, fn)
    if os.path.exists(p):
        return p
    try:
        return hf_hub_download(HF_REPO_DATA, fn, repo_type="dataset", token=TOKEN, local_dir=WORK)
    except Exception:
        return None


def cargar_muestras(modelo, bench):
    """Devuelve dict idx -> {seed: correcto}. Para base+aime reutiliza aime_gen.jsonl."""
    fn = f"eval_{modelo}_{bench}.jsonl"
    p = _baja(fn)
    por_idx = defaultdict(dict)
    if p and os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            por_idx[r["idx"]][r["seed"]] = bool(r["correcto"])
        return por_idx
    # Fallback base+aime: aime_gen.jsonl (campo correcto, seed=sol_idx)
    if modelo == "base" and bench == "aime":
        p = _baja("aime_gen.jsonl")
        if p and os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                l = l.strip()
                if not l:
                    continue
                r = json.loads(l)
                seed = r.get("sol_idx", len(por_idx[r["idx"]]))
                por_idx[r["idx"]][seed] = bool(r.get("correcto"))
            return por_idx
    return None


def metricas(por_idx):
    """(media%, sd, esperado%, P, K) del pass@1 a N=1."""
    if not por_idx:
        return None
    P = len(por_idx)
    K = min(len(d) for d in por_idx.values())
    accs = []
    for seed in range(K):
        c = sum(1 for idx in por_idx if por_idx[idx].get(seed))
        accs.append(100 * c / P)
    todas = [v for d in por_idx.values() for v in d.values()]
    esperado = 100 * sum(todas) / len(todas)
    media = statistics.mean(accs)
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    return {"media": media, "sd": sd, "esperado": esperado, "P": P, "K": K}


resultados = {}
for b in BENCHES:
    base = metricas(cargar_muestras("base", b))
    rft = metricas(cargar_muestras("rft", b))
    resultados[b] = {"base": base, "rft": rft}

# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------
lines = ["# Nova — RFT: modelo PURO a N=1 (pass@1)\n",
         "**Métrica oficial:** pass@1 a N=1 (una muestra/problema), media±desv. sobre semillas.",
         "El verificador ORM se usó SOLO para construir el dataset dorado, no en inferencia.\n",
         "| Benchmark | BASE (N=1) | RFT (N=1) | Δ | esperado base→rft |",
         "|---|---|---|---|---|"]


def fmt(m):
    if not m:
        return "—"
    return f"{m['media']:.1f}% ± {m['sd']:.1f} (P={m['P']},K={m['K']})"


veredicto = {}
for b in BENCHES:
    base, rft = resultados[b]["base"], resultados[b]["rft"]
    if base and rft:
        delta = rft["media"] - base["media"]
        esp = f"{base['esperado']:.1f}%→{rft['esperado']:.1f}%"
        dtxt = f"{delta:+.1f}pp"
        veredicto[b] = delta
    else:
        dtxt, esp = "—", "—"
    lines.append(f"| {b} | {fmt(base)} | {fmt(rft)} | {dtxt} | {esp} |")

lines.append("")
lines.append("## Veredicto")
aime_delta = veredicto.get("aime")
if aime_delta is not None:
    # ruido aprox de N=1 con P problemas: usar sd reportada
    rft_sd = resultados["aime"]["rft"]["sd"] if resultados["aime"]["rft"] else 0
    base_sd = resultados["aime"]["base"]["sd"] if resultados["aime"]["base"] else 0
    ruido = (rft_sd**2 + base_sd**2) ** 0.5
    if aime_delta > ruido:
        lines.append(f"- **AIME N=1: +{aime_delta:.1f}pp (> ruido ±{ruido:.1f}pp) → MEJORA REAL del modelo puro.**")
    elif aime_delta > 0:
        lines.append(f"- AIME N=1: +{aime_delta:.1f}pp (dentro del ruido ±{ruido:.1f}pp) → mejora no concluyente.")
    else:
        lines.append(f"- **AIME N=1: {aime_delta:.1f}pp → NO mejora. Considerar REVERTIR.**")
for b in ["gsm8k", "gpqa"]:
    d = veredicto.get(b)
    if d is not None:
        estado = "OK (no baja)" if d >= -2 else "REGRESIÓN"
        lines.append(f"- {b} N=1: {d:+.1f}pp → {estado}")
    else:
        lines.append(f"- {b}: no medido (sin acceso o sin datos).")

aime_ok = (aime_delta is not None and aime_delta > 0)
no_regresion = all((veredicto.get(b) is None or veredicto.get(b) >= -2) for b in ["gsm8k", "gpqa"])
if aime_ok and no_regresion:
    lines.append("\n**CONCLUSIÓN: candidato a Nova-v1 (sube N=1 sin regresión).**")
else:
    lines.append("\n**CONCLUSIÓN: no cumple criterio; documentar y revertir si procede.**")

reporte = "\n".join(lines)
print("\n" + reporte + "\n", flush=True)

dest = os.path.join(REPO_ROOT, "docs", "benchmarks", "fase_rft")
os.makedirs(dest, exist_ok=True)
with open(os.path.join(dest, "report_rft_n1.md"), "w", encoding="utf-8") as f:
    f.write(reporte)
with open(os.path.join(dest, "report_rft_n1.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2)

# subir a HF
for fn in ["report_rft_n1.md", "report_rft_n1.json"]:
    try:
        api.upload_file(path_or_fileobj=os.path.join(dest, fn), path_in_repo=fn,
                        repo_id=HF_REPO_DATA, repo_type="dataset", commit_message=f"rft report {fn}")
    except Exception as e:
        print(f"[CMP] aviso subida {fn}: {repr(e)[:120]}", flush=True)
print("[CMP] COMPARACIÓN COMPLETADA", flush=True)
