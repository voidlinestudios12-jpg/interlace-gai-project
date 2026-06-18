"""Fase 2 — preparar datos SFT.

Descarga Light-R1-SFTData (qihoo360/Light-R1-SFTData) — trazas de mates duras de
DeepSeek-R1, YA DESCONTAMINADAS contra AIME24/25, MATH-500 y GPQA-Diamond (exacto +
N-grama, segun la tarjeta del dataset). Filtra trazas con <think>...</think> y
\\boxed{}, descarta las demasiado largas, y guarda un .jsonl versionado de
{"messages":[user, assistant]} en el volumen nova-data (/data/sft/).

Uso: modal run nova/data/preparar_sft.py --n 1000
"""
import modal

app = modal.App("nova-sft-data")
vol = modal.Volume.from_name("nova-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("datasets", "huggingface_hub", "requests")


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def preparar(n: int, max_chars: int):
    import json
    import os

    from datasets import load_dataset

    os.makedirs("/data/sft", exist_ok=True)
    ruta = f"/data/sft/light_r1_n{n}.jsonl"
    ds = load_dataset("qihoo360/Light-R1-SFTData", split="train", streaming=True)

    escritas = saltadas = 0
    with open(ruta, "w", encoding="utf-8") as f:
        for ej in ds:
            conv = ej.get("conversations") or []
            user = next((c.get("value") for c in conv if c.get("from") == "user"), None)
            asst = next((c.get("value") for c in conv if c.get("from") == "assistant"), None)
            if not user or not asst:
                saltadas += 1
                continue
            if "</think>" not in asst or "\\boxed{" not in asst:
                saltadas += 1
                continue
            if len(user) + len(asst) > max_chars:  # descarta trazas larguisimas (memoria)
                saltadas += 1
                continue
            f.write(json.dumps({"messages": [
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": asst.strip()},
            ]}, ensure_ascii=False) + "\n")
            escritas += 1
            if escritas >= n:
                break
    vol.commit()
    print(f"[SFT-DATA] {escritas} trazas escritas (saltadas {saltadas}) -> {ruta}", flush=True)
    return {"ruta": ruta, "escritas": escritas, "saltadas": saltadas}


@app.local_entrypoint()
def main(n: int = 1000, max_chars: int = 30000):
    print(f"Preparando {n} trazas de Light-R1 (descontaminado) ...")
    print(preparar.remote(n, max_chars))
