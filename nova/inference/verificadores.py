"""Verificadores de tierra para el motor de inferencia (Fase 1).

- extraer_respuesta: saca la respuesta final de una muestra (reutiliza el extractor del arnes).
- es_valida: comprobacion de sanidad por benchmark (entero 0-999 en AIME, letra A-D en GPQA, numero en GSM8K).
- normalizar: forma canonica para agrupar respuestas iguales en el voto.
- es_correcto: comparacion con el gold (reutiliza el arnes; sympy como respaldo simbolico).

El sandbox de ejecucion de codigo se deja para una segunda iteracion (no hace falta
para AIME / GPQA / GSM8K, que son numericos o de opcion multiple).
"""
import math

import run_benchmark as rb  # arnes validado: extraer_num, extraer_letra, comparar_num


def extraer_respuesta(benchmark: str, texto: str) -> str:
    """Respuesta final extraida de una muestra, segun el tipo de benchmark."""
    if benchmark == "gpqa":
        return rb.extraer_letra(texto)
    return rb.extraer_num(texto)


def es_valida(benchmark: str, respuesta: str) -> bool:
    """¿La respuesta tiene la forma esperada? (verificador de tierra, filtra basura)."""
    if respuesta is None or respuesta == "":
        return False
    if benchmark == "gpqa":
        return respuesta.strip().upper() in ("A", "B", "C", "D")
    if benchmark == "aime":
        try:
            f = float(respuesta)
            if not math.isfinite(f):
                return False
            v = int(round(f))
            return 0 <= v <= 999
        except (TypeError, ValueError, OverflowError):
            return False
    try:
        float(respuesta)
        return True
    except (TypeError, ValueError):
        return False


def normalizar(benchmark: str, respuesta: str) -> str:
    """Forma canonica para que '204', '204.0' y ' 204 ' cuenten como el mismo voto."""
    if respuesta is None:
        return ""
    if benchmark == "gpqa":
        return respuesta.strip().upper()
    try:
        f = float(respuesta)
        if not math.isfinite(f):
            return respuesta.strip()
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError, OverflowError):
        return respuesta.strip()


def _equivalentes_sympy(a: str, b: str) -> bool:
    """Igualdad simbolica como respaldo (p. ej. fracciones/expresiones)."""
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
        return bool(sympy.simplify(parse_expr(str(a)) - parse_expr(str(b))) == 0)
    except Exception:
        return False


def es_correcto(benchmark: str, pred: str, gold: str) -> bool:
    """Comparacion robusta con el gold, reutilizando la logica del arnes."""
    if pred is None or pred == "":
        return False
    if benchmark == "gpqa":
        return pred.strip().upper() == str(gold).strip().upper()
    return rb.comparar_num(pred, gold) or _equivalentes_sympy(pred, gold)
