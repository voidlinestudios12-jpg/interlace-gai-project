"""Motor de computo en inferencia (Fase 1).

Dado el conjunto de N muestras de un problema, selecciona la respuesta final con
tres metodos comparables (de simple a sofisticado):
  - 'mayoria'      : voto mayoritario / self-consistency (sin verificador).
  - 'autocerteza'  : voto ponderado por la confianza del propio modelo (logprobs).
  - 'verificador'  : voto mayoritario SOLO entre muestras que pasan el verificador de tierra.

Cada muestra es un dict: {"respuesta": str, "certeza": float}  (certeza = media de logprob).
Tambien expone acuerdo() para la parada anticipada compute-optima (§7.1d).
"""
import math
from collections import defaultdict

import verificadores as V


def acuerdo(benchmark: str, muestras) -> float:
    """Fraccion de muestras que coinciden con la respuesta mas frecuente (0..1)."""
    if not muestras:
        return 0.0
    cuenta = defaultdict(int)
    for m in muestras:
        cuenta[V.normalizar(benchmark, m["respuesta"])] += 1
    return max(cuenta.values()) / len(muestras)


def _ganadora(pesos: dict) -> str:
    """Clave (respuesta normalizada) con mayor peso; "" si no hay ninguna."""
    if not pesos:
        return ""
    return max(pesos.items(), key=lambda kv: kv[1])[0]


def seleccionar(benchmark: str, muestras, metodo: str) -> str:
    """Devuelve la respuesta final (string) elegida entre las muestras.

    Metodos:
      - mayoria         : voto simple (1 por muestra).
      - autocerteza     : voto ponderado por la confianza del modelo (exp(media logprob)).
      - verificador     : voto simple SOLO entre muestras que pasan el verificador de tierra.
      - autoverificacion: voto ponderado por la puntuacion de autoverificacion (v_score:
                          el PROPIO modelo juzgo si la solucion es correcta). Si nadie fue
                          aprobado (todas v_score=0), cae a mayoria.
    """
    if metodo == "mayoria":
        candidatas = muestras

        def peso(m):
            return 1.0
    elif metodo == "autocerteza":
        candidatas = muestras

        def peso(m):
            return math.exp(m.get("certeza", 0.0))
    elif metodo == "verificador":
        candidatas = [m for m in muestras if V.es_valida(benchmark, m["respuesta"])] or muestras

        def peso(m):
            return 1.0
    elif metodo == "autoverificacion":
        candidatas = muestras

        def peso(m):
            return float(m.get("v_score", 0.0))
    else:
        raise ValueError(f"metodo desconocido: {metodo}")

    pesos = defaultdict(float)
    repr_orig = {}  # respuesta_normalizada -> una forma original legible
    for m in candidatas:
        r = V.normalizar(benchmark, m["respuesta"])
        if r == "":
            continue
        pesos[r] += peso(m)
        repr_orig.setdefault(r, m["respuesta"])

    # si ninguna respuesta acumulo peso (p. ej. autoverificacion sin aprobadas) -> mayoria
    if (not pesos or max(pesos.values()) <= 0) and metodo != "mayoria":
        return seleccionar(benchmark, muestras, "mayoria")
    ganadora = _ganadora(pesos)
    return repr_orig.get(ganadora, ganadora)


def indice_elegido(benchmark, muestras, metodo):
    """Indice de la muestra REPRESENTATIVA de la respuesta elegida (para recuperar
    su texto completo en correr_nova). Si no hay respuesta extraible (p. ej. pregunta
    general), devuelve la de mayor certeza = best-of-N por confianza."""
    if not muestras:
        return -1
    elegida = V.normalizar(benchmark, seleccionar(benchmark, muestras, metodo))
    cand = [i for i, m in enumerate(muestras)
            if elegida != "" and V.normalizar(benchmark, m["respuesta"]) == elegida]
    universo = cand if cand else list(range(len(muestras)))
    return max(universo, key=lambda i: muestras[i].get("certeza", float("-inf")))
