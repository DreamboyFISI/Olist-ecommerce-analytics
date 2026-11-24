"""Medidas de deriva (drift) entre una distribución de referencia y una actual.

Se usan sobre las probabilidades, errores y variables que cada módulo deja en
sus artefactos, sin depender de librerías externas: PSI para numéricas,
distancia de variación total para categóricas y cambio relativo para tasas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Umbrales habituales en la industria para el PSI (Population Stability Index).
PSI_VIGILAR = 0.10
PSI_ALERTA = 0.25


def psi(reference, current, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index entre dos muestras numéricas.

    Los cortes se definen con cuantiles de la referencia, de modo que cada
    tramo tenga la misma masa inicial; así el índice mide cuánto se movió la
    población actual respecto a esa partición. < 0.10 estable, 0.10–0.25
    vigilar, > 0.25 cambio importante.
    """
    ref = pd.Series(reference, dtype=float).dropna().to_numpy()
    cur = pd.Series(current, dtype=float).dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:            # referencia casi constante: no hay partición posible
        return 0.0 if np.allclose(ref.mean(), cur.mean()) else float("inf")
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, bins=cortes)[0] / len(ref)
    p_cur = np.histogram(cur, bins=cortes)[0] / len(cur)
    p_ref, p_cur = np.clip(p_ref, eps, None), np.clip(p_cur, eps, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def categorical_shift(reference, current) -> float:
    """Distancia de variación total entre dos distribuciones categóricas (0 a 1)."""
    ref = pd.Series(reference).dropna().astype(str).value_counts(normalize=True)
    cur = pd.Series(current).dropna().astype(str).value_counts(normalize=True)
    if ref.empty or cur.empty:
        return float("nan")
    categorias = ref.index.union(cur.index)
    return float(0.5 * np.abs(ref.reindex(categorias, fill_value=0) - cur.reindex(categorias, fill_value=0)).sum())


def relative_change(reference: float, current: float) -> float:
    """Cambio relativo (current - reference) / |reference|; NaN si no se puede calcular."""
    if reference is None or current is None or np.isnan(reference) or np.isnan(current) or reference == 0:
        return float("nan")
    return float((current - reference) / abs(reference))


def drift_status(valor: float, vigilar: float = PSI_VIGILAR, alerta: float = PSI_ALERTA) -> str:
    if valor is None or np.isnan(valor):
        return "Sin datos"
    if valor >= alerta:
        return "Alerta"
    if valor >= vigilar:
        return "Vigilar"
    return "OK"
