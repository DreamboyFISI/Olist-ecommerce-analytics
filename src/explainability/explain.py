"""Explicabilidad de predicciones individuales para pipelines de scikit-learn.

Tres caminos, del más preciso al más simple, y se usa el primero disponible:

1. **SHAP (TreeExplainer)** sobre el clasificador final del pipeline: exacto
   para modelos de árboles y no necesita datos de fondo.
2. **LIME tabular**, si la librería está instalada y existe una muestra de
   fondo del entrenamiento (``explain_background.csv`` en los artefactos).
3. **Perturbación con los valores típicos** aprendidos por los imputadores del
   pipeline: cuánto cambia la probabilidad al reemplazar cada variable por su
   mediana (numéricas) o su categoría más frecuente. Siempre disponible.

Las contribuciones de columnas one-hot se agregan a su variable original para
que la explicación hable el lenguaje del negocio (``categoria_principal`` y no
``categoria_principal_bed_bath_table``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _split_pipeline(pipeline) -> tuple[Any, Any]:
    steps = dict(pipeline.named_steps)
    prep = steps.get("prep")
    clf = steps.get("clf") or list(steps.values())[-1]
    return prep, clf


def _original_feature(nombre: str, variables: list[str]) -> str:
    """Mapea una columna transformada (p. ej. one-hot) a su variable de origen."""
    candidatos = [v for v in variables if nombre == v or nombre.startswith(v + "_")]
    return max(candidatos, key=len) if candidatos else nombre


def _aggregate(contribs: dict[str, float], variables: list[str]) -> dict[str, float]:
    agregado: dict[str, float] = {}
    for columna, valor in contribs.items():
        origen = _original_feature(columna, variables)
        agregado[origen] = agregado.get(origen, 0.0) + float(valor)
    return agregado


def _imputer_defaults(prep) -> dict[str, Any]:
    """Valor típico por variable según los imputadores ajustados del ColumnTransformer."""
    defaults: dict[str, Any] = {}
    if prep is None or not hasattr(prep, "transformers_"):
        return defaults
    for _, transformador, columnas in prep.transformers_:
        if transformador == "drop" or not hasattr(transformador, "steps"):
            continue
        imputer = transformador.steps[0][1]
        stats = getattr(imputer, "statistics_", None)
        if stats is None:
            continue
        for columna, valor in zip(columnas, stats):
            defaults[columna] = valor
    return defaults


def shap_contributions(pipeline, X: pd.DataFrame) -> dict[str, float] | None:
    """Contribuciones SHAP (log-odds) de la primera fila de X; None si SHAP no aplica."""
    try:
        import shap
    except ImportError:
        return None
    prep, clf = _split_pipeline(pipeline)
    try:
        Xt = prep.transform(X) if prep is not None else X.to_numpy()
        nombres = list(prep.get_feature_names_out()) if prep is not None else list(X.columns)
        explainer = shap.TreeExplainer(clf)
        valores = explainer.shap_values(Xt)
    except Exception:  # modelos no soportados por TreeExplainer
        return None
    valores = np.asarray(valores)
    if valores.ndim == 3:                       # (n, features, clases) o (clases, n, features)
        valores = valores[0, :, -1] if valores.shape[0] == 1 else valores[-1, 0, :]
    elif valores.ndim == 2:
        valores = valores[0]
    contribs = {nombre: float(v) for nombre, v in zip(nombres, np.ravel(valores))}
    return _aggregate(contribs, list(X.columns))


def lime_contributions(pipeline, X: pd.DataFrame, background: pd.DataFrame | None,
                       class_names=("Sin alerta", "Riesgo"), num_features: int = 10) -> dict[str, float] | None:
    """Explicación LIME sobre el espacio transformado; None si LIME o el fondo no están."""
    if background is None or background.empty:
        return None
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return None
    prep, clf = _split_pipeline(pipeline)
    try:
        fondo = prep.transform(background[X.columns]) if prep is not None else background.to_numpy()
        fila = prep.transform(X) if prep is not None else X.to_numpy()
        nombres = list(prep.get_feature_names_out()) if prep is not None else list(X.columns)
        explainer = LimeTabularExplainer(np.asarray(fondo), feature_names=nombres, class_names=list(class_names),
                                         discretize_continuous=False, random_state=42)
        explicacion = explainer.explain_instance(np.asarray(fila)[0], clf.predict_proba, num_features=num_features)
    except Exception:
        return None
    contribs = {}
    for regla, peso in explicacion.as_list():
        columna = next((n for n in sorted(nombres, key=len, reverse=True) if n in regla), regla)
        contribs[columna] = contribs.get(columna, 0.0) + float(peso)
    return _aggregate(contribs, list(X.columns))


def perturbation_contributions(pipeline, X: pd.DataFrame) -> dict[str, float]:
    """Cambio de probabilidad al reemplazar cada variable por su valor típico (siempre disponible)."""
    prep, _ = _split_pipeline(pipeline)
    defaults = _imputer_defaults(prep)
    base = float(pipeline.predict_proba(X)[:, 1][0])
    contribs = {}
    for columna in X.columns:
        if columna not in defaults:
            continue
        alterado = X.copy()
        alterado[columna] = defaults[columna]
        try:
            p = float(pipeline.predict_proba(alterado)[:, 1][0])
        except Exception:
            continue
        contribs[columna] = base - p      # positivo: la variable empuja el riesgo hacia arriba
    return contribs


def explain_instance(pipeline, X: pd.DataFrame, background: pd.DataFrame | None = None,
                     top_n: int = 10, prefer: str = "shap") -> dict:
    """Explicación de la primera fila de X con el mejor método disponible.

    Devuelve ``{"metodo", "unidad", "base", "contribuciones": [{variable, valor, contribucion}]}``.
    """
    metodo, contribs, unidad = None, None, None
    orden = {"shap": ["shap", "lime", "perturbacion"], "lime": ["lime", "shap", "perturbacion"],
             "perturbacion": ["perturbacion"]}.get(prefer, ["shap", "lime", "perturbacion"])
    for candidato in orden:
        if candidato == "shap":
            contribs = shap_contributions(pipeline, X)
            unidad = "log-odds"
        elif candidato == "lime":
            contribs = lime_contributions(pipeline, X, background)
            unidad = "peso LIME"
        else:
            contribs = perturbation_contributions(pipeline, X)
            unidad = "cambio de probabilidad"
        if contribs:
            metodo = candidato
            break
    contribs = contribs or {}
    filas = [{"variable": k, "valor": _json_value(X.iloc[0][k]) if k in X.columns else None,
              "contribucion": round(float(v), 5)} for k, v in contribs.items()]
    filas.sort(key=lambda f: abs(f["contribucion"]), reverse=True)
    return {"metodo": metodo or "ninguno", "unidad": unidad,
            "probabilidad": round(float(pipeline.predict_proba(X)[:, 1][0]), 4),
            "contribuciones": filas[:top_n]}


def _json_value(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)
