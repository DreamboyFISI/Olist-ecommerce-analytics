"""Ingeniería de características: selección y reducción de dimensionalidad.

Complementa ``src/data/build_master.py`` (que construye las variables) con las
operaciones que la propuesta pide demostrar explícitamente:

- **Eliminación de colinealidad**: descarta variables con |correlación| alta.
- **Selección por importancia**: conserva las variables que más aportan según
  un modelo de árboles o la importancia por permutación.
- **Reducción de dimensionalidad**: PCA (sin etiqueta) o LDA (con etiqueta);
  QDA no reduce dimensiones, se incluye solo como clasificador de referencia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_filter(df: pd.DataFrame, threshold: float = 0.8) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Variables a conservar y pares descartados por |ρ| >= threshold.

    De cada par correlacionado se elimina la segunda variable en el orden de
    columnas, de modo que el resultado es determinista.
    """
    numericas = df.select_dtypes(include="number")
    corr = numericas.corr().abs()
    descartadas: set[str] = set()
    pares: list[tuple[str, str, float]] = []
    columnas = list(corr.columns)
    for i, a in enumerate(columnas):
        if a in descartadas:
            continue
        for b in columnas[i + 1:]:
            if b in descartadas:
                continue
            rho = float(corr.loc[a, b])
            if rho >= threshold:
                descartadas.add(b)
                pares.append((a, b, round(rho, 4)))
    conservar = [c for c in columnas if c not in descartadas]
    return conservar, pares


def importance_selection(X: pd.DataFrame, y, top_k: int | None = None, min_importance: float = 0.0,
                         method: str = "tree", random_state: int = 42) -> pd.DataFrame:
    """Importancia de cada variable y marca de selección.

    ``method="tree"`` usa la impureza de un RandomForest; ``method="permutation"``
    usa la caída de score al permutar cada variable (más lento, agnóstico).
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance

    y = np.asarray(y)
    es_clasificacion = len(np.unique(y)) <= 20 and np.issubdtype(y.dtype, np.integer) or y.dtype == bool
    modelo = (RandomForestClassifier if es_clasificacion else RandomForestRegressor)(
        n_estimators=200, random_state=random_state, n_jobs=-1)
    Xn = X.select_dtypes(include="number").fillna(X.median(numeric_only=True))
    modelo.fit(Xn, y)
    if method == "permutation":
        res = permutation_importance(modelo, Xn, y, n_repeats=5, random_state=random_state, n_jobs=1)
        importancias = res.importances_mean
    else:
        importancias = modelo.feature_importances_
    tabla = (pd.DataFrame({"variable": Xn.columns, "importancia": importancias})
             .sort_values("importancia", ascending=False).reset_index(drop=True))
    tabla["seleccionada"] = tabla.importancia > min_importance
    if top_k is not None:
        tabla.loc[top_k:, "seleccionada"] = False
    return tabla


def reduce_dimensions(X: pd.DataFrame, n_components: int = 2, method: str = "pca", y=None,
                      random_state: int = 42) -> tuple[np.ndarray, dict]:
    """Proyección a ``n_components`` con PCA o LDA; devuelve (matriz, detalle).

    PCA no necesita etiqueta y maximiza varianza; LDA necesita ``y`` y maximiza
    la separación entre clases (a lo sumo clases − 1 componentes).
    """
    from sklearn.preprocessing import StandardScaler

    Xn = X.select_dtypes(include="number").fillna(X.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(Xn)
    if method == "pca":
        from sklearn.decomposition import PCA
        modelo = PCA(n_components=n_components, random_state=random_state).fit(Xs)
        proy = modelo.transform(Xs)
        detalle = {"metodo": "PCA", "varianza_explicada": modelo.explained_variance_ratio_.round(4).tolist(),
                   "varianza_acumulada": float(modelo.explained_variance_ratio_.cumsum()[-1].round(4))}
    elif method == "lda":
        if y is None:
            raise ValueError("LDA necesita la etiqueta y.")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clases = len(np.unique(y))
        k = min(n_components, clases - 1)
        modelo = LinearDiscriminantAnalysis(n_components=k).fit(Xs, y)
        proy = modelo.transform(Xs)
        detalle = {"metodo": "LDA", "componentes": k,
                   "varianza_explicada": modelo.explained_variance_ratio_.round(4).tolist()}
    else:
        raise ValueError("method debe ser 'pca' o 'lda'.")
    return proy, detalle


def discriminant_baselines(X: pd.DataFrame, y, random_state: int = 42) -> pd.DataFrame:
    """Exactitud balanceada de LDA y QDA en validación cruzada, como referencia lineal/cuadrática."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xn = X.select_dtypes(include="number").fillna(X.median(numeric_only=True))
    cv = StratifiedKFold(3, shuffle=True, random_state=random_state)
    filas = []
    for nombre, modelo in (("LDA", LinearDiscriminantAnalysis()),
                           ("QDA", QuadraticDiscriminantAnalysis(reg_param=0.1))):
        puntajes = cross_val_score(make_pipeline(StandardScaler(), modelo), Xn, y, cv=cv,
                                   scoring="balanced_accuracy")
        filas.append({"modelo": nombre, "balanced_accuracy": float(puntajes.mean()), "std": float(puntajes.std())})
    return pd.DataFrame(filas)
