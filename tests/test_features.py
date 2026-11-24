"""Pruebas de selección de variables y reducción de dimensionalidad."""

import numpy as np
import pandas as pd

from src.features.selection import (correlation_filter, discriminant_baselines, importance_selection,
                                    reduce_dimensions)


def _datos(n: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    df = pd.DataFrame({
        "x1": x1,
        "x1_copia": x1 * 2 + rng.normal(0, 0.01, n),   # colineal con x1
        "x2": rng.normal(size=n),
        "ruido": rng.normal(size=n),
        "texto": rng.choice(["a", "b"], n),              # no numérica: se ignora
    })
    y = (x1 + 0.5 * df.x2 + rng.normal(0, 0.3, n) > 0).astype(int)
    return df, y


def test_filtro_de_colinealidad_descarta_la_segunda_variable():
    df, _ = _datos()
    conservar, pares = correlation_filter(df, threshold=0.8)
    assert "x1" in conservar and "x1_copia" not in conservar
    assert pares and pares[0][:2] == ("x1", "x1_copia") and pares[0][2] > 0.99


def test_seleccion_por_importancia_prioriza_las_variables_utiles():
    df, y = _datos()
    tabla = importance_selection(df, y, top_k=2)
    assert tabla.variable.iloc[0] in {"x1", "x1_copia"}
    assert tabla.seleccionada.sum() == 2
    assert "ruido" not in tabla[tabla.seleccionada].variable.tolist()


def test_pca_y_lda_reducen_dimensiones():
    df, y = _datos()
    proy, detalle = reduce_dimensions(df, n_components=2, method="pca")
    assert proy.shape == (len(df), 2) and detalle["metodo"] == "PCA"
    assert 0 < detalle["varianza_acumulada"] <= 1
    proy_lda, detalle_lda = reduce_dimensions(df, n_components=2, method="lda", y=y)
    assert proy_lda.shape == (len(df), 1)       # dos clases → una componente discriminante
    assert detalle_lda["componentes"] == 1


def test_lda_y_qda_como_referencia():
    df, y = _datos()
    tabla = discriminant_baselines(df, y)
    assert set(tabla.modelo) == {"LDA", "QDA"}
    assert (tabla.balanced_accuracy > 0.7).all()
