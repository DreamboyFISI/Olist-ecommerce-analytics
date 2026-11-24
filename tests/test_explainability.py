"""Pruebas de explicabilidad: contribuciones por SHAP, LIME y perturbación sobre un pipeline pequeño."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.explainability.explain import (explain_instance, lime_contributions, perturbation_contributions,
                                        shap_contributions)
from src.models.classification import ARTIFACTS_DIR


def _datos(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "retraso": rng.normal(0, 5, n),
        "precio": rng.lognormal(4, 0.5, n),
        "categoria": rng.choice(["hogar", "moda", "tecno"], n),
    })
    # El retraso manda; la categoría "moda" suma riesgo; el precio no importa
    logit = 0.5 * df.retraso + np.where(df.categoria == "moda", 1.0, 0.0) - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, y


def _pipeline(clf):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["retraso", "precio"]),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False)), ["categoria"]),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("clf", clf)])


@pytest.fixture(scope="module")
def arbol():
    df, y = _datos()
    return _pipeline(HistGradientBoostingClassifier(max_iter=100, random_state=0)).fit(df, y), df


@pytest.fixture(scope="module")
def lineal():
    df, y = _datos()
    return _pipeline(LogisticRegression(max_iter=500)).fit(df, y), df


def test_perturbacion_identifica_la_variable_dominante(arbol):
    pipe, df = arbol
    fila = pd.DataFrame([{"retraso": 12.0, "precio": 50.0, "categoria": "hogar"}])
    contrib = perturbation_contributions(pipe, fila)
    assert set(contrib) == {"retraso", "precio", "categoria"}
    assert contrib["retraso"] > 0 and contrib["retraso"] > abs(contrib["precio"])


def test_shap_agrupa_one_hot_en_la_variable_original(arbol):
    pytest.importorskip("shap")
    pipe, df = arbol
    fila = pd.DataFrame([{"retraso": 12.0, "precio": 50.0, "categoria": "moda"}])
    contrib = shap_contributions(pipe, fila)
    assert contrib is not None
    assert set(contrib) == {"retraso", "precio", "categoria"}
    assert contrib["retraso"] > 0


def test_shap_devuelve_none_para_modelos_no_soportados(lineal):
    pipe, _ = lineal
    fila = pd.DataFrame([{"retraso": 1.0, "precio": 50.0, "categoria": "hogar"}])
    # TreeExplainer no acepta modelos lineales: debe caer al siguiente método sin romper
    assert shap_contributions(pipe, fila) is None
    exp = explain_instance(pipe, fila)
    assert exp["metodo"] in ("lime", "perturbacion") and exp["contribuciones"]


def test_lime_usa_el_fondo_cuando_existe(lineal):
    pytest.importorskip("lime")
    pipe, df = lineal
    fila = pd.DataFrame([{"retraso": 8.0, "precio": 50.0, "categoria": "moda"}])
    contrib = lime_contributions(pipe, fila, df.sample(200, random_state=0))
    assert contrib is not None and "retraso" in contrib
    assert lime_contributions(pipe, fila, None) is None


def test_explain_instance_ordena_por_magnitud(arbol):
    pipe, df = arbol
    fila = pd.DataFrame([{"retraso": -6.0, "precio": 50.0, "categoria": "hogar"}])
    exp = explain_instance(pipe, fila, top_n=3)
    valores = [abs(c["contribucion"]) for c in exp["contribuciones"]]
    assert valores == sorted(valores, reverse=True)
    assert 0 <= exp["probabilidad"] <= 1
    assert exp["contribuciones"][0]["valor"] is not None


@pytest.mark.skipif(not (ARTIFACTS_DIR / "model.joblib").exists(), reason="faltan artefactos de clasificación")
def test_explicacion_del_modelo_real_via_servicio():
    from backend.services.classification import explain_prediction

    exp = explain_prediction({"purchase_timestamp": "2018-07-10T14:30:00", "survey_timestamp": "2018-08-05T00:00:00",
                              "estimated_delivery_date": "2018-07-24T00:00:00", "price_total": 120.0,
                              "customer_state": "RJ", "seller_state": "SP", "main_category": "bed_bath_table"})
    assert exp["metodo"] != "ninguno"
    assert exp["contribuciones"] and exp["contexto"]["estado_en_encuesta"] == "Pendiente"
