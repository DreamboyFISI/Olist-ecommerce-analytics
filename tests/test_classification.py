"""Pruebas del módulo de clasificación (riesgo de reseña negativa)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import classification as clf
from src.models.classification import (ARTIFACTS_DIR, CAT, FEATURES, NUM, NUM_COMPRA, NUM_ENCUESTA, RAW_DIR,
                                       RAW_FILES, add_features, build_preprocessor, gains_table,
                                       holdout_metrics, select_by_parsimony, study_population,
                                       temporal_split, validate_manifest)

ROOT = Path(__file__).resolve().parents[1]
RAW_AVAILABLE = all((RAW_DIR / name).exists() for name in RAW_FILES.values())
ARTIFACTS_AVAILABLE = (ARTIFACTS_DIR / "model.joblib").exists() and (ARTIFACTS_DIR / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Pruebas sin datos: lógica de variables y de selección
# ---------------------------------------------------------------------------
def _pedido(**overrides) -> pd.DataFrame:
    base = {
        "order_id": "a", "order_purchase_timestamp": pd.Timestamp("2018-07-10 14:30"),
        "order_approved_at": pd.Timestamp("2018-07-10 15:00"),
        "order_delivered_carrier_date": pd.Timestamp("2018-07-13 10:00"),
        "order_delivered_customer_date": pd.Timestamp("2018-07-24 12:00"),
        "order_estimated_delivery_date": pd.Timestamp("2018-08-03"),
        "shipping_limit_date": pd.Timestamp("2018-07-16 23:59"),
        "review_creation_date": pd.Timestamp("2018-07-25"), "review_score": 5,
        "freight_total": 18.0, "price_total": 120.0, "n_items": 2, "n_productos": 2, "n_sellers": 1,
        "n_categorias": 1, "peso_total_g": 1500.0, "volumen_total_cm3": 12000.0, "fotos_media": 2.0,
        "largo_descripcion_media": 600.0, "n_cuotas": 1, "n_medios_pago": 1, "payment_type": "credit_card",
        "categoria_principal": "bed_bath_table",
        "cust_lat": -22.9, "cust_lng": -43.2, "seller_lat": -23.5, "seller_lng": -46.6,
        "customer_state": "RJ", "seller_state": "SP",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_variables_de_un_pedido_entregado_a_tiempo():
    df = add_features(_pedido())
    fila = df.iloc[0]
    assert fila.pendiente_en_encuesta == 0
    assert fila.estado_en_encuesta == "Entregado a tiempo"
    assert fila.dias_retraso < 0            # llegó antes de la fecha prometida
    assert fila.entrega_tardia == 0
    assert fila.dias_entrega == pytest.approx(13.9, abs=0.1)
    assert fila.mismo_estado == 0 and fila.region_vendedor == "Sudeste"
    assert 300 < fila.distancia_km < 450     # Rio de Janeiro ↔ São Paulo
    assert fila.review_negativa == 0


def test_pedido_pendiente_usa_solo_lo_observado_en_la_encuesta():
    """Nada posterior a la encuesta puede entrar al modelo."""
    df = add_features(_pedido(order_delivered_customer_date=pd.Timestamp("2018-08-10"),
                              review_creation_date=pd.Timestamp("2018-08-05"), review_score=1))
    fila = df.iloc[0]
    assert fila.pendiente_en_encuesta == 1
    assert fila.estado_en_encuesta == "Pendiente"
    assert np.isnan(fila.dias_entrega) and np.isnan(fila.dias_transporte)
    # Días que lleva vencida la promesa al enviar la encuesta: 05-ago − 03-ago = 2
    assert fila.dias_retraso == pytest.approx(2.0)
    assert fila.review_negativa == 1


def test_sin_resena_no_hay_etiqueta():
    df = add_features(_pedido(review_score=np.nan))
    assert np.isnan(df.iloc[0].review_negativa)


def test_poblacion_de_estudio_filtra_lo_que_no_sirve():
    df = add_features(pd.concat([
        _pedido(order_id="ok"),
        _pedido(order_id="sin_resena", review_score=np.nan),
        _pedido(order_id="2016", order_purchase_timestamp=pd.Timestamp("2016-10-01"),
                order_approved_at=pd.Timestamp("2016-10-01")),
        _pedido(order_id="sin_items", n_items=np.nan),
        _pedido(order_id="fechas_raras", order_delivered_customer_date=pd.Timestamp("2018-07-12"),
                order_delivered_carrier_date=pd.Timestamp("2018-07-13")),
    ], ignore_index=True))
    base, embudo = study_population(df)
    assert base.order_id.tolist() == ["ok"]
    assert embudo.pedidos.tolist() == [5, 4, 3, 2, 1]


def test_la_particion_es_temporal():
    df = add_features(pd.concat([
        _pedido(order_id="antes", order_purchase_timestamp=pd.Timestamp("2018-05-31 23:00"),
                order_approved_at=pd.Timestamp("2018-05-31 23:30")),
        _pedido(order_id="despues", order_purchase_timestamp=pd.Timestamp("2018-06-01 00:10"),
                order_approved_at=pd.Timestamp("2018-06-01 01:00")),
    ], ignore_index=True))
    train, test = temporal_split(df)
    assert train.order_id.tolist() == ["antes"] and test.order_id.tolist() == ["despues"]


def test_listas_de_variables_son_consistentes():
    assert FEATURES == NUM + CAT
    assert set(NUM) == set(NUM_COMPRA) | set(NUM_ENCUESTA)
    assert not set(NUM_COMPRA) & set(NUM_ENCUESTA)
    assert len(FEATURES) == 33


def test_preprocesador_produce_matriz_numerica_sin_nulos():
    df = add_features(pd.concat([_pedido(order_id=str(i), price_total=100 + i) for i in range(6)], ignore_index=True))
    df.loc[0, "distancia_km"] = np.nan           # imputación por mediana
    X = build_preprocessor(NUM, CAT).fit_transform(df[FEATURES])
    assert X.shape[0] == 6 and X.shape[1] >= len(NUM)
    assert np.isfinite(X).all()


def test_parsimonia_elige_el_mas_simple_entre_empatados():
    cv_df = pd.DataFrame([
        {"modelo": "Random Forest", "pr_auc_mean": 0.554, "pr_auc_std": 0.011},
        {"modelo": "HistGradientBoosting", "pr_auc_mean": 0.554, "pr_auc_std": 0.006},
        {"modelo": "XGBoost", "pr_auc_mean": 0.553, "pr_auc_std": 0.009},
        {"modelo": "Regresión Logística", "pr_auc_mean": 0.520, "pr_auc_std": 0.008},
    ])
    ganador, empatados = select_by_parsimony(cv_df)
    assert ganador == "HistGradientBoosting"
    assert "Regresión Logística" not in empatados


def test_metricas_y_ganancia_con_un_ordenamiento_perfecto():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = holdout_metrics(y, p, (p >= 0.5).astype(int))
    assert m["roc_auc"] == 1 and m["pr_auc"] == 1 and m["recall"] == 1
    g = gains_table(y, p)
    assert g.loc[1, "captura_acumulada_%"] == pytest.approx(50)
    assert g.loc[2, "captura_acumulada_%"] == pytest.approx(100)
    assert g.loc[1, "lift"] == pytest.approx(5)   # 100% de negativas en el decil vs 20% de prevalencia


# ---------------------------------------------------------------------------
# Pruebas con artefactos: contrato e inferencia
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="faltan artefactos en artifacts/classification")
def test_manifest_cumple_el_contrato():
    manifest = json.loads((ARTIFACTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert manifest["status"] == "Completado"
    assert 0 < manifest["metrics"]["pr_auc"] <= 1
    assert manifest["metrics"]["pr_auc"] > manifest["metrics"]["prevalencia_test"]


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="faltan artefactos en artifacts/classification")
def test_chart_data_tiene_lo_que_consume_el_dashboard():
    chart = json.loads((ARTIFACTS_DIR / "chart_data.json").read_text(encoding="utf-8"))
    for clave in ("threshold", "prevalence_test", "model_comparison_cv", "model_comparison_test",
                  "imbalance_strategies", "confusion_matrix", "roc_curve", "pr_curve", "gains_by_decile",
                  "feature_importance", "scenario_comparison"):
        assert clave in chart, clave
    assert len(chart["gains_by_decile"]) == 10
    cm = chart["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == sum(map(sum, cm["matrix"]))


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="faltan artefactos en artifacts/classification")
def test_inferencia_devuelve_probabilidad_valida_y_respeta_el_umbral():
    from backend.services.classification import predict_risk

    payload = {
        "purchase_timestamp": "2018-07-10T14:30:00", "survey_timestamp": "2018-07-25T00:00:00",
        "estimated_delivery_date": "2018-08-03T00:00:00", "shipping_limit_date": "2018-07-16T23:59:00",
        "delivered_carrier_date": "2018-07-13T12:00:00", "delivered_customer_date": "2018-07-24T12:00:00",
        "n_items": 1, "n_products": 1, "n_sellers": 1, "n_categories": 1,
        "price_total": 120.0, "freight_total": 18.0, "installments": 1, "payment_type": "credit_card",
        "customer_state": "RJ", "seller_state": "SP", "distance_km": 400.0, "main_category": "bed_bath_table",
    }
    r = predict_risk(payload)
    assert 0 <= r["probabilidad"] <= 1
    assert r["prediccion"] == int(r["probabilidad"] >= r["umbral"])
    assert r["contexto"]["estado_en_encuesta"] == "Entregado a tiempo"
    assert set(r["variables"]) == set(FEATURES)


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="faltan artefactos en artifacts/classification")
def test_un_pedido_pendiente_y_vencido_es_mas_riesgoso_que_uno_puntual():
    from backend.services.classification import predict_risk

    comun = {"purchase_timestamp": "2018-07-10T14:30:00", "estimated_delivery_date": "2018-07-24T00:00:00",
             "shipping_limit_date": "2018-07-16T23:59:00", "n_items": 1, "price_total": 120.0,
             "freight_total": 18.0, "payment_type": "credit_card", "customer_state": "RJ",
             "seller_state": "SP", "distance_km": 400.0, "main_category": "bed_bath_table"}
    puntual = predict_risk({**comun, "survey_timestamp": "2018-07-21T00:00:00",
                            "delivered_carrier_date": "2018-07-12T12:00:00",
                            "delivered_customer_date": "2018-07-20T12:00:00"})
    pendiente = predict_risk({**comun, "survey_timestamp": "2018-08-05T00:00:00",
                              "delivered_carrier_date": "2018-07-20T12:00:00",
                              "delivered_customer_date": None})
    assert pendiente["contexto"]["estado_en_encuesta"] == "Pendiente"
    assert pendiente["probabilidad"] > puntual["probabilidad"]


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="faltan artefactos en artifacts/classification")
def test_fechas_incoherentes_se_rechazan():
    from backend.services.classification import build_feature_frame

    with pytest.raises(ValueError):
        build_feature_frame({"purchase_timestamp": "2018-07-10", "survey_timestamp": "2018-07-01",
                             "price_total": 10, "customer_state": "SP", "seller_state": "SP"})


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="faltan artefactos en artifacts/classification")
def test_la_api_expone_resultados_y_prediccion():
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    assert client.get("/api/v1/classification/results").status_code == 200
    respuesta = client.post("/api/v1/classification/predict", json={
        "purchase_timestamp": "2018-07-10T14:30:00", "price_total": 120.0,
        "customer_state": "RJ", "seller_state": "SP", "main_category": "bed_bath_table",
    })
    assert respuesta.status_code == 200, respuesta.text
    assert "probabilidad" in respuesta.json()


# ---------------------------------------------------------------------------
# Prueba de extremo a extremo (requiere los CSV de Olist en data/raw)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RAW_AVAILABLE, reason="faltan los CSV de Olist en data/raw")
def test_corrida_corta_genera_artefactos_validos(tmp_path):
    manifest = clf.run_pipeline(artifacts_dir=tmp_path, quick=True)
    for nombre in ("manifest.json", "results.csv", "chart_data.json", "model.joblib"):
        assert (tmp_path / nombre).exists(), nombre
    assert manifest["metrics"]["pr_auc"] > manifest["metrics"]["prevalencia_test"]
