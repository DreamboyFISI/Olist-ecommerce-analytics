"""
Pruebas unitarias para el módulo de Clustering — DataSmart Olist.
================================================================
Responsable: Módulo de Clustering (rama feature/clustering)
Proyecto Integrador · Minería de Datos · UNMSM FISI 2026

Pruebas con datos sintéticos mínimos en memoria (sin depender de data/raw):
  1. Cálculo y agregación correcta de métricas RFM por customer_unique_id.
  2. Funcionalidad del método de inferencia asignar_segmento.
  3. Validación estricta del contrato manifest.json contra docs/module_contract.schema.json.
"""

import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.clustering import (
    SEGMENT_NAMES,
    asignar_segmento,
    construir_features,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts" / "clustering"
SCHEMA_PATH = ROOT / "docs" / "module_contract.schema.json"


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures con Datos Mínimos Sintéticos
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_data() -> dict[str, pd.DataFrame]:
    """Crea un dataset transaccional sintético para pruebas rápidas y reproducibles."""
    # 3 clientes únicos:
    # - user_1: 2 pedidos entregados (recomprador)
    # - user_2: 1 pedido entregado (comprador único)
    # - user_3: 1 pedido cancelado (debe ser ignorado por filtro de delivered)
    customers = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3", "c4"],
        "customer_unique_id": ["user_1", "user_1", "user_2", "user_3"],
        "customer_zip_code_prefix": [1000, 1000, 2000, 3000],
        "customer_city": ["sao paulo", "sao paulo", "rio", "curitiba"],
        "customer_state": ["SP", "SP", "RJ", "PR"],
    })

    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4"],
        "customer_id": ["c1", "c2", "c3", "c4"],
        "order_status": ["delivered", "delivered", "delivered", "canceled"],
        "order_purchase_timestamp": pd.to_datetime([
            "2018-01-10 10:00:00",
            "2018-03-15 14:00:00",
            "2018-05-20 18:00:00",
            "2018-06-01 12:00:00",
        ]),
        "order_approved_at": pd.to_datetime([
            "2018-01-10 11:00:00",
            "2018-03-15 15:00:00",
            "2018-05-20 19:00:00",
            "2018-06-01 13:00:00",
        ]),
        "order_delivered_carrier_date": pd.to_datetime([
            "2018-01-12 10:00:00",
            "2018-03-17 10:00:00",
            "2018-05-22 10:00:00",
            pd.NaT,
        ]),
        "order_delivered_customer_date": pd.to_datetime([
            "2018-01-20 12:00:00",
            "2018-03-22 16:00:00",
            "2018-05-28 15:00:00",
            pd.NaT,
        ]),
        "order_estimated_delivery_date": pd.to_datetime([
            "2018-01-25 00:00:00",
            "2018-03-20 00:00:00",  # Entregado con retraso de 2 días
            "2018-06-05 00:00:00",
            "2018-06-15 00:00:00",
        ]),
    })

    payments = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "payment_sequential": [1, 1, 1],
        "payment_type": ["credit_card", "boleto", "credit_card"],
        "payment_installments": [2, 1, 3],
        "payment_value": [100.0, 150.0, 80.0],
    })

    items = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "order_item_id": [1, 1, 1],
        "product_id": ["p1", "p2", "p3"],
        "seller_id": ["s1", "s1", "s2"],
        "shipping_limit_date": ["2018-01-15", "2018-03-20", "2018-05-25"],
        "price": [80.0, 130.0, 60.0],
        "freight_value": [20.0, 20.0, 20.0],
    })

    reviews = pd.DataFrame({
        "review_id": ["r1", "r2", "r3"],
        "order_id": ["o1", "o2", "o3"],
        "review_score": [5, 3, 4],
        "review_comment_title": [None, None, None],
        "review_comment_message": [None, None, None],
        "review_creation_date": ["2018-01-21", "2018-03-23", "2018-05-29"],
        "review_answer_timestamp": ["2018-01-22", "2018-03-24", "2018-05-30"],
    })

    products = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "product_category_name": ["telefonia", "informatica", "moda"],
    })

    return {
        "customers": customers,
        "orders": orders,
        "payments": payments,
        "items": items,
        "reviews": reviews,
        "products": products,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Construcción y Cálculo Correcto de RFM
# ═══════════════════════════════════════════════════════════════════════════

def test_construir_features_rfm(synthetic_data):
    """
    Verifica que:
    - Se agrupe estrictamente por customer_unique_id (2 clientes válidos resultantes).
    - Se descarte el pedido con estado cancelado.
    - user_1 tenga Frequency = 2 y Monetary = 250.0 (100 + 150).
    - user_2 tenga Frequency = 1 y Monetary = 80.0.
    - La recencia sea menor para el pedido más reciente (user_2 < user_1).
    - Existan los campos requeridos para Conjunto A, CLV y ABC.
    """
    df = construir_features(
        customers=synthetic_data["customers"],
        orders=synthetic_data["orders"],
        payments=synthetic_data["payments"],
        items=synthetic_data["items"],
        reviews=synthetic_data["reviews"],
        products=synthetic_data["products"],
    )

    # 1. Total de clientes válidos
    assert len(df) == 2, f"Se esperaban 2 clientes únicos válidos, se obtuvieron {len(df)}"
    assert set(df["customer_unique_id"]) == {"user_1", "user_2"}

    # 2. user_1 (recomprador)
    u1 = df[df["customer_unique_id"] == "user_1"].iloc[0]
    assert u1["Frequency"] == 2
    assert pytest.approx(u1["Monetary"], 0.01) == 250.0
    assert pytest.approx(u1["AvgTicket"], 0.01) == 125.0
    assert u1["CLV_historical"] == 250.0
    assert u1["ABC"] in {"A", "B", "C"}

    # 3. user_2 (comprador único)
    u2 = df[df["customer_unique_id"] == "user_2"].iloc[0]
    assert u2["Frequency"] == 1
    assert pytest.approx(u2["Monetary"], 0.01) == 80.0
    assert u2["Recency"] < u1["Recency"]  # user_2 compró en mayo, user_1 en marzo

    # 4. Columnas de Conjunto A presentes y sin NaN
    for col in ["feat_log_R", "feat_log_F", "feat_log_M"]:
        assert col in df.columns
        assert not df[col].isna().any()

    # user_2 tiene F=1 -> feat_log_F debe ser exactamente 0.0
    assert pytest.approx(u2["feat_log_F"], 1e-6) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Inferencia con asignar_segmento
# ═══════════════════════════════════════════════════════════════════════════

def test_asignar_segmento(tmp_path):
    """
    Verifica que asignar_segmento prediga correctamente clusters y nombres
    de segmento a partir de un pipeline serializado.
    """
    import joblib

    # Crear pipeline sintético
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=5)),
    ])
    X_dummy = np.array([
        [1.0, 0.0, 2.0],
        [4.0, 0.0, 5.0],
        [3.0, 1.0, 6.0],
        [5.0, 0.0, 1.0],
    ])
    pipe.fit(X_dummy)

    model_file = tmp_path / "model.joblib"
    joblib.dump({
        "pipeline": pipe,
        "feature_cols": ["feat_log_R", "feat_log_F", "feat_log_M"],
        "winner_name": "K-Means (k=4)",
        "cluster_to_name": SEGMENT_NAMES,
    }, model_file)

    # DataFrame de nuevos clientes
    new_customers = pd.DataFrame({
        "customer_unique_id": ["new_1", "new_2"],
        "Recency": [30, 300],
        "Frequency": [1, 2],
        "Monetary": [150.0, 500.0],
    })

    result = asignar_segmento(new_customers, model_path=model_file)

    assert "cluster" in result.columns
    assert "cluster_name" in result.columns
    assert "business_action" in result.columns
    assert len(result) == 2
    assert result["cluster"].isin([0, 1, 2, 3]).all()
    assert result["cluster_name"].notna().all()


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Validación del Manifest contra el Esquema del Contrato
# ═══════════════════════════════════════════════════════════════════════════

def test_manifest_schema_validation():
    """
    Valida que artifacts/clustering/manifest.json exista y cumpla
    estrictamente con el esquema formal docs/module_contract.schema.json.
    """
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    assert manifest_path.exists(), f"El manifest no existe en: {manifest_path}"
    assert SCHEMA_PATH.exists(), f"El esquema no existe en: {SCHEMA_PATH}"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    # Validación formal mediante jsonschema
    jsonschema.validate(instance=manifest, schema=schema)

    # Verificaciones semánticas del contrato de clustering
    assert manifest["status"] in ["Pendiente", "En entrenamiento", "Entrenado", "Completado"]
    assert manifest["winner"] != ""
    assert "silhouette_score" in manifest["metrics"]
    assert manifest["metrics"]["silhouette_score"] is not None

    # Comprobar que los archivos referenciados en el manifest existen
    res_path = ROOT / manifest["results_path"]
    chart_path = ROOT / manifest["chart_data_path"]
    assert res_path.exists(), f"No existe el results_path: {res_path}"
    assert chart_path.exists(), f"No existe el chart_data_path: {chart_path}"
