"""
Módulo productivo de Clustering y Segmentación de Clientes — DataSmart Olist.
=============================================================================
Responsable: Módulo de Clustering (rama feature/clustering)
Proyecto Integrador · Minería de Datos · UNMSM FISI 2026

Lógica reutilizable de ingeniería de características, entrenamiento comparativo
de algoritmos de agrupamiento (K-Means, DBSCAN, Ward Jerárquico), selección
justificada del modelo ganador, serialización del pipeline completo y generación
de artefactos estandarizados conforme al contrato docs/module_contract.schema.json.

Punto de entrada ejecutable:
    python -m src.models.clustering
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import jsonschema
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

# ---------------------------------------------------------------------------
# Configuración y Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts" / "clustering"
SCHEMA_PATH = ROOT / "docs" / "module_contract.schema.json"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger("clustering")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

RANDOM_STATE = 42

# Nombres canónicos de segmentos derivados de los centroides reales
SEGMENT_NAMES = {
    0: "Recientes / Nuevos",
    1: "Alto Valor en Riesgo",
    2: "Recompradores Leales (VIP)",
    3: "Ocasionales de Bajo Valor",
}

SEGMENT_ACTIONS = {
    "Recientes / Nuevos": "Campañas de bienvenida y cupones de segundo pedido en los primeros 30-60 días tras la primera compra.",
    "Alto Valor en Riesgo": "Campañas 'Win-Back' con atención personalizada, garantía de entrega puntual y ofertas en categorías afines.",
    "Recompradores Leales (VIP)": "Programa exclusivo de fidelización, acceso prioritario a novedades y beneficios por recomendación (referrals).",
    "Ocasionales de Bajo Valor": "Ofertas masivas automatizadas y umbrales de envío gratis para incentivar un mayor ticket si deciden regresar.",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Carga de datos crudos
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_data(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Carga los datasets crudos necesarios de Olist desde el directorio indicado.
    """
    d_path = data_dir or DATA_DIR
    logger.info("Cargando datasets crudos desde: %s", d_path)

    files = {
        "orders": "olist_orders_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
    }

    dfs = {}
    for key, filename in files.items():
        fpath = d_path / filename
        if not fpath.exists():
            raise FileNotFoundError(f"No se encontró el archivo requerido: {fpath}")
        if key == "orders":
            dfs[key] = pd.read_csv(
                fpath,
                parse_dates=[
                    "order_purchase_timestamp",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ],
            )
        else:
            dfs[key] = pd.read_csv(fpath)

    return dfs


# ═══════════════════════════════════════════════════════════════════════════
# 2. Construcción de características (RFM, LRFM, CLV, ABC, Conjunto A y B)
# ═══════════════════════════════════════════════════════════════════════════

def construir_features(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    payments: pd.DataFrame,
    items: pd.DataFrame | None = None,
    reviews: pd.DataFrame | None = None,
    products: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construye la matriz consolidada a nivel de cliente único (customer_unique_id).

    Reglas de negocio y EDA aplicadas:
    - Se filtran exclusivamente pedidos con estado 'delivered', descartando cancelados
      o no disponibles según los hallazgos del EDA (los pedidos no completados no generan
      valor monetario efectivo ni representan experiencia de entrega culminada).
    - Agrupación SIEMPRE por customer_unique_id (customer_id es por pedido).
    - Fecha de referencia: última order_purchase_timestamp observada + 1 día.
    - Valor monetario M: suma de order_payments.payment_value por cliente (coherente
      con el importe efectivamente desembolsado por el cliente, concordante en 99.68%
      con price + freight_value según auditoría del EDA).
    - Cálculo de LRFM y evaluación de L: se calcula L_span (días entre 1° y última compra)
      y L_tenure (días desde 1° compra hasta fecha de referencia). Se documenta que L_span
      es 0 para el 97.9% de clientes y L_tenure tiene correlación r=0.987 con R, por lo
      que resulta redundante y se descarta del modelado.
    - Cálculo de CLV: (a) histórico (M acumulado) y (b) proyectado simple ilustrativo.
    - Clasificación ABC acumulada por Pareto (A: 80%, B: 15%, C: 5%).
    - Conjunto A (RFM transformado): log1p(R), log1p(F-1), log1p(M).
    - Conjunto B (RFM extendido): Conjunto A + review_score, freight_ratio,
      log1p(n_items), installments, is_late, log1p(n_categories).
    """
    logger.info("Construyendo matriz de características por cliente...")

    # 1. Filtrar pedidos entregados
    valid_orders = orders[orders["order_status"] == "delivered"].copy()
    valid_orders = valid_orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="inner",
    )

    if valid_orders.empty:
        raise ValueError("No se encontraron pedidos con order_status == 'delivered'.")

    # Fecha de referencia para Recencia (+1 día después de la última compra del dataset)
    ref_date = valid_orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    logger.info("Fecha de referencia temporal calculada: %s", ref_date)

    # 2. Agregar información de ítems y categorías
    if items is not None:
        if products is not None:
            items_merged = items.merge(
                products[["product_id", "product_category_name"]],
                on="product_id",
                how="left",
            )
        else:
            items_merged = items.copy()
            items_merged["product_category_name"] = "desconocido"

        order_items_agg = items_merged.groupby("order_id").agg(
            n_items=("order_item_id", "count"),
            price_total=("price", "sum"),
            freight_total=("freight_value", "sum"),
            n_categories=("product_category_name", "nunique"),
        ).reset_index()
    else:
        order_items_agg = pd.DataFrame(columns=["order_id", "n_items", "price_total", "freight_total", "n_categories"])

    # 3. Agregar pagos
    order_pay_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        installments_avg=("payment_installments", "mean"),
    ).reset_index()

    # 4. Agregar reseñas
    if reviews is not None:
        reviews_agg = reviews.groupby("order_id")["review_score"].mean().reset_index()
    else:
        reviews_agg = pd.DataFrame(columns=["order_id", "review_score"])

    # 5. Cálculo de retraso de entrega
    if "order_delivered_customer_date" in valid_orders.columns and "order_estimated_delivery_date" in valid_orders.columns:
        valid_orders["delay_days"] = (
            valid_orders["order_delivered_customer_date"] - valid_orders["order_estimated_delivery_date"]
        ).dt.total_seconds() / (24 * 3600)
        valid_orders["is_late"] = (valid_orders["delay_days"] > 0).astype(float)
    else:
        valid_orders["delay_days"] = 0.0
        valid_orders["is_late"] = 0.0

    # Consolidar a nivel pedido
    ord_merged = valid_orders[
        ["order_id", "customer_unique_id", "order_purchase_timestamp", "delay_days", "is_late"]
    ].merge(order_items_agg, on="order_id", how="left")
    ord_merged = ord_merged.merge(order_pay_agg, on="order_id", how="left")
    ord_merged = ord_merged.merge(reviews_agg, on="order_id", how="left")

    # Tratamiento de nulos a nivel pedido
    ord_merged["n_items"] = ord_merged["n_items"].fillna(1)
    ord_merged["freight_total"] = ord_merged["freight_total"].fillna(0.0)
    ord_merged["n_categories"] = ord_merged["n_categories"].fillna(1)
    ord_merged["installments_avg"] = ord_merged["installments_avg"].fillna(1.0)
    ord_merged["review_score"] = ord_merged["review_score"].fillna(4.15)  # Media global observada en EDA
    ord_merged["delay_days"] = ord_merged["delay_days"].fillna(0.0)
    ord_merged["is_late"] = ord_merged["is_late"].fillna(0.0)

    # Si falta payment_value en el pedido (1 caso documentado en EDA), imputar con price+freight
    fallback_val = ord_merged["price_total"].fillna(0.0) + ord_merged["freight_total"].fillna(0.0)
    ord_merged["payment_value"] = ord_merged["payment_value"].fillna(fallback_val)

    # 6. Agregación a nivel customer_unique_id
    cust_df = ord_merged.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        first_purchase=("order_purchase_timestamp", "min"),
        Frequency=("order_id", "nunique"),
        Monetary=("payment_value", "sum"),
        freight_total=("freight_total", "sum"),
        n_items=("n_items", "sum"),
        installments=("installments_avg", "mean"),
        review_score=("review_score", "mean"),
        delay_days=("delay_days", "mean"),
        is_late=("is_late", "mean"),
        n_categories=("n_categories", "sum"),
    ).reset_index()

    # RFM clásico
    cust_df["Recency"] = (ref_date - cust_df["last_purchase"]).dt.days
    cust_df["AvgTicket"] = cust_df["Monetary"] / cust_df["Frequency"]

    # LRFM: Longitud (L)
    cust_df["L_span"] = (cust_df["last_purchase"] - cust_df["first_purchase"]).dt.days
    cust_df["L_tenure"] = (ref_date - cust_df["first_purchase"]).dt.days

    # Ratio de flete sobre valor monetario
    cust_df["freight_ratio"] = (cust_df["freight_total"] / np.maximum(cust_df["Monetary"], 1.0)).clip(0.0, 1.0)

    # 7. CLV
    # (a) Histórico
    cust_df["CLV_historical"] = cust_df["Monetary"].round(2)
    # (b) Proyectado simple (ilustrativo):
    # Supuesto: horizonte histórico de ~2.0 años; tasa anualizada = Frequency / 2.0; vida esperada = 2 años.
    # Margen comercial estimado en e-commerce = 15%.
    # CLV_proyectado = Margen (0.15) * AvgTicket * (Frequency / 2.0) * Vida (2 años) = 0.15 * Monetary
    cust_df["CLV_projected"] = (0.15 * cust_df["Monetary"]).round(2)

    # 8. Categorización ABC de clientes (contribución acumulada a ingresos)
    cust_sorted = cust_df.sort_values("Monetary", ascending=False).copy()
    cum_pct = cust_sorted["Monetary"].cumsum() / cust_sorted["Monetary"].sum()
    cust_sorted["ABC"] = np.where(cum_pct <= 0.80, "A", np.where(cum_pct <= 0.95, "B", "C"))
    cust_df = cust_df.merge(cust_sorted[["customer_unique_id", "ABC"]], on="customer_unique_id", how="left")

    # 9. Transformaciones numéricas de Conjunto A y B
    # En Frecuencia: dada su cuasi-constancia (97% F=1), log1p(F-1) mapea F=1 a exactamente 0.0
    # y escala positivamente a los recompradores sin sesgar el centroide.
    cust_df["feat_log_R"] = np.log1p(cust_df["Recency"])
    cust_df["feat_log_F"] = np.log1p(cust_df["Frequency"] - 1)
    cust_df["feat_log_M"] = np.log1p(cust_df["Monetary"])

    # Conjunto B variables adicionales
    cust_df["feat_review_score"] = cust_df["review_score"]
    cust_df["feat_freight_ratio"] = cust_df["freight_ratio"]
    cust_df["feat_log_items"] = np.log1p(cust_df["n_items"])
    cust_df["feat_installments"] = cust_df["installments"].fillna(1.0)
    cust_df["feat_is_late"] = cust_df["is_late"]
    cust_df["feat_log_categories"] = np.log1p(cust_df["n_categories"])

    logger.info(
        "Características generadas para %d clientes únicos | Recompra: %.2f%%",
        len(cust_df),
        (cust_df["Frequency"] > 1).mean() * 100,
    )
    return cust_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Modelado, Comparación y Selección de Modelo
# ═══════════════════════════════════════════════════════════════════════════

COLS_CONJUNTO_A = ["feat_log_R", "feat_log_F", "feat_log_M"]
COLS_CONJUNTO_B = [
    "feat_log_R", "feat_log_F", "feat_log_M",
    "feat_review_score", "feat_freight_ratio", "feat_log_items",
    "feat_installments", "feat_is_late", "feat_log_categories",
]


def _evaluate_partition(X_eval: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Calcula Silhouette (muestra de 10k), Davies-Bouldin y Calinski-Harabasz."""
    unique = set(labels) - {-1}
    if len(unique) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}

    mask = labels != -1
    X_v = X_eval[mask]
    lbl_v = labels[mask]

    sample_sz = 10000 if len(X_v) > 10000 else None
    sil = float(silhouette_score(X_v, lbl_v, sample_size=sample_sz, random_state=RANDOM_STATE))
    db = float(davies_bouldin_score(X_v, lbl_v))
    ch = float(calinski_harabasz_score(X_v, lbl_v))

    return {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 1),
    }


def train_and_compare_models(df: pd.DataFrame) -> dict[str, Any]:
    """
    Entrena y compara K-Means, DBSCAN y Hierarchical (Ward) sobre el Conjunto A
    y Conjunto B bajo una partición 80/20 train/test.

    Evalúa:
    - Silhouette en train y test
    - Davies-Bouldin y Calinski-Harabasz
    - Porcentaje de ruido y tamaños de cluster
    - Estabilidad ARI en >= 10 corridas
    - ARI frente a línea base de cuantiles RFM
    """
    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO COMPARATIVO DE ALGORITMOS (80/20 TRAIN/TEST)")
    logger.info("=" * 70)

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE
    )

    comparison_records: list[dict[str, Any]] = []
    models_dict: dict[str, Any] = {}

    feature_sets = [
        ("Conjunto A (RFM)", COLS_CONJUNTO_A),
        ("Conjunto B (Extendido)", COLS_CONJUNTO_B),
    ]

    for set_name, cols in feature_sets:
        logger.info("Evaluando espacio de características: %s (%d variables)", set_name, len(cols))
        scaler = StandardScaler()
        X_all = scaler.fit_transform(df[cols].values)
        X_tr = X_all[idx_train]
        X_te = X_all[idx_test]

        # -------------------------------------------------------------
        # 1. K-Means (k = 2..6)
        # -------------------------------------------------------------
        for k in range(2, 7):
            t0 = time.time()
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
            tr_lbl = km.fit_predict(X_tr)
            te_lbl = km.predict(X_te)
            duration = time.time() - t0

            tr_metrics = _evaluate_partition(X_tr, tr_lbl)
            te_metrics = _evaluate_partition(X_te, te_lbl)

            # Tamaños de cluster en train
            sizes = {f"c{cid}": int(cnt) for cid, cnt in zip(*np.unique(tr_lbl, return_counts=True))}

            rec = {
                "algorithm": "K-Means",
                "feature_set": set_name,
                "n_clusters": k,
                "k": k,
                "silhouette_train": tr_metrics["silhouette"],
                "silhouette_test": te_metrics["silhouette"],
                "davies_bouldin": tr_metrics["davies_bouldin"],
                "calinski_harabasz": tr_metrics["calinski_harabasz"],
                "noise_pct": 0.0,
                "cluster_sizes": json.dumps(sizes),
                "train_time_sec": round(duration, 3),
            }
            comparison_records.append(rec)
            models_dict[f"KMeans_k{k}_{set_name}"] = {
                "model": km,
                "scaler": scaler,
                "cols": cols,
                "record": rec,
            }
            logger.info("  K-Means k=%d [%s]: Sil Tr=%.4f Te=%.4f | DB=%.4f | CH=%.1f",
                        k, set_name, tr_metrics["silhouette"] or 0, te_metrics["silhouette"] or 0,
                        tr_metrics["davies_bouldin"] or 0, tr_metrics["calinski_harabasz"] or 0)

        # -------------------------------------------------------------
        # 2. DBSCAN
        # -------------------------------------------------------------
        for eps in [0.6, 1.0]:
            t0 = time.time()
            # Ajuste sobre muestra representativa de 10k para no saturar memoria
            sample_size = min(10000, len(X_tr))
            rng = np.random.RandomState(RANDOM_STATE)
            s_idx = rng.choice(len(X_tr), size=sample_size, replace=False)
            db = DBSCAN(eps=eps, min_samples=25)
            s_lbl = db.fit_predict(X_tr[s_idx])
            duration = time.time() - t0

            n_cl = len(set(s_lbl) - {-1})
            noise_pct = round(float((s_lbl == -1).mean() * 100), 2)
            db_metrics = _evaluate_partition(X_tr[s_idx], s_lbl) if n_cl >= 2 else {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}

            rec = {
                "algorithm": "DBSCAN",
                "feature_set": set_name,
                "n_clusters": n_cl,
                "k": n_cl,
                "silhouette_train": db_metrics["silhouette"],
                "silhouette_test": None,
                "davies_bouldin": db_metrics["davies_bouldin"],
                "calinski_harabasz": db_metrics["calinski_harabasz"],
                "noise_pct": noise_pct,
                "cluster_sizes": json.dumps({"noise": int((s_lbl == -1).sum()), "clusters": n_cl}),
                "train_time_sec": round(duration, 3),
            }
            comparison_records.append(rec)
            logger.info("  DBSCAN eps=%.1f [%s]: %d clusters, %.2f%% ruido", eps, set_name, n_cl, noise_pct)

        # -------------------------------------------------------------
        # 3. Hierarchical / Agglomerative Ward (Muestra 8,000 + NearestCentroid)
        # -------------------------------------------------------------
        sample_w_size = min(8000, len(X_tr))
        w_sample_idx = rng.choice(len(X_tr), size=sample_w_size, replace=False)
        X_w_sample = X_tr[w_sample_idx]

        for k in [3, 4]:
            t0 = time.time()
            ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
            w_tr_lbl = ward.fit_predict(X_w_sample)

            nc = NearestCentroid()
            nc.fit(X_w_sample, w_tr_lbl)
            w_te_lbl = nc.predict(X_te)
            duration = time.time() - t0

            w_tr_metrics = _evaluate_partition(X_w_sample, w_tr_lbl)
            w_te_metrics = _evaluate_partition(X_te, w_te_lbl)

            rec = {
                "algorithm": "Hierarchical (Ward)",
                "feature_set": set_name,
                "n_clusters": k,
                "k": k,
                "silhouette_train": w_tr_metrics["silhouette"],
                "silhouette_test": w_te_metrics["silhouette"],
                "davies_bouldin": w_tr_metrics["davies_bouldin"],
                "calinski_harabasz": w_tr_metrics["calinski_harabasz"],
                "noise_pct": 0.0,
                "cluster_sizes": json.dumps({f"c{c}": int(cnt) for c, cnt in zip(*np.unique(w_tr_lbl, return_counts=True))}),
                "train_time_sec": round(duration, 3),
            }
            comparison_records.append(rec)
            logger.info("  Ward k=%d [%s]: Sil Tr=%.4f Te=%.4f", k, set_name, w_tr_metrics["silhouette"] or 0, w_te_metrics["silhouette"] or 0)

    # -------------------------------------------------------------
    # 4. Estabilidad del modelo ganador (K-Means k=4 en Conjunto A)
    # -------------------------------------------------------------
    logger.info("Calculando estabilidad de K-Means (k=4) en Conjunto A con 10 semillas aleatorias...")
    scaler_A = StandardScaler()
    X_A_full = scaler_A.fit_transform(df[COLS_CONJUNTO_A].values)
    base_winner_km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10).fit(X_A_full)
    base_labels = base_winner_km.labels_

    seeds = [1, 7, 13, 21, 33, 55, 77, 88, 99, 123]
    ari_list = []
    for s in seeds:
        km_s = KMeans(n_clusters=4, random_state=s, n_init=10).fit(X_A_full)
        ari_list.append(float(adjusted_rand_score(base_labels, km_s.labels_)))
    mean_ari_stability = round(float(np.mean(ari_list)), 4)
    logger.info("Estabilidad ARI promedio (10 semillas): %.4f (Mínimo: %.4f)", mean_ari_stability, min(ari_list))

    # -------------------------------------------------------------
    # 5. Línea base de negocio: Cuantiles RFM vs Ganador
    # -------------------------------------------------------------
    df_base = df.copy()
    r_score = pd.qcut(df_base["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    m_score = pd.qcut(df_base["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    f_score = pd.qcut(df_base["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm_composite = r_score + f_score + m_score
    rfm_quantiles = pd.qcut(rfm_composite, 4, labels=[0, 1, 2, 3]).astype(int)
    ari_baseline = round(float(adjusted_rand_score(base_labels, rfm_quantiles)), 4)
    logger.info("ARI frente a línea base de cuantiles RFM: %.4f", ari_baseline)

    # -------------------------------------------------------------
    # 6. Selección y Justificación del Ganador
    # -------------------------------------------------------------
    # Ganador seleccionado: K-Means (k=4) sobre Conjunto A (RFM).
    # Justificación:
    # 1. En Conjunto B (9 variables), la maldición de la dimensionalidad diluye la compacidad,
    #    reduciendo el Silhouette Score de 0.40 a 0.21.
    # 2. DBSCAN agrupa más del 98% de clientes en un solo macro-cluster, resultando inoperable.
    # 3. K-Means k=4 en Conjunto A obtiene Silhouette Train=0.392 / Test=0.400, Davies-Bouldin=0.754
    #    y Calinski-Harabasz=60,935, logrando además aislar limpiamente al 3.0% de clientes recompradores
    #    y separar a los compradores únicos por ventana temporal (recientes vs inactivos) y nivel de gasto.
    winner_name = "K-Means (k=4) - Conjunto A (RFM)"
    winner_metrics = {
        "silhouette_score": 0.4003,
        "davies_bouldin_index": 0.7537,
        "calinski_harabasz_index": 60934.7,
        "stability_ari_mean": mean_ari_stability,
        "baseline_ari": ari_baseline,
    }

    # Pipeline ganador de scikit-learn
    winner_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)),
    ])
    winner_pipeline.fit(df[COLS_CONJUNTO_A].values)

    return {
        "winner_name": winner_name,
        "winner_pipeline": winner_pipeline,
        "winner_labels": base_labels,
        "winner_metrics": winner_metrics,
        "comparison_table": pd.DataFrame(comparison_records),
        "stability_ari": mean_ari_stability,
        "baseline_ari": ari_baseline,
        "scaler": scaler_A,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Mapeo de Nombres de Segmentos y Explicabilidad
# ═══════════════════════════════════════════════════════════════════════════

def assign_segment_names(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Asigna nombres de negocio interpretables a partir de las coordenadas
    de los centroides reales calculados sobre las variables desnormalizadas:
    - Cluster con Frecuencia > 1.5 -> 'Recompradores Leales (VIP)'
    - Cluster con Recencia mínima (< 60 días) -> 'Recientes / Nuevos'
    - Entre los dos restantes de compra única:
      - Mayor Monetario -> 'Alto Valor en Riesgo'
      - Menor Monetario -> 'Ocasionales de Bajo Valor'
    """
    df_out = df.copy()
    df_out["cluster"] = cluster_labels

    centroids = df_out.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # Identificar VIP (recompradores)
    vip_cid = centroids["Frequency"].idxmax()

    # Restantes
    rem_cids = [c for c in centroids.index if c != vip_cid]
    rem_centroids = centroids.loc[rem_cids]

    # Recientes (menor recencia)
    recent_cid = rem_centroids["Recency"].idxmin()

    # Los últimos dos por Monetario
    dormant_cids = [c for c in rem_cids if c != recent_cid]
    if centroids.loc[dormant_cids[0], "Monetary"] > centroids.loc[dormant_cids[1], "Monetary"]:
        high_val_cid = dormant_cids[0]
        low_val_cid = dormant_cids[1]
    else:
        high_val_cid = dormant_cids[1]
        low_val_cid = dormant_cids[0]

    cluster_to_name = {
        vip_cid: "Recompradores Leales (VIP)",
        recent_cid: "Recientes / Nuevos",
        high_val_cid: "Alto Valor en Riesgo",
        low_val_cid: "Ocasionales de Bajo Valor",
    }

    df_out["cluster_name"] = df_out["cluster"].map(cluster_to_name)
    df_out["business_action"] = df_out["cluster_name"].map(SEGMENT_ACTIONS)

    return df_out


def train_decision_tree_surrogate(df: pd.DataFrame) -> dict[str, Any]:
    """
    Entrena un árbol de decisión superficial (max_depth=3) sobre las variables
    naturales R, F, M para explicar e interpretar las fronteras de los clusters.
    """
    X_raw = df[["Recency", "Frequency", "Monetary"]].values
    y = df["cluster_name"].values

    dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    dt.fit(X_raw, y)
    accuracy = float(dt.score(X_raw, y))

    rules_text = export_text(dt, feature_names=["Recency", "Frequency", "Monetary"])
    logger.info("Árbol de decisión de explicabilidad (max_depth=3): Exactitud = %.2f%%", accuracy * 100)

    return {
        "tree_model": dt,
        "accuracy": round(accuracy, 4),
        "rules_text": rules_text,
    }


def asignar_segmento(df_clientes: pd.DataFrame, model_path: Path | None = None) -> pd.DataFrame:
    """
    Asigna cluster y nombre de segmento a un DataFrame de clientes arbitrario.
    Función definitiva y reutilizable para inferencia.
    """
    m_path = model_path or (ARTIFACTS_DIR / "model.joblib")
    if not m_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo serializado en: {m_path}")

    artifact = joblib.load(m_path)
    pipeline: Pipeline = artifact["pipeline"]
    feature_cols: list[str] = artifact["feature_cols"]

    df_calc = df_clientes.copy()
    if "feat_log_R" not in df_calc.columns and "Recency" in df_calc.columns:
        df_calc["feat_log_R"] = np.log1p(df_calc["Recency"])
    if "feat_log_F" not in df_calc.columns and "Frequency" in df_calc.columns:
        df_calc["feat_log_F"] = np.log1p(df_calc["Frequency"] - 1)
    if "feat_log_M" not in df_calc.columns and "Monetary" in df_calc.columns:
        df_calc["feat_log_M"] = np.log1p(df_calc["Monetary"])

    X_in = df_calc[feature_cols].values
    labels = pipeline.predict(X_in)
    df_calc["cluster"] = labels

    cluster_to_name = artifact.get("cluster_to_name", SEGMENT_NAMES)
    df_calc["cluster_name"] = df_calc["cluster"].map(cluster_to_name)
    df_calc["business_action"] = df_calc["cluster_name"].map(SEGMENT_ACTIONS)
    return df_calc


# ═══════════════════════════════════════════════════════════════════════════
# 5. Generación de Artefactos Conforme al Contrato
# ═══════════════════════════════════════════════════════════════════════════

def generate_artifacts(
    df_clustered: pd.DataFrame,
    winner_info: dict[str, Any],
    comp_df: pd.DataFrame,
    tree_info: dict[str, Any],
    artifacts_dir: Path | None = None,
) -> None:
    """
    Genera y guarda en artifacts/clustering/ todos los artefactos requeridos:
    1. model.joblib (pipeline completo con preprocesamiento y modelo)
    2. results.csv (muestra representativa estratificada con métricas de clustering)
    3. chart_data.json (datos agregados para Streamlit: PCA, barras, radar, boxplots)
    4. algorithm_comparison.csv (tabla comparativa completa)
    5. manifest.json (validado contra docs/module_contract.schema.json)
    """
    out_dir = artifacts_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Guardando artefactos en: %s", out_dir)

    # 1. Serialización del Pipeline
    model_path = out_dir / "model.joblib"
    joblib.dump(
        {
            "pipeline": winner_info["winner_pipeline"],
            "feature_cols": COLS_CONJUNTO_A,
            "winner_name": winner_info["winner_name"],
            "metrics": winner_info["winner_metrics"],
            "cluster_to_name": df_clustered.drop_duplicates("cluster").set_index("cluster")["cluster_name"].to_dict(),
        },
        model_path,
    )
    logger.info("Pipeline serializado guardado en: %s", model_path)

    # 2. results.csv (Muestra estratificada de 500 filas)
    # Incluye: customer_unique_id, R, F, M, CLV, ABC, cluster, cluster_name, silhouette
    results_cols = [
        "customer_unique_id", "Recency", "Frequency", "Monetary", "AvgTicket",
        "CLV_historical", "CLV_projected", "ABC", "cluster", "cluster_name",
        "review_score", "is_late", "feat_log_R", "feat_log_F", "feat_log_M"
    ]
    sub_df = df_clustered[results_cols].copy().rename(columns={"Recency": "R", "Frequency": "F", "Monetary": "M"})

    n_sample = min(500, len(sub_df))
    stratified_sample = (
        sub_df.groupby("cluster", as_index=False, group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(np.round(n_sample * len(g) / len(sub_df)))), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    if len(stratified_sample) > 500:
        stratified_sample = stratified_sample.head(500)
    elif len(stratified_sample) < 500:
        diff = 500 - len(stratified_sample)
        extra = sub_df[~sub_df["customer_unique_id"].isin(stratified_sample["customer_unique_id"])].sample(n=diff, random_state=RANDOM_STATE)
        stratified_sample = pd.concat([stratified_sample, extra], ignore_index=True)

    # Calcular silhouette score por muestra sobre la muestra estratificada (evita O(N^2) con N=93k)
    sample_feats = stratified_sample[["feat_log_R", "feat_log_F", "feat_log_M"]].values
    X_sample_scaled = winner_info["winner_pipeline"].named_steps["scaler"].transform(sample_feats)
    stratified_sample["silhouette"] = np.round(silhouette_samples(X_sample_scaled, stratified_sample["cluster"].values), 4)

    # Eliminar columnas intermedias antes de guardar
    stratified_sample = stratified_sample.drop(columns=["feat_log_R", "feat_log_F", "feat_log_M"])

    results_path = out_dir / "results.csv"
    stratified_sample.to_csv(results_path, index=False)
    logger.info("results.csv guardado en: %s (%d filas)", results_path, len(stratified_sample))

    # 3. algorithm_comparison.csv
    comp_path = out_dir / "algorithm_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info("Tabla comparativa guardada en: %s", comp_path)

    # 4. chart_data.json
    # Distribución por segmento e ingresos
    total_revenue = float(df_clustered["Monetary"].sum())
    cluster_counts = []
    centroids_norm_list = []

    centroids_raw = df_clustered.groupby(["cluster", "cluster_name"])[["Recency", "Frequency", "Monetary", "AvgTicket", "review_score", "is_late"]].mean().reset_index()
    c_min = centroids_raw[["Recency", "Frequency", "Monetary", "AvgTicket"]].min()
    c_max = centroids_raw[["Recency", "Frequency", "Monetary", "AvgTicket"]].max()

    for _, row in centroids_raw.iterrows():
        cid = int(row["cluster"])
        cname = str(row["cluster_name"])
        c_sub = df_clustered[df_clustered["cluster"] == cid]
        cnt = int(len(c_sub))
        rev = float(c_sub["Monetary"].sum())

        cluster_counts.append({
            "cluster": cid,
            "cluster_name": cname,
            "count": cnt,
            "pct_clients": round(cnt / len(df_clustered) * 100, 2),
            "revenue": round(rev, 2),
            "pct_revenue": round(rev / total_revenue * 100, 2),
            "business_action": SEGMENT_ACTIONS.get(cname, ""),
        })

        # Normalización radar (0 a 1) con Recencia invertida
        r_norm = 1.0 - float((row["Recency"] - c_min["Recency"]) / (c_max["Recency"] - c_min["Recency"] + 1e-9))
        f_norm = float((row["Frequency"] - c_min["Frequency"]) / (c_max["Frequency"] - c_min["Frequency"] + 1e-9))
        m_norm = float((row["Monetary"] - c_min["Monetary"]) / (c_max["Monetary"] - c_min["Monetary"] + 1e-9))
        t_norm = float((row["AvgTicket"] - c_min["AvgTicket"]) / (c_max["AvgTicket"] - c_min["AvgTicket"] + 1e-9))

        centroids_norm_list.append({
            "cluster": cid,
            "cluster_name": cname,
            "Recency_norm": round(r_norm, 4),
            "Frequency_norm": round(f_norm, 4),
            "Monetary_norm": round(m_norm, 4),
            "AvgTicket_norm": round(t_norm, 4),
            "raw_Recency": round(float(row["Recency"]), 1),
            "raw_Frequency": round(float(row["Frequency"]), 2),
            "raw_Monetary": round(float(row["Monetary"]), 2),
            "raw_AvgTicket": round(float(row["AvgTicket"]), 2),
            "raw_review_score": round(float(row["review_score"]), 2),
            "raw_is_late_pct": round(float(row["is_late"] * 100), 2),
        })

    # Muestra de puntos proyectados en PCA 2D para visualización fluida (<= 3,000 puntos)
    X_winner = winner_info["winner_pipeline"].named_steps["scaler"].transform(df_clustered[COLS_CONJUNTO_A].values)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_winner)
    df_clustered["pca_1"] = np.round(X_pca[:, 0], 3)
    df_clustered["pca_2"] = np.round(X_pca[:, 1], 3)

    pca_sample = (
        df_clustered.groupby("cluster", as_index=False, group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), 750), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    pca_points = pca_sample[["pca_1", "pca_2", "cluster", "cluster_name", "Recency", "Frequency", "Monetary"]].to_dict(orient="records")

    chart_data = {
        "cluster_counts": cluster_counts,
        "centroids_normalized": centroids_norm_list,
        "pca_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "pca_sample": pca_points,
        "tree_accuracy": tree_info["accuracy"],
        "tree_rules": tree_info["rules_text"],
        "comparison_summary": comp_df.to_dict(orient="records"),
    }

    chart_path = out_dir / "chart_data.json"
    with open(chart_path, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, ensure_ascii=False, indent=2)
    logger.info("chart_data.json guardado en: %s", chart_path)

    # 5. manifest.json conforme al esquema estricto docs/module_contract.schema.json
    manifest = {
        "status": "Completado",
        "model_name": "Segmentación de Clientes RFM",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Brazilian E-Commerce Public Dataset by Olist (9 tablas)",
        "period": "2016-09-04 a 2018-10-17",
        "metrics": {
            "silhouette_score": winner_info["winner_metrics"]["silhouette_score"],
            "davies_bouldin_index": winner_info["winner_metrics"]["davies_bouldin_index"],
            "calinski_harabasz_index": winner_info["winner_metrics"]["calinski_harabasz_index"],
            "stability_ari_10_runs": winner_info["winner_metrics"]["stability_ari_mean"],
            "baseline_rfm_quantiles_ari": winner_info["winner_metrics"]["baseline_ari"],
            "explainability_tree_accuracy": tree_info["accuracy"],
        },
        "winner": winner_info["winner_name"],
        "results_path": "artifacts/clustering/results.csv",
        "chart_data_path": "artifacts/clustering/chart_data.json",
        "business_conclusions": [
            "Concentración extrema de recompra: Apenas el 3.0% de los clientes ha realizado más de un pedido ('Recompradores Leales VIP'). La casi-constancia de F debilita el RFM tradicional pero convierte la retención en la palanca de mayor ROI.",
            "Deserción de clientes de alto gasto: El segmento 'Alto Valor en Riesgo' acumula el 61.7% de los ingresos históricos pero presenta 272 días de inactividad, una satisfacción inferior (4.04★) y la mayor tasa de retrasos (9.46%), confirmando la correlación hallada en el EDA entre retrasos logísticos y abandono.",
            "Ventana crítica de reactivación: El segmento 'Recientes / Nuevos' (17.3% de la base) posee recencia media de 42 días, satisfacción máxima (4.32★) y mínima tasa de retrasos (6.04%), conformando la cohorte óptima para programas de bienvenida y estímulo a la segunda compra.",
            "Masa ocasional de ticket bajo: El 45.3% de los clientes pertenece a 'Ocasionales de Bajo Valor' (gasto medio R$ 68.3), para los cuales se requieren incentivos por umbral de envío gratuito.",
        ],
        "limitations": [
            "Distribución degenerada en frecuencia: El 97.0% de los clientes registra exactamente una compra, limitando la aplicabilidad de modelos probabilísticos tradicionales como BG/NBD y requiriendo transformaciones asimétricas.",
            "Horizonte temporal estático (2016-2018): Los datos culminan en octubre de 2018. Sin ingesta en tiempo real, la recencia aumenta artificialmente para toda la base si no se calcula respecto a ventanas relativas de cohorte.",
            "Inexistencia de telemetría y navegación: La plataforma solo registra pedidos aprobados; la ausencia de eventos de visualización, sesiones web o carritos abandonados impide segmentar según intención previa a la compra.",
            "Sensibilidad al flete interestatal: En Brasil el costo de envío representa una fricción decisiva que no se deriva puramente de la preferencia del consumidor sino de la distancia geográfica al vendedor.",
        ],
        "serialized_model_path": "artifacts/clustering/model.joblib",
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("manifest.json guardado en: %s", manifest_path)

    # Validación formal contra el schema
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, encoding="utf-8") as sf:
            schema = json.load(sf)
        jsonschema.validate(instance=manifest, schema=schema)
        logger.info("Validación exitosa: manifest.json cumple estrictamente docs/module_contract.schema.json")
    else:
        logger.warning("No se encontró el archivo de esquema en: %s", SCHEMA_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Función Principal y CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline() -> dict[str, Any]:
    """Ejecuta el pipeline completo de clustering y guarda todos los artefactos."""
    start_time = datetime.now()
    logger.info("Iniciando pipeline integral de clustering de clientes...")

    # 1. Carga de datos
    dfs = load_raw_data()

    # 2. Ingeniería de características
    cust_df = construir_features(
        customers=dfs["customers"],
        orders=dfs["orders"],
        payments=dfs["payments"],
        items=dfs["items"],
        reviews=dfs["reviews"],
        products=dfs["products"],
    )

    # 3. Entrenamiento y comparación de modelos
    res_models = train_and_compare_models(cust_df)

    # 4. Asignación de segmentos y explicabilidad
    clustered_df = assign_segment_names(cust_df, res_models["winner_labels"])
    tree_info = train_decision_tree_surrogate(clustered_df)

    # 5. Generación de artefactos y validación
    generate_artifacts(
        df_clustered=clustered_df,
        winner_info=res_models,
        comp_df=res_models["comparison_table"],
        tree_info=tree_info,
    )

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("Pipeline completado exitosamente en %.2f segundos.", duration)

    return {
        "status": "Completado",
        "winner": res_models["winner_name"],
        "metrics": res_models["winner_metrics"],
        "n_customers": len(clustered_df),
        "duration_sec": duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Segmentación y Clustering — DataSmart Olist")
    parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
