"""
Módulo productivo de Clasificación: riesgo de reseña negativa — DataSmart Olist.
=================================================================================
Responsable: Módulo de Clasificación (rama feature/classification)
Proyecto Integrador · Minería de Datos · UNMSM FISI 2026

Pregunta de negocio
-------------------
¿Qué pedidos tienen riesgo de recibir una reseña negativa (1–2 estrellas)?

Objetivo: ``review_negativa = 1`` si ``review_score <= 2``, ``0`` en otro caso.

Momento de predicción
---------------------
El modelo se aplica cuando Olist envía la encuesta de satisfacción
(``review_creation_date``), antes de que el cliente responda. Solo entran
variables observables en ese instante: el pedido tal como se compró y la
ejecución logística hasta ese momento (despacho, entrega, retraso). Nada
posterior a la encuesta —el texto de la reseña, por ejemplo— se usa.

Por qué esta evaluación
-----------------------
- Las clases están desbalanceadas (≈ 1 de cada 7 pedidos es negativo), así que la
  métrica principal es **PR-AUC**: mide cuán bien se detecta la clase negativa
  siendo minoritaria. El azar vale la prevalencia (≈ 0.11 en prueba).
- La partición es **temporal**: se entrena con compras de enero 2017 a mayo 2018
  y se prueba con junio a agosto 2018, que el modelo nunca vio. Una partición
  aleatoria sería optimista porque la tasa de negativas varía entre meses.
- Entre modelos estadísticamente empatados (a menos de una desviación estándar
  del mejor en validación cruzada) se elige el más simple, por parsimonia.
- El umbral de decisión se calibra con predicciones out-of-fold del
  entrenamiento, nunca con el test.

El modelo desplegable es un ``Pipeline`` de scikit-learn (preprocesamiento +
clasificador) serializado en ``artifacts/classification/model.joblib`` junto con
el umbral y la lista de variables. La inferencia la hace ``backend/services``.

Punto de entrada ejecutable:
    python -m src.models.classification            # corrida completa
    python -m src.models.classification --quick    # corrida corta (pruebas)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts" / "classification"
SCHEMA_PATH = ROOT / "docs" / "module_contract.schema.json"

VERSION = "1.1.0"
RANDOM_STATE = 42
CORTE = pd.Timestamp("2018-06-01")
PERIODO_INICIO = "2017-01-01"
PERIODO_FIN = "2018-08-31 23:59:59"
DIA = pd.Timedelta(days=1)

logger = logging.getLogger("datasmart.classification")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
REGION = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}

NUM_COMPRA = [
    "dia_semana", "hora", "fin_de_semana", "horas_aprobacion", "dias_estimados", "dias_limite_despacho",
    "n_items", "n_productos", "n_sellers", "n_categorias", "price_total", "freight_total", "ratio_flete",
    "precio_por_item", "peso_total_g", "volumen_total_cm3", "fotos_media", "largo_descripcion_media",
    "n_cuotas", "n_medios_pago", "distancia_km", "mismo_estado",
]
NUM_ENCUESTA = ["pendiente_en_encuesta", "dias_despacho", "despacho_tardio", "dias_transporte",
                "dias_entrega", "dias_retraso", "entrega_tardia"]
CAT = ["payment_type", "customer_state", "region_vendedor", "categoria_principal"]
# Variables asimétricas y no negativas: log1p antes de escalar (ayuda a LR, SVM y NB; neutro para árboles)
LOG_COLS = ["horas_aprobacion", "price_total", "freight_total", "ratio_flete", "precio_por_item",
            "peso_total_g", "volumen_total_cm3", "largo_descripcion_media", "distancia_km"]

NUM = NUM_COMPRA + NUM_ENCUESTA
FEATURES = NUM + CAT
FEATURES_COMPRA = NUM_COMPRA + CAT
TARGET = "review_negativa"

# Orden fijo de complejidad para la regla de parsimonia (no depende del tiempo medido).
COMPLEJIDAD = ["Baseline (prior)", "Naive Bayes", "Regresión Logística", "SVM lineal",
               "HistGradientBoosting", "XGBoost", "Random Forest"]

RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_tr": "product_category_name_translation.csv",
}
DATE_COLS = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_raw(data_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Lee las nueve tablas de Olist desde ``data/raw``."""
    missing = [name for name in RAW_FILES.values() if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan CSV en {data_dir}: {', '.join(missing)}. "
            "Descárgalos como indica el README (kagglehub o work-at-olist-data)."
        )
    tables = {}
    for key, name in RAW_FILES.items():
        tables[key] = pd.read_csv(data_dir / name, encoding="utf-8-sig",
                                  parse_dates=DATE_COLS.get(key), low_memory=False)
    return tables


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Un registro por pedido con pagos, ítems, cliente, vendedor y reseña."""
    orders, customers, items = tables["orders"], tables["customers"], tables["items"]
    products, sellers, payments = tables["products"], tables["sellers"], tables["payments"]
    reviews, geo, cat_tr = tables["reviews"], tables["geolocation"], tables["category_tr"]

    # Reseñas: una por pedido (la última respondida), como en la preparación de datos
    rev = (reviews.sort_values("review_answer_timestamp")
           .drop_duplicates("order_id", keep="last")[["order_id", "review_score", "review_creation_date"]])

    # Centroide por prefijo de CEP, descartando coordenadas fuera de Brasil
    geo_ok = geo[geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -34)]
    coords = geo_ok.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    coords.columns = ["lat", "lng"]

    prod = products.merge(cat_tr, on="product_category_name", how="left")
    prod["categoria"] = (prod.product_category_name_english
                         .fillna(prod.product_category_name).fillna("sin_categoria"))
    prod["volumen_cm3"] = prod.product_length_cm * prod.product_height_cm * prod.product_width_cm

    it = (items.merge(prod[["product_id", "categoria", "product_weight_g", "volumen_cm3",
                            "product_photos_qty", "product_description_lenght"]], on="product_id", how="left")
          .merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left"))

    items_agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_productos=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_categorias=("categoria", "nunique"),
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        peso_total_g=("product_weight_g", "sum"),
        volumen_total_cm3=("volumen_cm3", "sum"),
        fotos_media=("product_photos_qty", "mean"),
        largo_descripcion_media=("product_description_lenght", "mean"),
        shipping_limit_date=("shipping_limit_date", "max"),
        categoria_principal=("categoria", "first"),
        seller_zip=("seller_zip_code_prefix", "first"),
        seller_state=("seller_state", "first"),
    ).reset_index()
    items_agg["seller_lat"] = items_agg.seller_zip.map(coords.lat)
    items_agg["seller_lng"] = items_agg.seller_zip.map(coords.lng)

    pay_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        n_cuotas=("payment_installments", "max"),
        n_medios_pago=("payment_type", "nunique"),
    ).reset_index()
    medio_principal = (payments.sort_values("payment_value")
                       .drop_duplicates("order_id", keep="last")[["order_id", "payment_type"]])
    pay_agg = pay_agg.merge(medio_principal, on="order_id", how="left")

    cust = customers[["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_state"]].copy()
    cust["cust_lat"] = cust.customer_zip_code_prefix.map(coords.lat)
    cust["cust_lng"] = cust.customer_zip_code_prefix.map(coords.lng)

    df = (orders.merge(cust, on="customer_id", how="left")
          .merge(items_agg, on="order_id", how="left")
          .merge(pay_agg, on="order_id", how="left")
          .merge(rev, on="order_id", how="left"))
    if len(df) != len(orders):
        raise ValueError("El tablón multiplicó filas al unir las tablas.")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de compra y de ejecución logística observada al enviar la encuesta."""
    df = df.copy()
    compra = df.order_purchase_timestamp

    df["dia_semana"] = compra.dt.dayofweek
    df["hora"] = compra.dt.hour
    df["fin_de_semana"] = (df.dia_semana >= 5).astype(int)
    df["horas_aprobacion"] = ((df.order_approved_at - compra) / pd.Timedelta(hours=1)).clip(lower=0)
    df["dias_estimados"] = (df.order_estimated_delivery_date - compra) / DIA
    df["dias_limite_despacho"] = (df.shipping_limit_date - compra) / DIA
    df["ratio_flete"] = df.freight_total / df.price_total
    df["precio_por_item"] = df.price_total / df.n_items
    df["distancia_km"] = haversine_km(df.cust_lat, df.cust_lng, df.seller_lat, df.seller_lng)
    df["mismo_estado"] = (df.customer_state == df.seller_state).astype(int)
    df["region_vendedor"] = df.seller_state.map(REGION)

    # Ejecución logística OBSERVADA al enviar la encuesta (T): nada posterior a T entra al modelo
    T = df.review_creation_date
    entregado_en_T = df.order_delivered_customer_date.dt.normalize() <= T
    despachado_en_T = df.order_delivered_carrier_date.dt.normalize() <= T
    df["pendiente_en_encuesta"] = (~entregado_en_T).astype(int)
    df["dias_despacho"] = ((df.order_delivered_carrier_date - compra) / DIA).where(despachado_en_T)
    df["despacho_tardio"] = np.where(despachado_en_T, df.order_delivered_carrier_date > df.shipping_limit_date,
                                     T > df.shipping_limit_date).astype(int)
    df["dias_transporte"] = ((df.order_delivered_customer_date - df.order_delivered_carrier_date) / DIA).where(entregado_en_T)
    df["dias_entrega"] = ((df.order_delivered_customer_date - compra) / DIA).where(entregado_en_T)
    df["dias_retraso"] = ((df.order_delivered_customer_date - df.order_estimated_delivery_date) / DIA).where(
        entregado_en_T, (T - df.order_estimated_delivery_date) / DIA)
    df["entrega_tardia"] = (df.dias_retraso > 0).astype(int)
    df["estado_en_encuesta"] = np.select([~entregado_en_T, df.dias_retraso > 0], ["Pendiente", "Entregado tarde"],
                                         "Entregado a tiempo")

    df[TARGET] = (df.review_score <= 2).astype(float).where(df.review_score.notna())
    return df


def study_population(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Embudo de filtros; devuelve (población, tabla del embudo)."""
    embudo = [("Pedidos totales", len(df))]
    base = df[df[TARGET].notna()]
    embudo.append(("Con reseña", len(base)))
    base = base[base.order_purchase_timestamp.between(PERIODO_INICIO, PERIODO_FIN)]
    embudo.append(("Compras ene-2017 a ago-2018", len(base)))
    base = base[base.n_items.notna()]
    embudo.append(("Con ítems registrados", len(base)))
    base = base[~(base.order_delivered_customer_date < base.order_delivered_carrier_date)]
    embudo.append(("Fechas logísticas coherentes", len(base)))
    tabla = pd.DataFrame(embudo, columns=["paso", "pedidos"])
    tabla["% del total"] = (tabla.pedidos / tabla.pedidos.iat[0] * 100).round(2)
    return base.reset_index(drop=True), tabla


def temporal_split(base: pd.DataFrame, corte: pd.Timestamp = CORTE):
    train = base[base.order_purchase_timestamp < corte]
    test = base[base.order_purchase_timestamp >= corte]
    return train, test


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    log_cols = [c for c in num_cols if c in LOG_COLS]
    lin_cols = [c for c in num_cols if c not in LOG_COLS]
    return ColumnTransformer([
        ("log", make_pipeline(SimpleImputer(strategy="median"),
                              FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                              StandardScaler()), log_cols),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), lin_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=300,
                                            sparse_output=False)), cat_cols),
    ], verbose_feature_names_out=False)


def model_candidates(y_train: pd.Series, quick: bool = False) -> dict[str, Any]:
    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    modelos: dict[str, Any] = {
        "Baseline (prior)": DummyClassifier(strategy="prior"),
        "Regresión Logística": LogisticRegression(class_weight="balanced", max_iter=2000),
        "Naive Bayes": GaussianNB(),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150 if quick else 400, class_weight="balanced",
            random_state=RANDOM_STATE),
    }
    if not quick:
        modelos["SVM lineal"] = CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=0.05), cv=3)
        modelos["Random Forest"] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE)
        try:
            from xgboost import XGBClassifier
            modelos["XGBoost"] = XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=5, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=ratio, eval_metric="aucpr",
                tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
        except ImportError:  # pragma: no cover - xgboost es opcional para la corrida corta
            logger.warning("xgboost no está instalado: se omite del benchmark.")
    return modelos


SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1", "precision": "precision",
           "recall": "recall", "balanced_accuracy": "balanced_accuracy"}

SEARCH_SPACES = {
    "XGBoost": {
        "n_estimators": [300, 500, 800], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 5, 10], "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1, 5, 10],
    },
    "HistGradientBoosting": {
        "learning_rate": [0.02, 0.05, 0.1], "max_iter": [300, 500, 800], "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 50, 100], "l2_regularization": [0, 1, 5],
    },
    "Random Forest": {
        "n_estimators": [200, 300], "max_depth": [12, 16, 20], "min_samples_leaf": [3, 5, 10, 20],
        "max_features": ["sqrt", 0.3],
    },
    "Regresión Logística": {"C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
}


def make_pipeline_for(model: Any, num_cols: list[str] = NUM, cat_cols: list[str] = CAT) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("clf", clone(model))])


def cross_validate_models(X: pd.DataFrame, y: pd.Series, modelos: dict[str, Any],
                          n_splits: int = 5) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for nombre, modelo in modelos.items():
        t0 = time.time()
        res = cross_validate(make_pipeline_for(modelo), X, y, cv=cv, scoring=SCORING)
        fila: dict[str, Any] = {"modelo": nombre}
        for m in SCORING:
            fila[f"{m}_mean"] = float(res[f"test_{m}"].mean())
            fila[f"{m}_std"] = float(res[f"test_{m}"].std())
        fila["segundos"] = time.time() - t0
        filas.append(fila)
        logger.info("%-22s PR-AUC=%.4f ± %.4f | ROC-AUC=%.4f | %.0fs", nombre, fila["pr_auc_mean"],
                    fila["pr_auc_std"], fila["roc_auc_mean"], fila["segundos"])
    return pd.DataFrame(filas).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def select_by_parsimony(cv_df: pd.DataFrame) -> tuple[str, list[str]]:
    """Entre los modelos a menos de 1 desviación estándar del mejor, el más simple."""
    mejor = cv_df.iloc[0]
    empatados = cv_df[cv_df.pr_auc_mean >= mejor.pr_auc_mean - mejor.pr_auc_std]
    ganador = min(empatados.modelo, key=lambda m: COMPLEJIDAD.index(m) if m in COMPLEJIDAD else 99)
    return ganador, empatados.modelo.tolist()


def tune_model(nombre: str, modelo: Any, X: pd.DataFrame, y: pd.Series, n_iter: int = 15) -> tuple[Any, dict]:
    if nombre not in SEARCH_SPACES:
        return clone(modelo), {}
    busqueda = RandomizedSearchCV(
        make_pipeline_for(modelo), {f"clf__{k}": v for k, v in SEARCH_SPACES[nombre].items()}, n_iter=n_iter,
        scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=1, refit=False)
    busqueda.fit(X, y)
    mejores = {k.removeprefix("clf__"): v for k, v in busqueda.best_params_.items()}
    logger.info("Mejores hiperparámetros de %s: %s (PR-AUC %.4f)", nombre, mejores, busqueda.best_score_)
    return clone(modelo).set_params(**mejores), mejores


def without_weighting(estimador: Any) -> Any:
    est = clone(estimador)
    params = est.get_params()
    if "scale_pos_weight" in params:
        est.set_params(scale_pos_weight=1)
    if "class_weight" in params:
        est.set_params(class_weight=None)
    return est


def imbalance_strategies(clf: Any) -> dict[str, Pipeline]:
    estrategias = {
        "Sin tratamiento": make_pipeline_for(without_weighting(clf)),
        "Ponderación de clases": make_pipeline_for(clf),
    }
    try:
        from imblearn.over_sampling import ADASYN, SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        estrategias["SMOTE"] = ImbPipeline([("prep", build_preprocessor(NUM, CAT)),
                                            ("smote", SMOTE(random_state=RANDOM_STATE)),
                                            ("clf", without_weighting(clf))])
        estrategias["ADASYN"] = ImbPipeline([("prep", build_preprocessor(NUM, CAT)),
                                             ("adasyn", ADASYN(random_state=RANDOM_STATE)),
                                             ("clf", without_weighting(clf))])
    except ImportError:  # pragma: no cover
        logger.warning("imbalanced-learn no está instalado: se omiten SMOTE y ADASYN.")
    return estrategias


def compare_imbalance(estrategias: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv3 = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for nombre, pipe in estrategias.items():
        res = cross_validate(pipe, X, y, cv=cv3, scoring=SCORING)
        filas.append({"estrategia": nombre, **{m: float(res[f"test_{m}"].mean()) for m in SCORING}})
    return pd.DataFrame(filas).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def optimal_threshold(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> tuple[float, float, np.ndarray]:
    """Umbral que maximiza F1 con predicciones out-of-fold (nunca con el test)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1s)]), float(f1s.max()), oof


def holdout_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
    }


def gains_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    orden = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    orden["decil"] = np.ceil((orden.index + 1) / len(orden) * 10).astype(int)
    g = orden.groupby("decil").agg(pedidos=("y", "size"), negativas=("y", "sum"),
                                   tasa_negativas=("y", "mean"), prob_min=("p", "min"))
    g["captura_acumulada_%"] = g.negativas.cumsum() / max(g.negativas.sum(), 1) * 100
    g["lift"] = g.tasa_negativas / max(orden.y.mean(), 1e-12)
    return g


def capture_top(y, p, frac: float = 0.10) -> float:
    orden = np.argsort(-np.asarray(p))
    y = np.asarray(y)
    return float(y[orden[: int(len(orden) * frac)]].sum() / max(y.sum(), 1) * 100)


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------
def _r4(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 4)


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: (v if isinstance(v, str) else _r4(v)) for k, v in fila.items()}
            for fila in df.to_dict(orient="records")]


def _subsample(*arrays, n: int = 120):
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, n).astype(int))
    return [[_r4(v) for v in np.asarray(a)[idx]] for a in arrays]


def validate_manifest(manifest: dict, schema_path: Path = SCHEMA_PATH) -> None:
    import jsonschema
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)


def run_pipeline(data_dir: Path = RAW_DIR, artifacts_dir: Path = ARTIFACTS_DIR, quick: bool = False) -> dict:
    """Corrida completa: tablón → comparación → ajuste → umbral → test → artefactos."""
    t_inicio = time.time()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = add_features(build_table(load_raw(data_dir)))
    base, embudo = study_population(df)
    estado = (base.groupby("estado_en_encuesta")[TARGET].agg(["size", "mean"])
              .rename(columns={"size": "pedidos", "mean": "tasa_negativas"})
              .reindex(["Entregado a tiempo", "Entregado tarde", "Pendiente"]))

    train, test = temporal_split(base)
    if quick:
        train = train.sample(n=min(12_000, len(train)), random_state=RANDOM_STATE)
        test = test.sample(n=min(4_000, len(test)), random_state=RANDOM_STATE)
    X_train, y_train = train[FEATURES], train[TARGET].astype(int)
    X_test, y_test = test[FEATURES], test[TARGET].astype(int)
    logger.info("Train %s pedidos (%.2f%% negativos) · Test %s pedidos (%.2f%% negativos)",
                f"{len(train):,}", y_train.mean() * 100, f"{len(test):,}", y_test.mean() * 100)

    # 1. Comparación de modelos en validación cruzada
    modelos = model_candidates(y_train, quick=quick)
    cv_df = cross_validate_models(X_train, y_train, modelos, n_splits=3 if quick else 5)
    ganador_base, empatados = select_by_parsimony(cv_df)
    logger.info("Ganador por parsimonia: %s (empatados: %s)", ganador_base, ", ".join(empatados))

    # 2. Ajuste e imbalance
    clf_ajustado, mejores = (clone(modelos[ganador_base]), {}) if quick else tune_model(
        ganador_base, modelos[ganador_base], X_train, y_train)
    estrategias = imbalance_strategies(clf_ajustado)
    desbalance_df = compare_imbalance(estrategias, X_train, y_train)
    estrategia = str(desbalance_df.estrategia.iat[0])
    pipe_final = clone(estrategias[estrategia])
    ganador = f"{ganador_base} ({estrategia.lower()}{', ajustado' if mejores else ''})"

    # 3. Umbral fuera de muestra y evaluación final
    umbral, f1_oof, _ = optimal_threshold(pipe_final, X_train, y_train, n_splits=3 if quick else 5)
    pipe_final.fit(X_train, y_train)
    p_test = pipe_final.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= umbral).astype(int)
    metricas_test = holdout_metrics(y_test.to_numpy(), p_test, y_pred)

    filas_test = []
    for nombre, modelo in modelos.items():
        p = make_pipeline_for(modelo).fit(X_train, y_train).predict_proba(X_test)[:, 1]
        filas_test.append({"modelo": nombre, "roc_auc": roc_auc_score(y_test, p),
                           "pr_auc": average_precision_score(y_test, p)})
    filas_test.append({"modelo": ganador, "roc_auc": metricas_test["roc_auc"], "pr_auc": metricas_test["pr_auc"]})
    test_df = pd.DataFrame(filas_test).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    frac_pos, prob_media = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    ganancia = gains_table(y_test.to_numpy(), p_test)

    # 4. Importancia por permutación (en test, sobre variables originales)
    muestra = X_test.sample(n=min(2_000 if quick else 8_000, len(X_test)), random_state=RANDOM_STATE)
    imp = permutation_importance(pipe_final, muestra, y_test.loc[muestra.index], scoring="average_precision",
                                 n_repeats=2 if quick else 5, random_state=RANDOM_STATE, n_jobs=1)
    importancias = (pd.DataFrame({"variable": FEATURES, "importancia": imp.importances_mean, "std": imp.importances_std})
                    .sort_values("importancia", ascending=False).reset_index(drop=True))
    importancias["familia"] = np.where(importancias.variable.isin(NUM_ENCUESTA), "Logística observada",
                                       "Conocida al comprar")

    # 5. Escenario complementario: solo variables conocidas al comprar
    pipe_compra = make_pipeline_for(pipe_final.named_steps["clf"], NUM_COMPRA, CAT)
    pipe_compra.fit(X_train[FEATURES_COMPRA], y_train)
    pc_test = pipe_compra.predict_proba(X_test[FEATURES_COMPRA])[:, 1]
    prevalencia = float(y_test.mean())
    escenarios_df = pd.DataFrame([
        {"escenario": "Al enviar la encuesta (modelo principal)", "variables": len(FEATURES),
         "roc_auc": metricas_test["roc_auc"], "pr_auc": metricas_test["pr_auc"],
         "lift_pr_auc": metricas_test["pr_auc"] / prevalencia, "captura_top10_%": capture_top(y_test, p_test)},
        {"escenario": "Al comprar (sin logística)", "variables": len(FEATURES_COMPRA),
         "roc_auc": float(roc_auc_score(y_test, pc_test)), "pr_auc": float(average_precision_score(y_test, pc_test)),
         "lift_pr_auc": float(average_precision_score(y_test, pc_test)) / prevalencia,
         "captura_top10_%": capture_top(y_test, pc_test)},
    ])

    # 6. Artefactos del contrato
    joblib.dump({
        "pipeline": pipe_final,
        "threshold": umbral,
        "features_num": NUM,
        "features_cat": CAT,
        "features_compra": FEATURES_COMPRA,
        "target": "review_negativa (review_score <= 2)",
        "momento_prediccion": "Al enviar la encuesta de satisfacción (review_creation_date), "
                              "antes de la respuesta del cliente",
        "winner_name": ganador,
        "version": VERSION,
        "train_medians": X_train[NUM].median(numeric_only=True).to_dict(),
    }, artifacts_dir / "model.joblib", compress=3)

    resultados = pd.DataFrame({
        "order_id": test.order_id.to_numpy(),
        "fecha_compra": test.order_purchase_timestamp.dt.date.to_numpy(),
        "categoria_principal": test.categoria_principal.to_numpy(),
        "dias_retraso": test.dias_retraso.round(1).to_numpy(),
        "clase_real": y_test.to_numpy(),
        "probabilidad": p_test.round(4),
        "prediccion": y_pred,
    })
    # Muestra de fondo para explicaciones LIME (solo entrenamiento, sin etiqueta)
    X_train.sample(n=min(500, len(X_train)), random_state=RANDOM_STATE).to_csv(
        artifacts_dir / "explain_background.csv", index=False)

    n_muestra = min(500, len(resultados))
    n_neg = int(round(n_muestra * prevalencia))
    negativos = resultados[resultados.clase_real == 1]
    positivos = resultados[resultados.clase_real == 0]
    muestra_res = pd.concat([
        negativos.sample(n=min(n_neg, len(negativos)), random_state=RANDOM_STATE),
        positivos.sample(n=min(n_muestra - n_neg, len(positivos)), random_state=RANDOM_STATE),
    ]).sort_values("probabilidad", ascending=False).reset_index(drop=True)
    muestra_res.to_csv(artifacts_dir / "results.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    pr, rc, _ = precision_recall_curve(y_test, p_test)
    fpr_s, tpr_s = _subsample(fpr, tpr)
    rc_s, pr_s = _subsample(rc[::-1], pr[::-1])
    chart_data = {
        "threshold": _r4(umbral),
        "prevalence_test": _r4(prevalencia),
        "model_comparison_cv": _records(cv_df.drop(columns="segundos")),
        "model_comparison_test": _records(test_df),
        "imbalance_strategies": _records(desbalance_df),
        "confusion_matrix": {"labels": ["Positiva/neutra (0)", "Negativa (1)"], "matrix": cm.tolist(),
                             "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr_s, "tpr": tpr_s, "auc": _r4(metricas_test["roc_auc"])},
        "pr_curve": {"recall": rc_s, "precision": pr_s, "average_precision": _r4(metricas_test["pr_auc"])},
        "calibration": {"prob_media": [_r4(v) for v in prob_media], "fraccion_positivos": [_r4(v) for v in frac_pos]},
        "gains_by_decile": [{"decil": int(d), **{k: _r4(v) for k, v in fila.items()}}
                            for d, fila in ganancia.to_dict(orient="index").items()],
        "feature_importance": [{"variable": f.variable, "familia": f.familia, "importancia": _r4(f.importancia),
                                "std": _r4(f.std)} for f in importancias.head(15).itertuples()],
        "scenario_comparison": _records(escenarios_df),
        "status_at_survey": [{"estado": idx, "pedidos": int(fila.pedidos), "tasa_negativas": _r4(fila.tasa_negativas)}
                             for idx, fila in estado.iterrows()],
        "funnel": _records(embudo),
    }
    with (artifacts_dir / "chart_data.json").open("w", encoding="utf-8") as f:
        json.dump(chart_data, f, ensure_ascii=False, indent=2)

    top3 = importancias.variable.head(3).tolist()
    dec1, dec3 = ganancia.loc[1], ganancia.loc[3] if 3 in ganancia.index else ganancia.iloc[-1]
    esc_compra = escenarios_df.iloc[1]
    pr_cv_ganador = float(cv_df.loc[cv_df.modelo == ganador_base, "pr_auc_mean"].iat[0])
    tasas = (estado.tasa_negativas * 100).fillna(0)
    conclusiones = [
        f"Priorización accionable: contactar el 10% de pedidos con mayor riesgo captura el "
        f"{dec1['captura_acumulada_%']:.1f}% de las reseñas negativas (lift {dec1['lift']:.1f}x); "
        f"con el 30% se captura el {dec3['captura_acumulada_%']:.1f}%.",
        f"La promesa de entrega es la palanca principal: los pedidos pendientes al enviar la encuesta tienen "
        f"{tasas['Pendiente']:.1f}% de negativas y los entregados tarde {tasas['Entregado tarde']:.1f}%, frente a "
        f"{tasas['Entregado a tiempo']:.1f}% de los puntuales. Variables más influyentes: {', '.join(top3)}.",
        f"Con el umbral {umbral:.2f} el modelo detecta el {metricas_test['recall'] * 100:.1f}% de las negativas con "
        f"una precisión del {metricas_test['precision'] * 100:.1f}% (PR-AUC {metricas_test['pr_auc']:.3f} frente a "
        f"una prevalencia de {prevalencia:.3f}).",
        f"Al momento de la compra el riesgo es poco anticipable (PR-AUC {esc_compra.pr_auc:.3f} vs "
        f"{metricas_test['pr_auc']:.3f} con seguimiento logístico): la acción preventiva debe apoyarse en el "
        "monitoreo del despacho y la entrega, no solo en el perfil del pedido.",
    ]
    limitaciones = [
        "Momento de predicción: el modelo se aplica al enviar la encuesta; para usarlo antes (al comprar o al "
        "despachar) el desempeño cae, como muestra el escenario complementario.",
        "Deriva temporal: la tasa de negativas varía entre meses (picos en nov-2017 y feb–mar 2018); el modelo "
        f"debe reentrenarse periódicamente y vigilar la prevalencia (test: {prevalencia * 100:.1f}%).",
        ("Probabilidades no calibradas: la ponderación de clases o SMOTE inflan las probabilidades; sirven para "
         "ordenar el riesgo, no como frecuencia esperada."
         if estrategia != "Sin tratamiento" else
         f"Brecha validación–test: la PR-AUC baja de {pr_cv_ganador:.3f} (CV) a {metricas_test['pr_auc']:.3f} "
         f"(test) sobre todo porque la prevalencia cae de {y_train.mean() * 100:.1f}% a {prevalencia * 100:.1f}%; "
         "el lift sobre el azar se mantiene, pero la ROC-AUC también desciende, señal de deriva a monitorear."),
        "Sin variables de texto ni historial del vendedor: el comentario de la reseña es posterior a la "
        "predicción y el historial de reputación del vendedor no se incluyó para evitar fuga temporal.",
    ]
    manifest = {
        "status": "Completado",
        "model_name": "Riesgo de reseña negativa",
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Olist Brazilian E-Commerce",
        "period": f"Entrenamiento {train.order_purchase_timestamp.min():%Y-%m} a "
                  f"{train.order_purchase_timestamp.max():%Y-%m}; prueba "
                  f"{test.order_purchase_timestamp.min():%Y-%m} a {test.order_purchase_timestamp.max():%Y-%m}",
        "metrics": {
            **{k: _r4(v) for k, v in metricas_test.items()},
            "umbral": _r4(umbral),
            "f1_oof": _r4(f1_oof),
            "prevalencia_test": _r4(prevalencia),
            "cv_pr_auc": _r4(pr_cv_ganador),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "winner": ganador,
        "results_path": "artifacts/classification/results.csv",
        "chart_data_path": "artifacts/classification/chart_data.json",
        "models_compared": _records(cv_df[["modelo", "pr_auc_mean", "pr_auc_std", "roc_auc_mean"]]),
        "tied_models": empatados,
        "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in mejores.items()},
        "imbalance_strategy": estrategia,
        "validation": "Partición temporal (corte 2018-06-01) + validación cruzada estratificada en entrenamiento",
        "business_conclusions": conclusiones,
        "limitations": limitaciones,
        "serialized_model_path": "artifacts/classification/model.joblib",
        "quick_run": bool(quick),
    }
    validate_manifest(manifest)
    with (artifacts_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    cargado = joblib.load(artifacts_dir / "model.joblib")
    if not np.allclose(cargado["pipeline"].predict_proba(X_test)[:, 1], p_test):
        raise RuntimeError("El modelo serializado no reproduce las probabilidades del test.")
    logger.info("Artefactos generados en %s en %.1f min", artifacts_dir, (time.time() - t_inicio) / 60)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y exporta el módulo de clasificación.")
    parser.add_argument("--quick", action="store_true",
                        help="Corrida corta (submuestra, sin ajuste de hiperparámetros) para pruebas.")
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = run_pipeline(args.data_dir, args.artifacts_dir, quick=args.quick)
    print(json.dumps(manifest["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
