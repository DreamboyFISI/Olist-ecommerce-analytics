"""Servicio de inferencia y consulta del módulo de clasificación (riesgo de reseña negativa)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.classification import CAT, NUM, NUM_ENCUESTA, REGION

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "classification"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
CHART_DATA_PATH = ARTIFACT_DIR / "chart_data.json"
RESULTS_PATH = ARTIFACT_DIR / "results.csv"

DIA = pd.Timedelta(days=1)
STATES = sorted(REGION)


@lru_cache(maxsize=1)
def _load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No se encontró artifacts/classification/model.joblib.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No se encontró artifacts/classification/manifest.json.")
    with MANIFEST_PATH.open(encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def _load_chart_data() -> dict:
    if not CHART_DATA_PATH.exists():
        raise FileNotFoundError("No se encontró artifacts/classification/chart_data.json.")
    with CHART_DATA_PATH.open(encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def _load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("No se encontró artifacts/classification/results.csv.")
    return pd.read_csv(RESULTS_PATH)


def _onehot_categories() -> dict[str, list[str]]:
    """Valores conocidos por el modelo para cada variable categórica (del OneHotEncoder ajustado)."""
    pipeline = _load_model()["pipeline"]
    prep = pipeline.named_steps["prep"]
    cat_pipe = prep.named_transformers_["cat"]
    onehot = cat_pipe.steps[-1][1]
    cat_cols = [cols for name, _, cols in prep.transformers_ if name == "cat"][0]
    return {col: [str(v) for v in cats] for col, cats in zip(cat_cols, onehot.categories_)}


def load_classification_results() -> dict:
    """Manifest, datos de gráficas, muestra de predicciones y catálogos para el formulario."""
    results = _load_results()
    try:
        categorias = _onehot_categories()
    except (FileNotFoundError, AttributeError, KeyError, IndexError):
        categorias = {}
    return {
        "manifest": _load_manifest(),
        "chart_data": _load_chart_data(),
        "results": results.head(200).where(pd.notna(results), None).to_dict(orient="records"),
        "states": STATES,
        "payment_types": categorias.get("payment_type", ["credit_card", "boleto", "voucher", "debit_card"]),
        "categories": categorias.get("categoria_principal", sorted(results.categoria_principal.dropna().unique())),
        "features_num": NUM,
        "features_cat": CAT,
    }


def _ts(value) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _num(payload: dict, key: str, default: float = np.nan) -> float:
    """Lee un número del payload tratando ``None`` como ausente (el pipeline lo imputa)."""
    value = payload.get(key)
    return default if value is None else float(value)


def build_feature_frame(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Reconstruye las variables del modelo a partir de los datos de un pedido.

    Reproduce exactamente las reglas de ``src.models.classification.add_features``
    para el instante en que se envía la encuesta (``survey_timestamp``):
    todo lo posterior a ese momento se ignora.
    """
    compra = _ts(payload["purchase_timestamp"])
    encuesta = _ts(payload.get("survey_timestamp")) or (compra + DIA * 20)
    estimada = _ts(payload.get("estimated_delivery_date")) or (compra + DIA * 24)
    limite = _ts(payload.get("shipping_limit_date")) or (compra + DIA * 6)
    despacho = _ts(payload.get("delivered_carrier_date"))
    entrega = _ts(payload.get("delivered_customer_date"))

    if encuesta < compra:
        raise ValueError("La fecha de la encuesta no puede ser anterior a la compra.")
    if estimada < compra:
        raise ValueError("La fecha prometida no puede ser anterior a la compra.")
    if despacho is not None and despacho < compra:
        raise ValueError("La fecha de despacho no puede ser anterior a la compra.")
    if entrega is not None and entrega < compra:
        raise ValueError("La fecha de entrega no puede ser anterior a la compra.")
    if entrega is not None and despacho is not None and entrega < despacho:
        raise ValueError("La entrega no puede ser anterior al despacho.")

    T = encuesta.normalize()
    entregado_en_T = entrega is not None and entrega.normalize() <= T
    despachado_en_T = despacho is not None and despacho.normalize() <= T

    n_items = max(int(_num(payload, "n_items", 1)), 1)
    price_total = _num(payload, "price_total", 0.0)
    freight_total = _num(payload, "freight_total", 0.0)
    customer_state = str(payload.get("customer_state") or "SP").upper()
    seller_state = str(payload.get("seller_state") or "SP").upper()

    fila = {
        "dia_semana": compra.day_of_week,
        "hora": compra.hour,
        "fin_de_semana": int(compra.day_of_week >= 5),
        "horas_aprobacion": max(_num(payload, "approval_hours", 0.5), 0.0),
        "dias_estimados": (estimada - compra) / DIA,
        "dias_limite_despacho": (limite - compra) / DIA,
        "n_items": n_items,
        "n_productos": int(_num(payload, "n_products", n_items)),
        "n_sellers": int(_num(payload, "n_sellers", 1)),
        "n_categorias": int(_num(payload, "n_categories", 1)),
        "price_total": price_total,
        "freight_total": freight_total,
        "ratio_flete": freight_total / price_total if price_total > 0 else np.nan,
        "precio_por_item": price_total / n_items,
        "peso_total_g": _num(payload, "weight_total_g"),
        "volumen_total_cm3": _num(payload, "volume_total_cm3"),
        "fotos_media": _num(payload, "photos_mean"),
        "largo_descripcion_media": _num(payload, "description_length_mean"),
        "n_cuotas": int(_num(payload, "installments", 1)),
        "n_medios_pago": int(_num(payload, "n_payment_types", 1)),
        "distancia_km": _num(payload, "distance_km"),
        "mismo_estado": int(customer_state == seller_state),
        "pendiente_en_encuesta": int(not entregado_en_T),
        "dias_despacho": (despacho - compra) / DIA if despachado_en_T else np.nan,
        "despacho_tardio": int(despacho > limite) if despachado_en_T else int(T > limite),
        "dias_transporte": ((entrega - despacho) / DIA) if (entregado_en_T and despacho is not None) else np.nan,
        "dias_entrega": (entrega - compra) / DIA if entregado_en_T else np.nan,
        "dias_retraso": ((entrega - estimada) / DIA) if entregado_en_T else ((T - estimada) / DIA),
        "payment_type": str(payload.get("payment_type") or "credit_card"),
        "customer_state": customer_state,
        "region_vendedor": REGION.get(seller_state, "Sudeste"),
        "categoria_principal": str(payload.get("main_category") or "sin_categoria"),
    }
    fila["entrega_tardia"] = int(fila["dias_retraso"] > 0)
    if not entregado_en_T:
        estado = "Pendiente"
    elif fila["dias_retraso"] > 0:
        estado = "Entregado tarde"
    else:
        estado = "Entregado a tiempo"
    contexto = {
        "estado_en_encuesta": estado,
        "dias_retraso": round(float(fila["dias_retraso"]), 1),
        "dias_estimados": round(float(fila["dias_estimados"]), 1),
        "dias_entrega": None if np.isnan(fila["dias_entrega"]) else round(float(fila["dias_entrega"]), 1),
        "pendiente_en_encuesta": bool(fila["pendiente_en_encuesta"]),
        "despacho_tardio": bool(fila["despacho_tardio"]),
    }
    return pd.DataFrame([fila])[NUM + CAT], contexto


@lru_cache(maxsize=1)
def _load_background() -> pd.DataFrame | None:
    """Muestra del entrenamiento para LIME; la exporta el pipeline cuando se reentrena."""
    ruta = ARTIFACT_DIR / "explain_background.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


def explain_prediction(payload: dict, top_n: int = 10, prefer: str = "shap") -> dict:
    """Por qué el modelo asigna ese riesgo: contribución de cada variable (SHAP, LIME o perturbación)."""
    from src.explainability.explain import explain_instance

    modelo = _load_model()
    X, contexto = build_feature_frame(payload)
    explicacion = explain_instance(modelo["pipeline"], X, background=_load_background(), top_n=top_n, prefer=prefer)
    explicacion["umbral"] = round(float(modelo.get("threshold", 0.5)), 4)
    explicacion["contexto"] = contexto
    return explicacion


def _risk_level(probabilidad: float, umbral: float) -> str:
    if probabilidad >= umbral:
        return "Alto"
    if probabilidad >= umbral / 2:
        return "Medio"
    return "Bajo"


def _decile_context(probabilidad: float) -> dict | None:
    """Ubica la probabilidad en los deciles de riesgo del test (curva de ganancia)."""
    try:
        deciles = _load_chart_data().get("gains_by_decile", [])
    except FileNotFoundError:
        return None
    for fila in sorted(deciles, key=lambda d: d["decil"]):
        if fila.get("prob_min") is not None and probabilidad >= fila["prob_min"]:
            return {"decil": int(fila["decil"]), "tasa_negativas": fila.get("tasa_negativas"),
                    "lift": fila.get("lift"), "captura_acumulada_%": fila.get("captura_acumulada_%")}
    return {"decil": 10, "tasa_negativas": deciles[-1].get("tasa_negativas") if deciles else None,
            "lift": deciles[-1].get("lift") if deciles else None, "captura_acumulada_%": 100.0}


def predict_risk(payload: dict) -> dict:
    """Probabilidad de reseña negativa para un pedido descrito en ``payload``.

    Inferencia pura: el pipeline serializado incluye imputación, escalado y
    codificación, así que aquí solo se reconstruyen las variables y se llama a
    ``predict_proba``. No se entrena nada desde Streamlit ni desde la API.
    """
    modelo = _load_model()
    pipeline = modelo["pipeline"]
    umbral = float(modelo.get("threshold", 0.5))

    X, contexto = build_feature_frame(payload)
    probabilidad = float(pipeline.predict_proba(X)[:, 1][0])
    prediccion = int(probabilidad >= umbral)
    nivel = _risk_level(probabilidad, umbral)

    try:
        prevalencia = float(_load_manifest()["metrics"].get("prevalencia_test") or 0.0)
    except (FileNotFoundError, KeyError):
        prevalencia = 0.0

    return {
        "probabilidad": round(probabilidad, 4),
        "prediccion": prediccion,
        "etiqueta": "Riesgo de reseña negativa" if prediccion else "Sin alerta",
        "nivel_riesgo": nivel,
        "umbral": round(umbral, 4),
        "prevalencia_referencia": round(prevalencia, 4),
        "veces_sobre_prevalencia": round(probabilidad / prevalencia, 2) if prevalencia else None,
        "decil_referencia": _decile_context(probabilidad),
        "contexto": contexto,
        "variables": {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
                      for k, v in X.iloc[0].to_dict().items()},
        "modelo": modelo.get("winner_name", "Riesgo de reseña negativa"),
        "version": modelo.get("version"),
        "momento_prediccion": modelo.get("momento_prediccion"),
    }
