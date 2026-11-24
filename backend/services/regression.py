import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "regression"
MODEL_PATH = ARTIFACT_DIR / "hybrid_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.csv"
PREDICTIONS_PATH = ARTIFACT_DIR / "predictions.csv"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
CLASSIFIER_METRICS_PATH = ARTIFACT_DIR / "classifier_metrics.csv"
NORMAL_METRICS_PATH = ARTIFACT_DIR / "normal_regression_metrics.csv"
HIGH_METRICS_PATH = ARTIFACT_DIR / "high_regression_metrics.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"
HIGH_FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "high_feature_importance.csv"
SELLER_HISTORY_PATH = ARTIFACT_DIR / "seller_freight_history.csv"
ROUTE_HISTORY_PATH = ARTIFACT_DIR / "route_freight_history.csv"
SELLER_LOCATIONS_PATH = ARTIFACT_DIR / "seller_locations.csv"
CUSTOMER_LOCATIONS_PATH = ARTIFACT_DIR / "customer_locations.csv"


def _set_pipeline_to_cpu(pipeline) -> None:
    model = getattr(pipeline, "named_steps", {}).get("model")
    if model is not None and hasattr(model, "set_params"):
        try:
            model.set_params(device="cpu")
        except ValueError:
            pass


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    bundle = joblib.load(MODEL_PATH)
    for key in ("classifier", "normal_regressor", "high_regressor"):
        _set_pipeline_to_cpu(bundle[key])
    return bundle


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def _load_histories() -> tuple[dict[str, float], dict[str, float], float]:
    bundle = _load_bundle()
    seller_history = bundle.get("seller_freight_history")
    route_history = bundle.get("route_freight_history")
    fallback = bundle.get("historical_freight_fallback")

    if seller_history is None:
        seller_frame = pd.read_csv(SELLER_HISTORY_PATH)
        seller_history = seller_frame.set_index("main_seller_id")["seller_historical_freight"].to_dict()
    if route_history is None:
        route_frame = pd.read_csv(ROUTE_HISTORY_PATH)
        route_history = route_frame.set_index("state_pair")["route_historical_freight"].to_dict()
    if fallback is None:
        fallback = _load_manifest()["historical_features"]["fallback_brl"]

    return seller_history, route_history, float(fallback)


def _available_categories() -> list[str]:
    pipeline = _load_bundle()["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    for name, transformer, columns in preprocessor.transformers_:
        if name != "categorical" or "main_category" not in columns:
            continue
        encoder = transformer.named_steps["onehot"]
        position = list(columns).index("main_category")
        return sorted(str(value) for value in encoder.categories_[position])
    return ["beleza_saude"]


def _available_states() -> list[str]:
    seller_states = pd.read_csv(SELLER_LOCATIONS_PATH, usecols=["state"])["state"]
    customer_states = pd.read_csv(CUSTOMER_LOCATIONS_PATH, usecols=["state"])["state"]
    return sorted(set(seller_states.dropna()) | set(customer_states.dropna()))


def _build_feature_frame(payload: dict) -> pd.DataFrame:
    bundle = _load_bundle()
    seller_history, route_history, fallback = _load_histories()
    purchased_at = pd.Timestamp(payload["purchase_timestamp"])
    seller_state = payload["main_seller_state"].strip().upper()
    customer_state = payload["customer_state"].strip().upper()
    seller_id = payload["seller_id"].strip()
    state_pair = f"{seller_state}_{customer_state}"

    n_items = int(payload["n_items"])
    weight_total_g = float(payload["weight_total_g"])
    volume_total_cm3 = float(payload["volume_total_cm3"])
    distance_km = float(payload["distance_km"])
    weight_total_kg = weight_total_g / 1_000
    volumetric_weight_kg = volume_total_cm3 / 6_000
    chargeable_weight_kg = max(weight_total_kg, volumetric_weight_kg)

    geo_pipeline = bundle["geographic_cluster_pipeline"]
    seller_cluster = str(int(geo_pipeline.predict([[
        float(payload["seller_lat"]), float(payload["seller_lng"])
    ]])[0]))
    customer_cluster = str(int(geo_pipeline.predict([[
        float(payload["customer_lat"]), float(payload["customer_lng"])
    ]])[0]))

    row = {
        "n_items": n_items,
        "n_products": int(payload["n_products"]),
        "n_sellers": int(payload["n_sellers"]),
        "price_total": float(payload["price_total"]),
        "weight_total_g": weight_total_g,
        "volume_total_cm3": volume_total_cm3,
        "main_category": payload["main_category"].strip().lower(),
        "customer_state": customer_state,
        "main_seller_state": seller_state,
        "purchase_year": purchased_at.year,
        "purchase_month": purchased_at.month,
        "purchase_dayofweek": purchased_at.dayofweek,
        "purchase_hour": purchased_at.hour,
        "purchase_is_weekend": int(purchased_at.dayofweek >= 5),
        "distance_km": distance_km,
        "same_state": int(seller_state == customer_state),
        "state_pair": state_pair,
        "weight_total_kg": weight_total_kg,
        "volumetric_weight_kg": volumetric_weight_kg,
        "chargeable_weight_kg": chargeable_weight_kg,
        "avg_item_price": float(payload["price_total"]) / n_items,
        "weight_per_item_kg": weight_total_kg / n_items,
        "volume_per_item_cm3": volume_total_cm3 / n_items,
        "distance_x_chargeable_weight": distance_km * chargeable_weight_kg,
        "distance_x_items": distance_km * n_items,
        "seller_historical_freight": float(seller_history.get(seller_id, fallback)),
        "route_historical_freight": float(route_history.get(state_pair, fallback)),
        "seller_geo_cluster": seller_cluster,
        "customer_geo_cluster": customer_cluster,
        "geo_cluster_pair": f"{seller_cluster}_{customer_cluster}",
    }
    return pd.DataFrame([row])


def predict_freight(payload: dict) -> dict:
    bundle = _load_bundle()
    features = _build_feature_frame(payload)
    classifier_features = bundle["classifier_features"]
    high_probability = float(
        bundle["classifier"].predict_proba(features[classifier_features])[0, 1]
    )
    threshold = float(bundle["classifier_threshold"])
    use_high_regressor = high_probability >= threshold

    if use_high_regressor:
        regressor = bundle["high_regressor"]
        regressor_features = bundle["high_features"]
        segment = "Alto"
        typical_error = float(_load_manifest()["test_metrics"]["MAE_fletes_altos"])
    else:
        regressor = bundle["normal_regressor"]
        regressor_features = bundle["normal_features"]
        segment = "Normal"
        normal_metrics = pd.read_csv(NORMAL_METRICS_PATH)
        chosen = normal_metrics.loc[
            normal_metrics["modelo"].eq("XGBoost")
            & normal_metrics["subconjunto"].eq("Top 15")
        ].iloc[0]
        typical_error = float(chosen["MAE"])

    prediction = max(0.0, float(regressor.predict(features[regressor_features])[0]))
    return {
        "predicted_freight": round(prediction, 2),
        "currency": "BRL",
        "segment": segment,
        "high_freight_probability": round(high_probability, 4),
        "high_freight_limit": float(bundle["high_freight_limit_brl"]),
        "typical_error": round(typical_error, 2),
        "model": "XGBoost",
        "seller_reference": round(float(features.loc[0, "seller_historical_freight"]), 2),
        "route_reference": round(float(features.loc[0, "route_historical_freight"]), 2),
        "distance_km": round(float(features.loc[0, "distance_km"]), 1),
        "chargeable_weight_kg": round(float(features.loc[0, "chargeable_weight_kg"]), 2),
    }


def _selected_component_metrics() -> dict:
    test_metrics = _load_manifest()["test_metrics"]
    classifier = pd.read_csv(CLASSIFIER_METRICS_PATH)
    classifier_row = classifier.loc[
        classifier["modelo"].eq("XGBoost")
        & classifier["subconjunto"].eq("Todas")
        & np.isclose(classifier["umbral"], 0.50)
    ].iloc[0]

    normal = pd.read_csv(NORMAL_METRICS_PATH)
    normal_row = normal.loc[
        normal["modelo"].eq("XGBoost")
        & normal["subconjunto"].eq("Top 15")
    ].iloc[0]

    high = pd.read_csv(HIGH_METRICS_PATH)
    high_row = high.loc[
        high["candidate_key"].eq("xgb_quantile_70")
        & high["subconjunto"].eq("Top 15")
    ].iloc[0]

    return {
        "classifier": {
            "model": "XGBoost",
            "precision": float(test_metrics["precision_clase_alta"]),
            "recall": float(test_metrics["recall_clase_alta"]),
            "f1": float(test_metrics["F1_clase_alta"]),
            "accuracy": float(classifier_row["accuracy"]),
        },
        "normal_regressor": {
            "model": "XGBoost",
            "MAE": float(normal_row["MAE"]),
            "RMSE": float(normal_row["RMSE"]),
            "R2": float(normal_row["R2"]),
        },
        "high_regressor": {
            "model": "XGBoost · cuantil 0.70",
            "MAE": float(high_row["MAE"]),
            "RMSE": float(high_row["RMSE"]),
            "R2": float(high_row["R2"]),
        },
    }


def load_regression_results() -> dict:
    manifest = _load_manifest()
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["order_purchase_timestamp"])
    final_metrics = manifest["test_metrics"]

    version_comparison = [
        {
            "version": "Sin clasificación",
            "enfoque": "Un regresor para todos los fletes",
            "MAE": 5.53,
            "RMSE": 14.86,
            "R2": 0.668,
            "WAPE": 22.32,
            "MAE_fletes_altos": 18.04,
            "sesgo_fletes_altos": 15.93,
        },
        {
            "version": "Con clasificación",
            "enfoque": "Clasifica y usa un modelo especializado",
            "MAE": float(final_metrics["MAE"]),
            "RMSE": float(final_metrics["RMSE"]),
            "R2": float(final_metrics["R2"]),
            "WAPE": float(final_metrics["WAPE"]),
            "MAE_fletes_altos": float(final_metrics["MAE_fletes_altos"]),
            "sesgo_fletes_altos": float(final_metrics["sesgo_fletes_altos"]),
        },
    ]

    predictions["rango_flete"] = pd.cut(
        predictions["freight_total"],
        bins=[-np.inf, 15, 30, 60, np.inf],
        labels=["Hasta R$15", "R$15–30", "R$30–60", "Más de R$60"],
    )
    error_by_range = (
        predictions.groupby("rango_flete", observed=False)
        .agg(
            pedidos=("order_id", "count"),
            flete_promedio=("freight_total", "mean"),
            MAE=("absolute_error", "mean"),
        )
        .reset_index()
    )
    error_by_range["rango_flete"] = error_by_range["rango_flete"].astype(str)

    visible_limit = predictions["freight_total"].quantile(0.99)
    visible = predictions.loc[predictions["freight_total"].le(visible_limit)]
    step = max(1, len(visible) // 700)
    sample = visible.iloc[::step, :].head(700)

    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(10)
    high_importance = pd.read_csv(HIGH_FEATURE_IMPORTANCE_PATH).head(10)
    return {
        "manifest": manifest,
        "metrics": pd.read_csv(METRICS_PATH).round(4).to_dict(orient="records"),
        "version_comparison": version_comparison,
        "component_metrics": _selected_component_metrics(),
        "prediction_sample": sample[[
            "freight_total", "prediction", "absolute_error", "segmento_predicho"
        ]].round(3).to_dict(orient="records"),
        "error_by_range": error_by_range.round(3).to_dict(orient="records"),
        "feature_importance": importance.round(4).to_dict(orient="records"),
        "high_feature_importance": high_importance.round(4).to_dict(orient="records"),
        "states": _available_states(),
        "categories": _available_categories(),
        "test_orders": int(len(predictions)),
    }
