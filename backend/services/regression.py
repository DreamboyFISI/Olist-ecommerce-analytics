import json
from functools import lru_cache
from pathlib import Path

import joblib
import holidays
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "regression"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "inference_metadata.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.csv"
PREDICTIONS_PATH = ARTIFACT_DIR / "predictions.csv"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"

REGIONS_BY_STATE = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte",
    "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste",
    "MA": "Nordeste", "PB": "Nordeste", "PE": "Nordeste",
    "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sur", "RS": "Sur", "SC": "Sur",
}


@lru_cache(maxsize=1)
def _load_model():
    pipeline = joblib.load(MODEL_PATH)
    model = getattr(pipeline, "named_steps", {}).get("model")
    if model is not None and hasattr(model, "set_params"):
        model.set_params(device="cpu")
    return pipeline


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    with METADATA_PATH.open(encoding="utf-8") as source:
        return json.load(source)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _working_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end <= start:
        raise ValueError("La fecha prometida debe ser posterior a la fecha de compra.")
    brazil_holidays = holidays.Brazil(years=range(start.year, end.year + 1))
    dates = pd.date_range(start=start.date(), end=end.date(), freq="D")
    business_days = sum(
        day.dayofweek < 5 and day.date() not in brazil_holidays
        for day in dates
    )
    return max(0, business_days - 1)


def _next_business_day(value: pd.Timestamp) -> pd.Timestamp | None:
    """Devuelve el siguiente día operativo solo si la fecha original no lo es."""
    brazil_holidays = holidays.Brazil(years=[value.year, value.year + 1])
    adjusted = value
    while adjusted.dayofweek >= 5 or adjusted.date() in brazil_holidays:
        adjusted += pd.Timedelta(days=1)
    return adjusted if adjusted.date() != value.date() else None


@lru_cache(maxsize=1)
def _prediction_interval_margin() -> float:
    predictions = pd.read_csv(PREDICTIONS_PATH, usecols=["absolute_error"])
    return float(predictions["absolute_error"].quantile(0.80))


def build_feature_frame(payload: dict) -> pd.DataFrame:
    metadata = _load_metadata()
    purchased_at = pd.Timestamp(payload["purchase_timestamp"])
    customer_state = payload["customer_state"].strip().upper()
    seller_state = payload["main_seller_state"].strip().upper()
    category = payload["main_category"].strip().lower()
    state_pair = f"{customer_state}_{seller_state}"
    customer_region = REGIONS_BY_STATE.get(customer_state, "desconocido")
    seller_region = REGIONS_BY_STATE.get(seller_state, "desconocido")
    promised_at = pd.Timestamp(payload["estimated_delivery_date"])
    brazil_holidays = holidays.Brazil(years=[purchased_at.year])

    row = {
        "n_items": payload["n_items"],
        "n_products": payload["n_products"],
        "n_sellers": payload["n_sellers"],
        "price_total": payload["price_total"],
        "freight_total": payload["freight_total"],
        "weight_total_g": payload["weight_total_g"],
        "volume_total_cm3": payload["volume_total_cm3"],
        "purchase_year": purchased_at.year,
        "purchase_month": purchased_at.month,
        "purchase_dayofweek": purchased_at.dayofweek,
        "purchase_hour": purchased_at.hour,
        "is_weekend": int(purchased_at.dayofweek >= 5),
        "same_state": int(customer_state == seller_state),
        "customer_state": customer_state,
        "main_seller_state": seller_state,
        "main_category": category,
        "distance_km": payload["distance_km"],
        "freight_ratio": _safe_divide(payload["freight_total"], payload["price_total"] + 1),
        "volumetric_density": _safe_divide(payload["weight_total_g"], payload["volume_total_cm3"] + 1),
        "distance_per_freight": _safe_divide(payload["distance_km"], payload["freight_total"] + 1),
        "items_per_seller": _safe_divide(payload["n_items"], payload["n_sellers"]),
        "purchase_is_holiday": int(purchased_at.date() in brazil_holidays),
        "estimated_working_days": _working_days_between(purchased_at, promised_at),
        "is_interregional": int(customer_region != seller_region),
        "geo_state_pair_delivery_days_encoded": metadata["state_pair_target_map"].get(state_pair, metadata["global_median"]),
        "seller_state_latency": metadata["seller_latency_map"].get(seller_state, metadata["global_latency_mean"]),
    }
    return pd.DataFrame([row], columns=metadata["features"])


def predict_delivery(payload: dict) -> dict:
    features = build_feature_frame(payload)
    log_prediction = float(_load_model().predict(features)[0])
    predicted_days = max(0.0, float(np.expm1(log_prediction)))
    purchased_at = pd.Timestamp(payload["purchase_timestamp"])
    arrival_at = purchased_at + pd.to_timedelta(predicted_days, unit="D")
    operational_arrival = _next_business_day(arrival_at)
    metadata = _load_metadata()
    with MANIFEST_PATH.open(encoding="utf-8") as source:
        manifest = json.load(source)
    margin = _prediction_interval_margin()
    lower_days = max(0.0, predicted_days - margin)
    upper_days = predicted_days + margin
    return {
        "model": manifest["winner"],
        "version": manifest["version"],
        "predicted_days": round(predicted_days, 2),
        "estimated_arrival": arrival_at.isoformat(),
        "operational_arrival": operational_arrival.isoformat() if operational_arrival is not None else None,
        "interval": {
            "coverage": 80,
            "lower_days": round(lower_days, 2),
            "upper_days": round(upper_days, 2),
            "earliest_arrival": (purchased_at + pd.to_timedelta(lower_days, unit="D")).isoformat(),
            "latest_arrival": (purchased_at + pd.to_timedelta(upper_days, unit="D")).isoformat(),
        },
        "baseline_days": round(float(metadata["global_median"]), 2),
        "difference_vs_baseline": round(predicted_days - float(metadata["global_median"]), 2),
    }


def load_regression_results() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as source:
        manifest = json.load(source)
    metrics = pd.read_csv(METRICS_PATH).round(4).to_dict(orient="records")
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["order_purchase_timestamp"])
    step = max(1, len(predictions) // 500)
    sample = predictions.iloc[::step, :].head(500)
    error_bins = pd.cut(
        predictions["absolute_error"],
        bins=[0, 1, 2, 3, 5, 7, 10, float("inf")],
        labels=["0–1", "1–2", "2–3", "3–5", "5–7", "7–10", "10+"],
        right=False,
    )
    metadata = _load_metadata()
    return {
        "manifest": manifest,
        "metrics": metrics,
        "prediction_sample": sample[["delivery_days", "prediction", "absolute_error"]].round(3).to_dict(orient="records"),
        "error_distribution": [
            {"range": str(label), "orders": int(count)}
            for label, count in error_bins.value_counts(sort=False).items()
        ],
        "states": metadata["states"],
        "categories": metadata["categories"],
        "test_orders": int(len(predictions)),
    }
