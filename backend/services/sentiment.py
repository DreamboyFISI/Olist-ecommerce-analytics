"""Servicio de inferencia y consulta del módulo de sentimientos."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.sentiment import LABEL_NAMES, clean_text

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "nlp"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
CHART_DATA_PATH = ARTIFACT_DIR / "chart_data.json"
RESULTS_PATH = ARTIFACT_DIR / "results.csv"


@lru_cache(maxsize=1)
def _load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No se encontró artifacts/nlp/model.joblib.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No se encontró artifacts/nlp/manifest.json.")
    with MANIFEST_PATH.open(encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def _load_chart_data() -> dict:
    if not CHART_DATA_PATH.exists():
        raise FileNotFoundError("No se encontró artifacts/nlp/chart_data.json.")
    with CHART_DATA_PATH.open(encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def _load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("No se encontró artifacts/nlp/results.csv.")
    return pd.read_csv(RESULTS_PATH)


def load_sentiment_results() -> dict:
    """Devuelve manifest, datos de gráficas y una muestra de predicciones."""
    results = _load_results()
    return {
        "manifest": _load_manifest(),
        "chart_data": _load_chart_data(),
        "results": results.head(200).where(pd.notna(results), None).to_dict(orient="records"),
    }


def predict_sentiment(text: str) -> dict:
    """Clasifica una reseña en negativo, neutro o positivo.

    El pipeline incluye la vectorización TF-IDF, así que aquí solo se limpia
    el texto y se llama a `predict_proba`. Es inferencia pura: no se entrena
    nada desde Streamlit ni desde la API.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("La reseña no puede estar vacía.")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("La reseña no contiene texto analizable.")

    payload = _load_model()
    pipeline = payload["pipeline"]
    probabilities = np.asarray(pipeline.predict_proba([cleaned])[0], dtype=float)
    class_index = int(np.argmax(probabilities))
    labels = payload.get("labels", [0, 1, 2])
    label_id = int(labels[class_index]) if class_index < len(labels) else class_index
    label_name = LABEL_NAMES.get(label_id, str(label_id))

    return {
        "texto": text,
        "texto_limpio": cleaned,
        "sentimiento": label_name,
        "clase": label_id,
        "confianza": float(probabilities[class_index]),
        "probabilidades": {
            LABEL_NAMES[index]: float(value)
            for index, value in enumerate(probabilities[: len(LABEL_NAMES)])
        },
        "modelo": payload.get("winner", payload.get("model_name", "Modelo de sentimientos")),
        "version": payload.get("version"),
        "umbral_negativo": payload.get("negative_threshold"),
    }
