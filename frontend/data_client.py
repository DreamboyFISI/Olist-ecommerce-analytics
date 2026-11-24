import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("OLIST_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _local_json(relative_path: str):
    with (ROOT / relative_path).open(encoding="utf-8") as source:
        return json.load(source)


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str):
    with urlopen(f"{API_URL}{path}", timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post(path: str, payload: dict):
    request = Request(
        f"{API_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_health() -> tuple[bool, str]:
    try:
        payload = api_get("/health")
        return payload.get("status") == "ok", "API conectada"
    except (HTTPError, URLError, TimeoutError, ValueError):
        return False, "Modo local"


def get_eda_summary() -> tuple[dict, str]:
    try:
        return api_get("/api/v1/eda/summary"), "API"
    except (HTTPError, URLError, TimeoutError, ValueError):
        return _local_json("docs/hallazgos_eda.json"), "archivo local"


def get_modules() -> list[dict]:
    try:
        return api_get("/api/v1/modules")
    except (HTTPError, URLError, TimeoutError, ValueError):
        from backend.services.repository import load_module_registry
        return load_module_registry()


@st.cache_data(ttl=60, show_spinner=False)
def get_regression_results() -> dict:
    return api_get("/api/v1/regression/results")


def predict_regression(payload: dict) -> dict:
    return api_post("/api/v1/regression/predict", payload)


@st.cache_data(ttl=300, show_spinner=False)
def get_location_cities(role: str, state: str) -> list[str]:
    query = urlencode({"role": role, "state": state})
    return api_get(f"/api/v1/regression/locations/cities?{query}")


@st.cache_data(ttl=300, show_spinner=False)
def get_locations(role: str, state: str, city: str) -> list[dict]:
    query = urlencode({"role": role, "state": state, "city": city})
    return api_get(f"/api/v1/regression/locations?{query}")


def resolve_location_state(lat: float, lng: float) -> dict:
    query = urlencode({"lat": lat, "lng": lng})
    return api_get(f"/api/v1/regression/locations/resolve-state?{query}")


@st.cache_data(ttl=60, show_spinner=False)
def get_classification_results() -> dict:
    try:
        return api_get("/api/v1/classification/results")
    except (HTTPError, URLError, TimeoutError, ValueError):
        from backend.services.classification import load_classification_results
        return load_classification_results()


def predict_classification(payload: dict) -> dict:
    try:
        return api_post("/api/v1/classification/predict", payload)
    except (HTTPError, URLError, TimeoutError):
        from backend.services.classification import predict_risk
        return predict_risk(payload)


def explain_classification(payload: dict, method: str = "shap") -> dict:
    try:
        return api_post(f"/api/v1/classification/explain?method={method}", payload)
    except (HTTPError, URLError, TimeoutError):
        from backend.services.classification import explain_prediction
        return explain_prediction(payload, prefer=method)


@st.cache_data(ttl=60, show_spinner=False)
def get_monitoring_report(refresh: bool = False) -> dict:
    try:
        return api_get(f"/api/v1/monitoring/report?refresh={'true' if refresh else 'false'}")
    except (HTTPError, URLError, TimeoutError, ValueError):
        from backend.services.monitoring import load_monitoring_report
        return load_monitoring_report(refresh=refresh)


@st.cache_data(ttl=60, show_spinner=False)
def get_nlp_results() -> dict:
    return api_get("/api/v1/nlp/results")


def predict_nlp(text: str) -> dict:
    return api_post("/api/v1/nlp/predict", {"text": text})
