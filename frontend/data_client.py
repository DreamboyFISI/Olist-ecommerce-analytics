import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

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
