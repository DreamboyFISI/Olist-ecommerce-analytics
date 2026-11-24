"""Pruebas de la carga de artefactos a la base de datos (con SQLite, sin PostgreSQL)."""

import json

import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")

from backend.services.db import database_status, get_engine  # noqa: E402
from scripts.load_artifacts_to_db import load_all  # noqa: E402


def _artefactos(tmp_path):
    carpeta = tmp_path / "classification"
    carpeta.mkdir()
    pd.DataFrame({"order_id": ["a", "b"], "fecha_compra": ["2018-06-01", "2018-07-01"], "clase_real": [0, 1],
                  "probabilidad": [0.1, 0.8], "prediccion": [0, 1]}).to_csv(carpeta / "results.csv", index=False)
    (carpeta / "manifest.json").write_text(json.dumps({
        "status": "Completado", "model_name": "demo", "version": "1.2.3", "generated_at": "2026-11-01T00:00:00+00:00",
        "dataset": "demo", "period": "demo", "metrics": {"pr_auc": 0.4, "recall": 0.3}, "winner": "demo",
        "results_path": "x", "chart_data_path": "x", "business_conclusions": [], "limitations": []}), encoding="utf-8")
    return tmp_path


def test_carga_en_sqlite_crea_las_tablas(tmp_path):
    url = f"sqlite:///{(tmp_path / 'prueba.db').as_posix()}"
    filas = load_all(url, artifacts_dir=_artefactos(tmp_path))
    assert filas["model_registry"] == 7            # un registro por módulo, con o sin manifest
    assert filas["module_metrics"] == 2
    assert filas["predictions_classification"] == 2
    estado = database_status(url)
    assert estado["ok"] and {"model_registry", "module_metrics", "predictions_classification"} <= set(estado["tablas"])
    registro = pd.read_sql("SELECT modulo, version FROM model_registry WHERE modulo = 'classification'", get_engine(url))
    assert registro.version.iat[0] == "1.2.3"


def test_estado_de_base_inaccesible_no_lanza():
    estado = database_status("postgresql+psycopg2://nadie:nada@127.0.0.1:1/inexistente")
    assert estado["ok"] is False and "error" in estado
