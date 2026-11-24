"""Pruebas del monitoreo: deriva, desempeño por periodo, registro, bitácora y plan de reentrenamiento."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring import monitor
from src.monitoring.drift import categorical_shift, drift_status, psi, relative_change
from src.monitoring.monitor import ARTIFACTS_DIR, build_registry, build_report, evaluate_module, run
from src.monitoring.retrain import backup_manifest, execute_plan, plan_retraining

ARTIFACTS_AVAILABLE = (ARTIFACTS_DIR / "classification" / "results.csv").exists()


# ---------------------------------------------------------------------------
# Medidas de deriva
# ---------------------------------------------------------------------------
def test_psi_es_cero_para_la_misma_distribucion():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_crece_cuando_la_distribucion_se_mueve():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    poco = rng.normal(0.1, 1, 5000)
    mucho = rng.normal(1.5, 1, 5000)
    assert psi(ref, poco) < 0.10
    assert psi(ref, mucho) > 0.25
    assert drift_status(psi(ref, mucho)) == "Alerta"


def test_psi_con_referencia_constante_no_explota():
    assert psi([1, 1, 1, 1], [1, 1, 1]) == 0.0
    assert np.isinf(psi([1, 1, 1, 1], [2, 2, 2]))


def test_cambio_categorico_y_relativo():
    assert categorical_shift(["a", "a", "b"], ["a", "a", "b"]) == pytest.approx(0.0)
    assert categorical_shift(["a"] * 10, ["b"] * 10) == pytest.approx(1.0)
    assert relative_change(10, 12) == pytest.approx(0.2)
    assert np.isnan(relative_change(0, 5))
    assert drift_status(float("nan")) == "Sin datos"


# ---------------------------------------------------------------------------
# Evaluación de un módulo con artefactos sintéticos
# ---------------------------------------------------------------------------
def _artefactos_sinteticos(tmp_path: Path, degradar: bool) -> Path:
    """Simula artifacts/classification con un modelo que se degrada (o no) en los últimos meses."""
    rng = np.random.default_rng(7)
    carpeta = tmp_path / "classification"
    carpeta.mkdir(parents=True)
    fechas = pd.date_range("2018-03-01", "2018-08-31", freq="D")
    filas = []
    for fecha in fechas:
        for _ in range(30):
            y = int(rng.random() < 0.15)
            # Un modelo útil separa clases; si se degrada, en los últimos meses predice ruido
            ruido = degradar and fecha >= pd.Timestamp("2018-06-01")
            p = rng.random() if ruido else np.clip(0.1 + 0.6 * y + rng.normal(0, 0.15), 0, 1)
            filas.append({"order_id": f"o{len(filas)}", "fecha_compra": fecha.date(), "clase_real": y,
                          "probabilidad": round(float(p), 4), "prediccion": int(p >= 0.3)})
    pd.DataFrame(filas).to_csv(carpeta / "results.csv", index=False)
    (carpeta / "manifest.json").write_text(json.dumps({
        "status": "Completado", "model_name": "demo", "version": "9.9.9", "generated_at": "2026-11-01T00:00:00+00:00",
        "dataset": "demo", "period": "demo", "metrics": {"pr_auc": 0.80, "prevalencia_test": 0.15}, "winner": "demo",
        "results_path": "x", "chart_data_path": "x", "business_conclusions": [], "limitations": [],
    }), encoding="utf-8")
    return tmp_path


def test_modulo_estable_queda_ok(tmp_path):
    artefactos = _artefactos_sinteticos(tmp_path, degradar=False)
    r = evaluate_module("classification", artefactos)
    assert r["n_predicciones"] == 30 * 184
    assert len(r["periodos"]) == 6
    assert r["estado"] == "OK", r["motivos"]


def test_modulo_degradado_se_marca(tmp_path):
    artefactos = _artefactos_sinteticos(tmp_path, degradar=True)
    r = evaluate_module("classification", artefactos)
    assert r["estado"] in ("Vigilar", "Alerta")
    assert any("degrad" in m or "PSI" in m or "empeor" in m for m in r["motivos"])


def test_modulo_sin_artefactos_informa_sin_datos(tmp_path):
    r = evaluate_module("regression", tmp_path)
    assert r["estado"] == "Sin datos" and r["n_predicciones"] == 0


def test_reporte_registro_y_bitacora_en_carpeta_temporal(tmp_path):
    artefactos = _artefactos_sinteticos(tmp_path, degradar=True)
    destino = tmp_path / "monitoring"
    report = run(artifacts_dir=artefactos, write=True, monitoring_dir=destino)
    assert set(report["resumen"]) == {"OK", "Vigilar", "Alerta", "Sin datos"}
    assert (destino / "report.json").exists() and (destino / "model_registry.json").exists()
    bitacora = pd.read_csv(destino / "audit_log.csv")
    assert len(bitacora) == len(report["modulos"])
    # Una segunda corrida agrega filas, nunca sobreescribe
    run(artifacts_dir=artefactos, write=True, monitoring_dir=destino)
    assert len(pd.read_csv(destino / "audit_log.csv")) == 2 * len(report["modulos"])
    assert (destino / "monitoring.log").read_text(encoding="utf-8").count("classification") >= 2


def test_registro_detecta_antiguedad(tmp_path):
    artefactos = _artefactos_sinteticos(tmp_path, degradar=False)
    reciente = build_registry(artefactos, now=pd.Timestamp("2026-11-15T00:00:00+00:00").to_pydatetime())
    viejo = build_registry(artefactos, now=pd.Timestamp("2027-06-01T00:00:00+00:00").to_pydatetime())
    clasif_reciente = next(r for r in reciente if r["modulo"] == "classification")
    clasif_viejo = next(r for r in viejo if r["modulo"] == "classification")
    assert clasif_reciente["antiguedad"] == "OK" and clasif_viejo["antiguedad"] == "Vigilar"
    assert clasif_reciente["artefactos"][0]["sha256"]


def test_plan_de_reentrenamiento_respeta_la_regla(tmp_path):
    report = {"modulos": [
        {"modulo": "classification", "estado": "Alerta", "version": "1", "motivos": ["x"]},
        {"modulo": "nlp", "estado": "Vigilar", "version": "1", "motivos": ["y"]},
        {"modulo": "graphs", "estado": "OK", "version": "1", "motivos": []},
        {"modulo": "regression", "estado": "Alerta", "version": "3", "motivos": ["z"]},
    ]}
    plan = plan_retraining(report)
    assert [p["modulo"] for p in plan] == ["classification", "regression"]
    assert plan[0]["ejecutable"] and not plan[1]["ejecutable"]
    assert len(plan_retraining(report, include_watch=True)) == 3
    resultados = execute_plan(plan, dry_run=True)
    assert all(not r["ejecutado"] for r in resultados)


def test_respaldo_del_manifest(tmp_path):
    artefactos = _artefactos_sinteticos(tmp_path, degradar=False)
    respaldo = backup_manifest("classification", artefactos, tmp_path / "backups")
    assert respaldo is not None and respaldo.exists() and "9.9.9" in respaldo.name


@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason="faltan artefactos reales")
def test_reporte_con_los_artefactos_del_repositorio():
    report = build_report()
    modulos = {m["modulo"] for m in report["modulos"]}
    assert {"classification", "regression", "nlp", "timeseries", "clustering", "graphs", "association"} <= modulos
    clasif = next(m for m in report["modulos"] if m["modulo"] == "classification")
    assert clasif["n_predicciones"] > 0 and clasif["actual"] is not None
