"""Servicio de consulta del monitoreo de modelos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.monitoring.monitor import AUDIT_LOG_PATH, REPORT_PATH, load_report, run
from src.monitoring.retrain import plan_retraining

ROOT = Path(__file__).resolve().parents[2]


def load_monitoring_report(refresh: bool = False, max_audit_rows: int = 300) -> dict:
    """Reporte consolidado + plan de reentrenamiento + últimas filas de auditoría.

    ``refresh=True`` recalcula el reporte y deja rastro en la bitácora; por
    defecto se sirve el último reporte escrito por ``python -m src.monitoring``.
    """
    report = run(write=True) if refresh else load_report()
    if report is None:
        report = run(write=True)
    if report is None:
        raise FileNotFoundError(REPORT_PATH)
    auditoria = []
    if AUDIT_LOG_PATH.exists():
        log = pd.read_csv(AUDIT_LOG_PATH)
        auditoria = log.tail(max_audit_rows).where(pd.notna(log.tail(max_audit_rows)), None).to_dict(orient="records")
    return {
        **report,
        "plan_reentrenamiento": plan_retraining(report, include_watch=True),
        "auditoria": auditoria,
    }
