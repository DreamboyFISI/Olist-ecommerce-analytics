"""Carga los artefactos de los módulos y el monitoreo en la base de datos.

Tablas que crea o reemplaza:

- ``model_registry``      — una fila por módulo (versión, fecha, ganador, métricas en JSON).
- ``module_metrics``      — métricas del manifiesto en formato largo (modulo, metrica, valor).
- ``predictions_<slug>``  — las predicciones/resultados que cada módulo dejó en sus artefactos.
- ``monitoring_report``   — el último reporte de monitoreo por módulo.
- ``audit_log``           — la bitácora completa de corridas de monitoreo.

Uso:
    python scripts/load_artifacts_to_db.py                       # SQLite local o DATABASE_URL
    DATABASE_URL=postgresql+psycopg2://u:p@localhost:5432/datasmart python scripts/load_artifacts_to_db.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.db import get_engine  # noqa: E402
from src.monitoring.monitor import AUDIT_LOG_PATH, REPORT_PATH, build_registry  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
PREDICTION_FILES = {
    "classification": "results.csv", "regression": "predictions.csv", "nlp": "results.csv",
    "timeseries": "forecast.csv", "clustering": "results.csv", "association": "results.csv",
}


def load_all(url: str | None = None, artifacts_dir: Path = ARTIFACTS, max_rows: int | None = None) -> dict[str, int]:
    """Carga todo y devuelve el número de filas escritas por tabla."""
    engine = get_engine(url)
    cargado_en = datetime.now(timezone.utc).isoformat(timespec="seconds")
    filas: dict[str, int] = {}

    registro = build_registry(artifacts_dir)
    tabla_registro = pd.DataFrame([{
        "modulo": r["modulo"], "titulo": r["titulo"], "modelo": r.get("modelo"), "version": r.get("version"),
        "generado": r.get("generado"), "edad_dias": r.get("edad_dias"), "antiguedad": r["antiguedad"],
        "estado_manifest": r["estado_manifest"], "metricas_json": json.dumps(r["metricas"], ensure_ascii=False),
        "artefactos_json": json.dumps(r["artefactos"], ensure_ascii=False), "cargado_en": cargado_en,
    } for r in registro])
    tabla_registro.to_sql("model_registry", engine, if_exists="replace", index=False)
    filas["model_registry"] = len(tabla_registro)

    largo = [{"modulo": r["modulo"], "version": r.get("version"), "metrica": k, "valor": float(v), "cargado_en": cargado_en}
             for r in registro for k, v in r["metricas"].items()]
    tabla_metricas = pd.DataFrame(largo, columns=["modulo", "version", "metrica", "valor", "cargado_en"])
    tabla_metricas.to_sql("module_metrics", engine, if_exists="replace", index=False)
    filas["module_metrics"] = len(tabla_metricas)

    for slug, archivo in PREDICTION_FILES.items():
        ruta = artifacts_dir / slug / archivo
        if not ruta.exists():
            continue
        df = pd.read_csv(ruta)
        if max_rows:
            df = df.head(max_rows)
        df["cargado_en"] = cargado_en
        df.to_sql(f"predictions_{slug}", engine, if_exists="replace", index=False, chunksize=5000)
        filas[f"predictions_{slug}"] = len(df)

    if REPORT_PATH.exists():
        reporte = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        tabla_reporte = pd.DataFrame([{
            "generated_at": reporte["generated_at"], "modulo": m["modulo"], "titulo": m["titulo"],
            "version": m.get("version"), "metrica": m.get("metrica"), "referencia": m.get("referencia"),
            "actual": m.get("actual"), "cambio_relativo": m.get("cambio_relativo"), "estado": m["estado"],
            "motivos": " | ".join(m.get("motivos", [])), "drift_json": json.dumps(m.get("drift", {}), ensure_ascii=False),
        } for m in reporte["modulos"]])
        tabla_reporte.to_sql("monitoring_report", engine, if_exists="replace", index=False)
        filas["monitoring_report"] = len(tabla_reporte)

    if AUDIT_LOG_PATH.exists():
        auditoria = pd.read_csv(AUDIT_LOG_PATH)
        auditoria.to_sql("audit_log", engine, if_exists="replace", index=False)
        filas["audit_log"] = len(auditoria)
    return filas


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga artefactos y monitoreo en la base de datos.")
    parser.add_argument("--url", default=None, help="URL SQLAlchemy; por defecto DATABASE_URL o SQLite local.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limitar filas por tabla de predicciones.")
    args = parser.parse_args()
    filas = load_all(args.url, max_rows=args.max_rows)
    for tabla, n in filas.items():
        print(f"{tabla:28s} {n:>8,} filas")


if __name__ == "__main__":
    main()
