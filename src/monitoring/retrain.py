"""Pipeline de reentrenamiento guiado por el reporte de monitoreo.

Sin orquestador externo: la regla es explícita y reproducible.

- Un módulo en **Alerta** se reentrena (``python -m src.models.<modulo>``).
- Un módulo en **Vigilar** se reporta y queda a criterio del responsable.
- Antes de reentrenar se respalda el manifest vigente en
  ``artifacts/monitoring/backups/<modulo>_<version>_<fecha>.json`` para dejar
  rastro del cambio de versión (auditoría).

Uso:
    python -m src.monitoring --retrain          # ejecuta lo que esté en alerta
    python -m src.monitoring --retrain --dry-run  # solo muestra el plan
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
BACKUP_DIR = ARTIFACTS_DIR / "monitoring" / "backups"

RETRAIN_COMMANDS: dict[str, list[str]] = {
    "classification": [sys.executable, "-m", "src.models.classification"],
    "regression": [sys.executable, "notebooks/07_regression/07-regression-fletev3.ipynb"],
    "nlp": [sys.executable, "-m", "src.models.sentiment"],
    "timeseries": [sys.executable, "-c", "from src.models.timeseries import generar_artefactos; generar_artefactos()"],
    "clustering": [sys.executable, "-m", "src.models.clustering"],
    "graphs": [sys.executable, "-m", "src.models.graphs", "--refresh"],
    "association": [sys.executable, "-m", "src.models.association"],
}
# Módulos cuyo entrenamiento vive solo en un notebook: se informa, no se ejecuta.
NOTEBOOK_ONLY = {"regression"}


def plan_retraining(report: dict, include_watch: bool = False) -> list[dict]:
    """Lista de acciones por módulo según el estado del reporte."""
    plan = []
    for m in report.get("modulos", []):
        estado = m.get("estado")
        if estado == "Alerta" or (include_watch and estado == "Vigilar"):
            slug = m["modulo"]
            plan.append({
                "modulo": slug, "estado": estado, "version_actual": m.get("version"),
                "comando": " ".join(RETRAIN_COMMANDS.get(slug, [])),
                "ejecutable": slug in RETRAIN_COMMANDS and slug not in NOTEBOOK_ONLY,
                "motivos": m.get("motivos", []),
            })
    return plan


def backup_manifest(slug: str, artifacts_dir: Path = ARTIFACTS_DIR, backup_dir: Path = BACKUP_DIR) -> Path | None:
    manifest = artifacts_dir / slug / "manifest.json"
    if not manifest.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    try:
        version = json.loads(manifest.read_text(encoding="utf-8")).get("version", "sin-version")
    except json.JSONDecodeError:
        version = "sin-version"
    destino = backup_dir / f"{slug}_{version}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    shutil.copy2(manifest, destino)
    return destino


def execute_plan(plan: list[dict], dry_run: bool = True, artifacts_dir: Path = ARTIFACTS_DIR) -> list[dict]:
    resultados = []
    for accion in plan:
        registro = {**accion, "ejecutado": False, "codigo_salida": None, "respaldo": None}
        if not accion["ejecutable"]:
            registro["nota"] = "Entrenamiento en notebook: ejecutarlo manualmente y volver a correr el monitoreo."
        elif dry_run:
            registro["nota"] = "Simulación: no se ejecutó."
        else:
            respaldo = backup_manifest(accion["modulo"], artifacts_dir)
            registro["respaldo"] = str(respaldo) if respaldo else None
            proceso = subprocess.run(RETRAIN_COMMANDS[accion["modulo"]], cwd=ROOT)
            registro["ejecutado"] = True
            registro["codigo_salida"] = proceso.returncode
        resultados.append(registro)
    return resultados
