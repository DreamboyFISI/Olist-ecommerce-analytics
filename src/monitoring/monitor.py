"""Monitoreo y mantenimiento de los modelos de DataSmart (sin orquestador externo).

Qué vigila
----------
1. **Desempeño en el tiempo.** Cada módulo deja en sus artefactos predicciones
   con fecha (``results.csv`` o ``predictions.csv``). Se recalcula la métrica
   principal por periodo (mes) y se compara con el valor del manifiesto.
2. **Deriva (drift).** PSI de las probabilidades o errores por periodo frente al
   primer periodo de prueba, y cambio relativo de la tasa objetivo.
3. **Versionado y antigüedad.** Registro de cada modelo (nombre, versión, fecha,
   métricas y huella SHA-256 de sus artefactos) y aviso cuando el artefacto
   supera la antigüedad máxima.
4. **Auditoría.** Cada corrida agrega filas a ``artifacts/monitoring/audit_log.csv``
   y escribe ``artifacts/monitoring/monitoring.log``.

El reporte consolidado queda en ``artifacts/monitoring/report.json`` y lo
consumen la API (``GET /api/v1/monitoring/report``) y la página Monitoreo.
La decisión de reentrenar la toma ``src.monitoring.retrain`` a partir de los
estados de este reporte.

Punto de entrada:
    python -m src.monitoring            # genera el reporte y registra la corrida
    python -m src.monitoring --retrain  # además ejecuta los pipelines en alerta
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.drift import drift_status, psi, relative_change

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
MONITORING_DIR = ARTIFACTS_DIR / "monitoring"
REPORT_PATH = MONITORING_DIR / "report.json"
REGISTRY_PATH = MONITORING_DIR / "model_registry.json"
AUDIT_LOG_PATH = MONITORING_DIR / "audit_log.csv"
LOG_PATH = MONITORING_DIR / "monitoring.log"

MAX_AGE_DAYS = 90            # un modelo más viejo que esto se marca para revisión
DEGRADACION_VIGILAR = 0.10   # caída relativa de la métrica principal
DEGRADACION_ALERTA = 0.20
MIN_PERIODO = 30             # filas mínimas para evaluar un periodo
MIN_DRIFT = 200              # filas mínimas por periodo para medir deriva (el PSI es ruidoso con pocas)

logger = logging.getLogger("datasmart.monitoring")
logger.setLevel(logging.INFO)

ESTADOS = ["OK", "Vigilar", "Alerta", "Sin datos"]


# ---------------------------------------------------------------------------
# Especificación por módulo: dónde están las predicciones y cómo se evalúan
# ---------------------------------------------------------------------------
def _pr_auc(df: pd.DataFrame) -> float:
    from sklearn.metrics import average_precision_score
    if df.clase_real.nunique() < 2:
        return float("nan")
    return float(average_precision_score(df.clase_real, df.probabilidad))


def _mae(df: pd.DataFrame) -> float:
    return float(np.mean(np.abs(df.real - df.prediccion)))


def _macro_f1(df: pd.DataFrame) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(df.label, df.label_pred, average="macro"))


MODULES: dict[str, dict[str, Any]] = {
    "classification": {
        "titulo": "Clasificación",
        "archivo": "results.csv",
        "fecha": "fecha_compra",
        "metrica": "PR-AUC",
        "mayor_es_mejor": True,
        "manifest_metric": ("metrics", "pr_auc"),
        "score": "probabilidad",
        "objetivo": "clase_real",
        "evaluar": _pr_auc,
    },
    "regression": {
        "titulo": "Regresión",
        "archivo": "predictions.csv",
        "fecha": "order_purchase_timestamp",
        "metrica": "MAE",
        "mayor_es_mejor": False,
        "manifest_metric": ("test_metrics", "MAE"),
        "score": "absolute_error",
        "objetivo": None,
        "renombrar": {"freight_total": "real", "prediction": "prediccion"},
        "evaluar": _mae,
    },
    "nlp": {
        "titulo": "Sentimientos",
        "archivo": "results.csv",
        "fecha": "review_creation_date",
        "metrica": "macro-F1",
        "mayor_es_mejor": True,
        "manifest_metric": ("metrics", "macro_f1"),
        "score": "prob_negativo",
        "objetivo": "label",
        "evaluar": _macro_f1,
    },
    "timeseries": {
        "titulo": "Series temporales",
        "archivo": "forecast.csv",
        "fecha": "fecha",
        "metrica": "MAE",
        "mayor_es_mejor": False,
        "manifest_metric": ("metrics", "MAE"),
        "score": "error",
        "objetivo": None,
        "renombrar": {"pronostico": "prediccion"},
        "filtro": lambda df: df[df.real.notna()],
        "evaluar": _mae,
        "periodo": "W",
    },
}
# Módulos sin dimensión temporal en sus artefactos: solo versionado y antigüedad.
STATIC_MODULES = {"clustering": "Clustering", "graphs": "Grafos", "association": "Asociación"}


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _get_nested(d: dict | None, keys: tuple[str, ...]):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _parse_date(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.to_pydatetime()


# ---------------------------------------------------------------------------
# Registro de modelos (versionado)
# ---------------------------------------------------------------------------
def build_registry(artifacts_dir: Path = ARTIFACTS_DIR, now: datetime | None = None) -> list[dict]:
    """Un registro por módulo con versión, fecha, métricas y huella de artefactos."""
    now = now or datetime.now(timezone.utc)
    registro = []
    for slug in list(MODULES) + list(STATIC_MODULES):
        carpeta = artifacts_dir / slug
        manifest = _read_json(carpeta / "manifest.json")
        archivos = sorted(p for p in carpeta.glob("*") if p.is_file() and p.name != ".gitkeep") if carpeta.exists() else []
        generado = _parse_date(manifest.get("generated_at")) if manifest else None
        edad = (now - generado).days if generado else None
        metricas = (manifest or {}).get("metrics") or (manifest or {}).get("test_metrics") or {}
        registro.append({
            "modulo": slug,
            "titulo": MODULES.get(slug, {}).get("titulo") or STATIC_MODULES.get(slug, slug),
            "estado_manifest": (manifest or {}).get("status", "Pendiente"),
            "modelo": (manifest or {}).get("model_name") or (manifest or {}).get("winner"),
            "ganador": (manifest or {}).get("winner"),
            "version": (manifest or {}).get("version"),
            "generado": (manifest or {}).get("generated_at"),
            "edad_dias": edad,
            "antiguedad": ("Sin manifest" if edad is None else "Vigilar" if edad > MAX_AGE_DAYS else "OK"),
            "metricas": {k: v for k, v in metricas.items() if isinstance(v, (int, float))},
            "artefactos": [{"archivo": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in archivos],
        })
    return registro


# ---------------------------------------------------------------------------
# Desempeño por periodo y deriva
# ---------------------------------------------------------------------------
def load_predictions(slug: str, artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame | None:
    spec = MODULES[slug]
    ruta = artifacts_dir / slug / spec["archivo"]
    if not ruta.exists():
        return None
    df = pd.read_csv(ruta)
    df = df.rename(columns=spec.get("renombrar", {}))
    if "filtro" in spec:
        df = spec["filtro"](df)
    df[spec["fecha"]] = pd.to_datetime(df[spec["fecha"]], errors="coerce")
    return df.dropna(subset=[spec["fecha"]]).sort_values(spec["fecha"]).reset_index(drop=True)


def performance_by_period(slug: str, df: pd.DataFrame) -> pd.DataFrame:
    spec = MODULES[slug]
    freq = spec.get("periodo", "M")
    periodo = df[spec["fecha"]].dt.to_period(freq).astype(str)
    filas = []
    for nombre, grupo in df.groupby(periodo):
        if len(grupo) < MIN_PERIODO:
            continue
        fila = {"periodo": nombre, "n": int(len(grupo)), "metrica": spec["evaluar"](grupo)}
        if spec.get("objetivo") and spec["objetivo"] in grupo:
            fila["tasa_objetivo"] = float((grupo[spec["objetivo"]] == 1).mean()) if slug == "classification" \
                else float((grupo[spec["objetivo"]] == 0).mean())
        if spec.get("score") in grupo:
            fila["score_medio"] = float(grupo[spec["score"]].mean())
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluate_module(slug: str, artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    spec = MODULES[slug]
    manifest = _read_json(artifacts_dir / slug / "manifest.json")
    referencia = _get_nested(manifest, spec["manifest_metric"])
    df = load_predictions(slug, artifacts_dir)
    resultado = {
        "modulo": slug, "titulo": spec["titulo"], "metrica": spec["metrica"],
        "mayor_es_mejor": spec["mayor_es_mejor"], "version": (manifest or {}).get("version"),
        "referencia": referencia, "actual": None, "cambio_relativo": None, "n_predicciones": 0,
        "periodos": [], "drift": {}, "estado": "Sin datos", "motivos": [],
    }
    if df is None or df.empty:
        resultado["motivos"].append("No hay predicciones con fecha en los artefactos.")
        return resultado

    resultado["n_predicciones"] = int(len(df))
    por_periodo = performance_by_period(slug, df)
    resultado["periodos"] = por_periodo.round(4).to_dict(orient="records")
    actual = spec["evaluar"](df)
    resultado["actual"] = round(float(actual), 4) if not np.isnan(actual) else None

    motivos, niveles = [], []
    if referencia is not None and resultado["actual"] is not None:
        cambio = relative_change(float(referencia), float(actual))
        resultado["cambio_relativo"] = round(cambio, 4) if not np.isnan(cambio) else None
        degradacion = -cambio if spec["mayor_es_mejor"] else cambio
        if degradacion >= DEGRADACION_ALERTA:
            niveles.append("Alerta"); motivos.append(f"{spec['metrica']} se degradó {degradacion * 100:.1f}% frente al manifiesto.")
        elif degradacion >= DEGRADACION_VIGILAR:
            niveles.append("Vigilar"); motivos.append(f"{spec['metrica']} se degradó {degradacion * 100:.1f}% frente al manifiesto.")

    # Deriva de la distribución de scores/errores: primer periodo vs último.
    # Solo con periodos suficientemente grandes: con pocas filas el PSI dispara falsas alarmas.
    grandes = por_periodo[por_periodo.n >= MIN_DRIFT] if not por_periodo.empty else por_periodo
    if len(grandes) >= 2 and spec.get("score") in df:
        periodo_col = df[spec["fecha"]].dt.to_period(spec.get("periodo", "M")).astype(str)
        primero, ultimo = grandes.periodo.iat[0], grandes.periodo.iat[-1]
        ref, cur = df.loc[periodo_col == primero, spec["score"]], df.loc[periodo_col == ultimo, spec["score"]]
        valor_psi = psi(ref, cur, bins=10 if min(len(ref), len(cur)) >= 1000 else 5)
        estado_psi = drift_status(valor_psi)
        resultado["drift"]["psi_score"] = {"valor": round(valor_psi, 4), "estado": estado_psi,
                                           "desde": primero, "hasta": ultimo}
        if estado_psi in ("Vigilar", "Alerta"):
            niveles.append(estado_psi); motivos.append(f"PSI de {spec['score']} = {valor_psi:.3f} entre {primero} y {ultimo}.")
    elif spec.get("score") in df:
        resultado["drift"]["psi_score"] = {"valor": None, "estado": "Sin datos",
                                           "nota": f"Se necesitan al menos {MIN_DRIFT} predicciones por periodo."}
    if "tasa_objetivo" in grandes and len(grandes) >= 2:
        t0, t1 = grandes.tasa_objetivo.iat[0], grandes.tasa_objetivo.iat[-1]
        cambio_tasa = relative_change(t0, t1)
        # Puntos porcentuales, no cambio relativo: una tasa del 15% oscila ±2 pp solo por azar.
        estado_tasa = drift_status(abs(t1 - t0), vigilar=0.05, alerta=0.10)
        resultado["drift"]["tasa_objetivo"] = {"inicial": round(t0, 4), "final": round(t1, 4),
                                               "cambio_relativo": round(cambio_tasa, 4), "estado": estado_tasa}
        if estado_tasa in ("Vigilar", "Alerta"):
            niveles.append(estado_tasa); motivos.append(f"La tasa objetivo pasó de {t0:.3f} a {t1:.3f}.")

    # Deriva temporal de la métrica entre periodos (tendencia)
    if len(por_periodo) >= 3:
        serie = por_periodo.metrica.dropna()
        if len(serie) >= 3:
            inicio, fin = serie.iloc[:2].mean(), serie.iloc[-2:].mean()
            deg = relative_change(inicio, fin)
            deg = -deg if spec["mayor_es_mejor"] else deg
            if not np.isnan(deg) and deg >= DEGRADACION_ALERTA:
                niveles.append("Vigilar"); motivos.append(f"La métrica empeoró {deg * 100:.0f}% entre los primeros y los últimos periodos.")

    resultado["estado"] = "Alerta" if "Alerta" in niveles else "Vigilar" if "Vigilar" in niveles else "OK"
    resultado["motivos"] = motivos or ["Desempeño y distribución estables respecto a la referencia."]
    return resultado


# ---------------------------------------------------------------------------
# Reporte, auditoría y logging
# ---------------------------------------------------------------------------
def build_report(artifacts_dir: Path = ARTIFACTS_DIR, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    registro = build_registry(artifacts_dir, now)
    modulos = [evaluate_module(slug, artifacts_dir) for slug in MODULES]
    antiguedad = {r["modulo"]: r["antiguedad"] for r in registro}
    for m in modulos:
        if antiguedad.get(m["modulo"]) == "Vigilar" and m["estado"] == "OK":
            m["estado"] = "Vigilar"
            m["motivos"].append(f"El artefacto supera {MAX_AGE_DAYS} días; conviene reentrenar con datos recientes.")
    for slug, titulo in STATIC_MODULES.items():
        reg = next(r for r in registro if r["modulo"] == slug)
        modulos.append({"modulo": slug, "titulo": titulo, "metrica": None, "mayor_es_mejor": None,
                        "version": reg["version"], "referencia": None, "actual": None, "cambio_relativo": None,
                        "n_predicciones": 0, "periodos": [], "drift": {},
                        "estado": "Vigilar" if reg["antiguedad"] == "Vigilar" else ("Sin datos" if reg["antiguedad"] == "Sin manifest" else "OK"),
                        "motivos": ["Sin predicciones fechadas: se vigila versión y antigüedad del artefacto."]})
    resumen = {estado: sum(m["estado"] == estado for m in modulos) for estado in ESTADOS}
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "umbrales": {"psi_vigilar": 0.10, "psi_alerta": 0.25, "degradacion_vigilar": DEGRADACION_VIGILAR,
                     "degradacion_alerta": DEGRADACION_ALERTA, "max_edad_dias": MAX_AGE_DAYS},
        "resumen": resumen,
        "modulos": modulos,
        "registro": registro,
    }


def append_audit_log(report: dict, path: Path = AUDIT_LOG_PATH, evento: str = "monitoreo") -> pd.DataFrame:
    filas = [{
        "timestamp": report["generated_at"], "evento": evento, "modulo": m["modulo"], "version": m.get("version"),
        "metrica": m.get("metrica"), "referencia": m.get("referencia"), "actual": m.get("actual"),
        "cambio_relativo": m.get("cambio_relativo"), "estado": m["estado"], "motivo": " | ".join(m.get("motivos", [])),
    } for m in report["modulos"]]
    nuevo = pd.DataFrame(filas)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        nuevo.to_csv(path, mode="a", header=False, index=False)
    else:
        nuevo.to_csv(path, index=False)
    return nuevo


def run(artifacts_dir: Path = ARTIFACTS_DIR, write: bool = True, monitoring_dir: Path | None = None) -> dict:
    monitoring_dir = monitoring_dir or (artifacts_dir / "monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    handler = None
    if write:
        handler = logging.FileHandler(monitoring_dir / "monitoring.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    try:
        report = build_report(artifacts_dir)
        for m in report["modulos"]:
            logger.info("%s v%s → %s (%s)", m["modulo"], m.get("version"), m["estado"], "; ".join(m["motivos"]))
        if write:
            with (monitoring_dir / "report.json").open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            with (monitoring_dir / "model_registry.json").open("w", encoding="utf-8") as f:
                json.dump(report["registro"], f, ensure_ascii=False, indent=2)
            append_audit_log(report, monitoring_dir / "audit_log.csv")
        return report
    finally:
        if handler:
            logger.removeHandler(handler)
            handler.close()


def load_report(path: Path = REPORT_PATH) -> dict | None:
    return _read_json(path)
