"""Genera docs/informe_tecnico.md a partir de los manifiestos y del reporte de monitoreo.

El informe consolida, para cada módulo, pregunta de negocio, modelo ganador,
métricas, conclusiones y limitaciones tal como están en ``artifacts/``; así la
documentación nunca se desalinea de los artefactos que sirve el dashboard.

Uso:
    python scripts/build_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.repository import MODULES  # noqa: E402
from src.monitoring.monitor import load_report  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
SALIDA = ROOT / "docs" / "informe_tecnico.md"

INTRO = """# DataSmart · Olist — Informe técnico consolidado

Proyecto Integrador · Minería de Datos · UNMSM FISI · 2026

**Problema.** DataSmart E-Commerce (materializada con el dataset público de Olist: 9 tablas,
99,441 pedidos de septiembre 2016 a octubre 2018) necesita responder con datos qué productos se
compran juntos, qué tipos de clientes existen, cuánto venderá, cómo se relacionan categorías y
vendedores, cuánto costará el flete, qué pedidos terminarán en una reseña negativa y qué dicen
los clientes en sus reseñas.

**Metodología.** CRISP-DM: comprensión del negocio y de los datos (EDA y calidad), preparación
(tabla maestra de 99,441 × 51 con flags de calidad y feature engineering), modelamiento por
módulos con comparación de al menos tres alternativas cada uno, evaluación contra un baseline
propio, despliegue en una plataforma web (API FastAPI + dashboard Streamlit + PostgreSQL) y
monitoreo con deriva, versionado, auditoría y reentrenamiento.

**Hallazgos transversales.** 14.69 % de reseñas negativas; 54.07 % de negativas cuando la entrega
es tardía frente a 9.22 % a tiempo; la promesa de entrega de Olist falla en 12.73 días; 3.06 % de
clientes recompra; 0.80 % de pedidos mezcla categorías.
"""


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:,.2f}"
    return str(v)


def _seccion_modulo(definicion: dict) -> str:
    slug = definicion["slug"]
    ruta = ARTIFACTS / slug / "manifest.json"
    lineas = [f"## {definicion['title']}", "", f"**Pregunta de negocio.** {definicion['question']}", ""]
    if not ruta.exists():
        lineas.append("_Sin manifiesto: módulo pendiente._\n")
        return "\n".join(lineas)
    m = json.loads(ruta.read_text(encoding="utf-8"))
    lineas.append(f"**Modelo ganador.** {m.get('winner') or m.get('model_name')} · versión {m.get('version')} · "
                  f"generado {str(m.get('generated_at', ''))[:19]}")
    if m.get("period"):
        lineas.append(f"\n**Periodo.** {m['period']}")
    if m.get("validation"):
        lineas.append(f"\n**Validación.** {m['validation']}")
    metricas = m.get("metrics") or m.get("test_metrics") or {}
    if metricas:
        lineas += ["", "| Métrica | Valor |", "|---|---|"]
        lineas += [f"| {k} | {_fmt(v)} |" for k, v in metricas.items() if isinstance(v, (int, float, str))]
    for clave, titulo in (("models_compared", "Modelos comparados"), ("why_winner", "Por qué ganó"),
                          ("business_conclusions", "Conclusiones de negocio"), ("limitations", "Limitaciones")):
        valores = m.get(clave)
        if not valores:
            continue
        lineas += ["", f"**{titulo}.**", ""]
        for v in valores:
            lineas.append(f"- {json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v}")
    lineas.append("")
    return "\n".join(lineas)


def _seccion_monitoreo() -> str:
    report = load_report()
    if not report:
        return "## Monitoreo\n\n_Ejecuta `python -m src.monitoring` para generar el reporte._\n"
    lineas = ["## Monitoreo y mantenimiento", "",
              f"Último reporte: {report['generated_at']}. Resumen: " +
              ", ".join(f"{k} {v}" for k, v in report["resumen"].items()) + ".", "",
              "| Módulo | Versión | Métrica | Referencia | Actual | Estado | Motivo |", "|---|---|---|---|---|---|---|"]
    for mod in report["modulos"]:
        lineas.append(f"| {mod['titulo']} | {mod.get('version') or '—'} | {mod.get('metrica') or '—'} | "
                      f"{_fmt(mod['referencia']) if mod.get('referencia') is not None else '—'} | "
                      f"{_fmt(mod['actual']) if mod.get('actual') is not None else '—'} | {mod['estado']} | "
                      f"{' '.join(mod.get('motivos', []))} |")
    lineas += ["", "Reglas: PSI < 0.10 estable, 0.10–0.25 vigilar, > 0.25 alerta; degradación de la métrica "
               "principal ≥ 10 % vigilar, ≥ 20 % alerta; artefactos de más de 90 días se marcan para revisión. "
               "Un módulo en Alerta se reentrena con `python -m src.monitoring --retrain`; cada corrida queda en "
               "`artifacts/monitoring/audit_log.csv`.", ""]
    return "\n".join(lineas)


ARQUITECTURA = """## Plataforma

- **Contrato de artefactos**: cada módulo publica `manifest.json`, `results.csv`, `chart_data.json` y
  `model.joblib` validados con `docs/module_contract.schema.json`.
- **API FastAPI**: salud, resumen del EDA, registro de módulos, resultados y predicción en línea de
  regresión, clasificación (con explicación SHAP/LIME) y sentimientos, y reporte de monitoreo.
- **Dashboard Streamlit**: Resumen, EDA y calidad, una página por módulo, Evaluación comparativa,
  Explicabilidad y Monitoreo (estado, desempeño en el tiempo, deriva, versionado, auditoría y plan
  de reentrenamiento).
- **Persistencia**: PostgreSQL en Docker Compose (`deployment/schema.sql`) o SQLite local;
  `scripts/load_artifacts_to_db.py` carga registro de modelos, métricas, predicciones y bitácora.
- **Explicabilidad**: importancia por permutación, SHAP y LIME por predicción, árbol sustituto para
  los segmentos y pesos del modelo de texto (`src/explainability`).
- **Ingeniería de características**: construcción e integración en `src/data/build_master.py`;
  colinealidad, selección por importancia, PCA/LDA y referencia QDA en `src/features/selection.py`;
  balanceo con SMOTE y ADASYN dentro de los pipelines.
"""


def build() -> Path:
    partes = [INTRO, ARQUITECTURA] + [_seccion_modulo(d) for d in MODULES] + [_seccion_monitoreo()]
    partes.append(f"\n_Generado automáticamente el {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC con "
                  "`python scripts/build_report.py`._\n")
    SALIDA.write_text("\n".join(partes), encoding="utf-8")
    return SALIDA


if __name__ == "__main__":
    print(f"Informe escrito en {build()}")
