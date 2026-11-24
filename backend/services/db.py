"""Conexión opcional a la base de datos (PostgreSQL en Compose, SQLite en local).

La lógica analítica no depende de la base: los artefactos siguen siendo la
fuente de verdad. La base sirve para persistir predicciones, registro de
modelos y bitácora de monitoreo de forma consultable (SQL, BI, auditoría).

``DATABASE_URL`` decide el destino; si no está definida se usa un archivo
SQLite en ``data/processed/datasmart.db`` para no exigir Postgres en desarrollo.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE = ROOT / "data" / "processed" / "datasmart.db"


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    DEFAULT_SQLITE.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE.as_posix()}"


def get_engine(url: str | None = None):
    """Motor SQLAlchemy; lanza ImportError claro si la librería no está instalada."""
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Instala sqlalchemy (y psycopg2-binary para PostgreSQL).") from exc
    return create_engine(url or database_url(), future=True)


def database_status(url: str | None = None) -> dict:
    """Comprueba la conexión y lista las tablas disponibles."""
    try:
        from sqlalchemy import inspect
        engine = get_engine(url)
        with engine.connect() as conn:
            tablas = inspect(conn).get_table_names()
        return {"ok": True, "url": _mask(engine.url.render_as_string(hide_password=True)), "tablas": tablas}
    except Exception as exc:  # cualquier fallo de conexión se informa, no se propaga
        return {"ok": False, "url": _mask(url or database_url()), "error": str(exc)}


def _mask(url: str) -> str:
    return url.replace(ROOT.as_posix(), "<repo>")
