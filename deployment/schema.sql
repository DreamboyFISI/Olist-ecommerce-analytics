-- Esquema base de DataSmart en PostgreSQL.
-- Las tablas de predicciones (predictions_<modulo>) las crea scripts/load_artifacts_to_db.py
-- a partir de las columnas de cada artefacto; aquí se fijan las tablas estables.

CREATE TABLE IF NOT EXISTS model_registry (
    modulo           TEXT PRIMARY KEY,
    titulo           TEXT NOT NULL,
    modelo           TEXT,
    version          TEXT,
    generado         TEXT,
    edad_dias        INTEGER,
    antiguedad       TEXT,
    estado_manifest  TEXT,
    metricas_json    TEXT,
    artefactos_json  TEXT,
    cargado_en       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS module_metrics (
    modulo      TEXT NOT NULL,
    version     TEXT,
    metrica     TEXT NOT NULL,
    valor       DOUBLE PRECISION,
    cargado_en  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monitoring_report (
    generated_at     TEXT NOT NULL,
    modulo           TEXT NOT NULL,
    titulo           TEXT,
    version          TEXT,
    metrica          TEXT,
    referencia       DOUBLE PRECISION,
    actual           DOUBLE PRECISION,
    cambio_relativo  DOUBLE PRECISION,
    estado           TEXT NOT NULL,
    motivos          TEXT,
    drift_json       TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    "timestamp"      TEXT NOT NULL,
    evento           TEXT NOT NULL,
    modulo           TEXT NOT NULL,
    version          TEXT,
    metrica          TEXT,
    referencia       DOUBLE PRECISION,
    actual           DOUBLE PRECISION,
    cambio_relativo  DOUBLE PRECISION,
    estado           TEXT NOT NULL,
    motivo           TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_log_modulo ON audit_log (modulo, "timestamp");
CREATE INDEX IF NOT EXISTS idx_monitoring_report_modulo ON monitoring_report (modulo, generated_at);
