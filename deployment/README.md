# Despliegue de DataSmart · Olist

Tres servicios con Docker Compose: `db` (PostgreSQL 16), `backend` (FastAPI) y `frontend`
(Streamlit). Los artefactos de `artifacts/` se montan en los contenedores: entrenar es un paso
previo y separado del despliegue.

## 1. Preparar

```bash
cp deployment/.env.example .env      # ajusta la clave de PostgreSQL
python -m src.monitoring             # genera artifacts/monitoring/ (reporte, registro, bitácora)
```

Si vas a reentrenar, hazlo antes de levantar los servicios (ver README principal: cada módulo
tiene su `python -m src.models.<modulo>`).

## 2. Levantar

```bash
docker compose up --build
```

- Dashboard: <http://localhost:8501>
- API: <http://localhost:8010/docs>
- PostgreSQL: `localhost:5432` (credenciales del `.env`)

`deployment/schema.sql` se ejecuta automáticamente la primera vez que arranca la base y crea las
tablas estables (`model_registry`, `module_metrics`, `monitoring_report`, `audit_log`).

## 3. Cargar artefactos en la base

```bash
docker compose exec backend python scripts/load_artifacts_to_db.py
```

Fuera de Docker, con `DATABASE_URL` definida en el entorno:

```bash
python scripts/load_artifacts_to_db.py
```

Sin `DATABASE_URL` el script usa SQLite en `data/processed/datasmart.db`, útil para desarrollo.

## 4. Operación recurrente (sin orquestador)

Programa estas dos líneas en el planificador del sistema (cron, Task Scheduler) o ejecútalas tras
cada reentrenamiento:

```bash
python -m src.monitoring --retrain      # evalúa deriva y desempeño; reentrena lo que esté en Alerta
python scripts/load_artifacts_to_db.py  # refresca registro, predicciones y bitácora en la base
```

Ejemplo de cron semanal (lunes 06:00):

```
0 6 * * 1 cd /ruta/al/repo && python -m src.monitoring --retrain && python scripts/load_artifacts_to_db.py
```

## 5. Detener y limpiar

```bash
docker compose down            # conserva los datos de PostgreSQL
docker compose down -v         # además borra el volumen datasmart_pgdata
```
