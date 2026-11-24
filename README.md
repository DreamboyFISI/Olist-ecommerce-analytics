# DataSmart · Olist — Plataforma de Minería de Datos

Proyecto Integrador · Minería de Datos · UNMSM FISI · 2026

## Estado
- [x] 1° avance (S8): comprensión del negocio y de los datos — `notebooks/01_eda/`
- [ ] 2° avance (S12): preparación, asociación, clustering, series, grafos, regresión, prototipo
- [ ] Entrega final (S14): clasificación, NLP, explicabilidad, monitoreo, app web
- [x] Módulo NLP / sentimientos de reseñas — `notebooks/09_nlp/`, `src/models/sentiment.py`
- [x] Módulo de clasificación / riesgo de reseña negativa — `notebooks/08_classification/`, `src/models/classification.py`
- [x] Explicabilidad (SHAP, LIME, permutación, árbol sustituto) — `src/explainability/`, página **Explicabilidad**
- [x] Monitoreo, versionado, auditoría y reentrenamiento (sin orquestador) — `src/monitoring/`, página **Monitoreo**
- [x] PostgreSQL en Compose y carga de artefactos — `deployment/`, `scripts/load_artifacts_to_db.py`
- [x] Informe técnico consolidado — `python scripts/build_report.py` → `docs/informe_tecnico.md`

## Cómo ejecutar el EDA
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda/01_eda_calidad_datos.ipynb   # Run All
```
El notebook usa los CSV de `data/raw/`. Si no están, los descarga con `kagglehub`.
Alternativa sin cuenta de Kaggle:
```bash
git clone --depth 1 https://github.com/olist/work-at-olist-data.git
cp work-at-olist-data/datasets/*.csv data/raw/
```

Genera:
- `docs/diccionario_datos.xlsx` — diccionario de las 52 columnas
- `docs/figures/*.png` — 11 gráficos de EDA y calidad
- `docs/hallazgos_eda.json` — métricas clave usadas en el informe

## Hallazgos principales
| Indicador | Valor |
|---|---|
| Reseñas negativas (1–2★) | 14.69% |
| % negativas: entrega tardía vs a tiempo | 54.07% vs 9.22% |
| MAE de la fecha estimada por Olist | 12.73 días |
| Clientes con recompra | 3.06% |
| Pedidos con más de una categoría | 0.80% |

## Resultados del módulo de sentimientos (NLP)

El módulo clasifica el texto de las reseñas en **negativo (1–2★), neutro (3★) y positivo (4–5★)**
usando 35,044 reseñas únicas en portugués. En esta población con texto, 29.3% son negativas, 9.1%
neutras y 61.6% positivas (la tasa de 14.69% de todo el dataset incluye reseñas sin comentario).
La métrica principal es **macro-F1** por el desbalance de clases; también se reporta PR-AUC de la
clase negativa y QWK para la nota ordinal 1–5.

| Indicador | Valor |
|---|---:|
| Macro-F1 (test) | 0.677 |
| Accuracy | 83.7% |
| PR-AUC negativa | 0.879 |
| Recall negativa (umbral 0.374) | 90.6% |
| QWK nota 1–5 (modelo auxiliar) | 0.841 |
| Acierto ±1 estrella | 88.8% |
| Transformer externo zero-shot (macro-F1 balanceado) | 0.649 |

El modelo desplegado es una **regresión logística con TF-IDF de palabras y caracteres**; el
transformer BERTimbau entrenado en B2W se evalúa solo como referencia. Detalle metodológico y
limitaciones en [`docs/nlp_resumen.md`](docs/nlp_resumen.md).

## Dataset
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0) — 9 tablas, pedidos de sep-2016 a oct-2018.

## Dashboard analítico

El repositorio incluye una base modular para que los integrantes trabajen en ramas separadas sin
modificar la navegación o los componentes compartidos. La interfaz usa Streamlit y Plotly con las
pantallas conectadas mediante un menú lateral accesible.

La página **EDA y calidad** consume `docs/hallazgos_eda.json` y presenta KPIs, gráficas interactivas y
las figuras originales del notebook. La página **Sentimientos** consume `artifacts/nlp/` y permite
analizar una reseña en portugués, comparar modelos y revisar los cruces con logística. Las pantallas
de modelamiento muestran un estado vacío explícito hasta que exista un artefacto real; no se publican
resultados simulados.

### Ejecutar backend y frontend

Desde la raíz del repositorio, instalar primero las dependencias:

```bash
pip install -r requirements.txt
```

Terminal 1 — API FastAPI:

```bash
uvicorn backend.main:app --reload --port 8000
```

Comprobar la API en `http://127.0.0.1:8000/docs` o `http://127.0.0.1:8000/health`.

Terminal 2 — dashboard Streamlit:

```bash
streamlit run frontend/app.py
```

Abrir `http://localhost:8501`. Si la API no está disponible, el EDA utiliza automáticamente el JSON
local y muestra **Modo local**. Para usar otra dirección de API, definir `OLIST_API_URL` antes de
iniciar Streamlit.

### Ramas de trabajo

- `feature/association`
- `feature/clustering`
- `feature/timeseries`
- `feature/graphs`
- `modelo-regresion-jc`
- `feature/classification`
- `feature/nlp`

Cada integrante trabaja en su notebook, módulo productivo, pruebas, carpeta dentro de `artifacts/` y
pantalla correspondiente. No debe colocar entrenamiento pesado dentro de Streamlit ni repetir la
limpieza de los CSV en cada página.

Las vistas aisladas están en `frontend/views/model_pages/`. Cada rama debe modificar únicamente su
archivo homónimo y reutilizar los componentes compartidos; los cambios a navegación, estilos o
contratos se coordinan en la rama de integración.

### Contrato común de los módulos

Cada módulo debe producir:

- Estado del entrenamiento.
- Nombre y versión del modelo.
- Fecha de generación.
- Dataset o periodo utilizado.
- Métricas de entrenamiento y prueba.
- Modelo ganador.
- Tabla pequeña con resultados o predicciones.
- Datos preparados para las gráficas.
- Conclusiones de negocio.
- Limitaciones.
- Modelo serializado cuando corresponda.

`docs/module_contract.schema.json` formaliza el manifest que activa los resultados en el dashboard.
Cada módulo escribe su manifest en `artifacts/<modulo>/manifest.json`.

Salidas específicas:

- **Asociación:** reglas, support, confidence, lift y algoritmo.
- **Clustering:** cliente, cluster, métricas RFM, perfil y Silhouette.
- **Series temporales:** fecha, valor real, pronóstico, intervalos y error.
- **Grafos:** nodos, aristas, peso, comunidad y centralidad.
- **Regresión:** valor real, predicción, error y métricas.
- **Clasificación:** clase real, probabilidad, predicción, matriz de confusión y métricas.
- **NLP / Sentimientos:** sentimiento predicho, probabilidad, términos influyentes y cruces con logística.

### Ejecución de módulos de modelamiento (ejemplos: Clustering y Asociación)

Antes de iniciar el dashboard o levantar el entorno con Docker, se debe ejecutar el pipeline productivo para entrenar los modelos y generar los artefactos requeridos por el contrato:

```bash
python -m src.models.clustering
```

Para el módulo de Asociación (Market Basket Analysis):

```bash
python -m src.models.association
```

Para ejecutar las pruebas unitarias del módulo de asociación:

```bash
python tests/test_association.py
```

El pipeline de clustering ejecuta de principio a fin:
1. Agrupación RFM por `customer_unique_id` a partir de `data/raw/`.
2. Preprocesamiento con `log1p` y `StandardScaler`.
3. Comparación de modelos (K-Means k=3..6, DBSCAN y GMM) mediante Silhouette Score, Davies-Bouldin y Calinski-Harabasz.
4. Generación y validación de los cuatro artefactos en `artifacts/clustering/` (`manifest.json`, `results.csv`, `chart_data.json` y `model.joblib`).

Para verificar el cumplimiento del contrato antes de confirmar los cambios:

```bash
python -c "import json, jsonschema, pathlib; jsonschema.validate(json.loads(pathlib.Path('artifacts/clustering/manifest.json').read_text(encoding='utf-8')), json.loads(pathlib.Path('docs/module_contract.schema.json').read_text(encoding='utf-8'))); print('Contrato válido')"
```

### Clasificación: riesgo de reseña negativa

El módulo predice, al momento en que Olist envía la encuesta de satisfacción, qué pedidos
recibirán 1 o 2 estrellas. Entrena desde `data/raw/`, compara seis algoritmos con validación
cruzada estratificada, elige el más simple entre los empatados, calibra el umbral fuera de
muestra y evalúa en una partición temporal (junio–agosto 2018):

```bash
python -m src.models.classification            # corrida completa (≈ 10–15 min)
python -m src.models.classification --quick    # corrida corta para verificar el flujo
```

Genera `artifacts/classification/manifest.json`, `results.csv`, `chart_data.json` y `model.joblib`
(pipeline + umbral + lista de variables). La página **Clasificación** permite evaluar un pedido
(formulario → `POST /api/v1/classification/predict`) y revisa comparación de modelos, curvas,
matriz de confusión, ganancia por deciles, importancia por permutación y escenarios.

Pruebas del módulo:

```bash
pytest tests/test_classification.py
```

### Análisis de sentimientos (NLP)

Para entrenar el módulo de sentimientos y generar sus artefactos:

```bash
python -m src.models.sentiment
```

Genera `artifacts/nlp/manifest.json`, `results.csv`, `chart_data.json`, `model.joblib`,
`comparacion_modelos.csv`, `metricas_por_clase.csv` y `terminos_por_clase.csv`.
El benchmark opcional con un transformer preentrenado se ejecuta aparte y requiere
`requirements-nlp.txt`:

```bash
pip install -r requirements-nlp.txt
python scripts/benchmark_transformer.py --sample-per-class 500
```

Para ejecutar las pruebas del módulo de sentimientos:

```bash
python -m pytest tests/test_sentiment.py -q
```

### Actualizar el módulo de grafos

La vista de Grafos lee `artifacts/graphs/chart_data.json`; no entrena dentro de Streamlit. Cuando cambien
los CSV de `data/raw/` o la tabla procesada:

```bash
python -m src.models.graphs --refresh
```

Ese comando reconstruye `data/processed/grafos/items_pedidos.csv` desde `data/raw/` y vuelve a generar
todos los artefactos (`nodos.csv`, `aristas.csv`, `comunidades.csv`, `metricas.json`, `chart_data.json` y
`manifest.json`). Si solo cambió `items_pedidos.csv`, basta con:

```bash
python -m src.models.graphs
```

La vista detecta si los artefactos están desactualizados y muestra un aviso con el comando que
corresponde. El diagrama se redibuja desde los artefactos en cada recarga, sin imágenes ni datos fijos.
### Integración y evaluación

La API expone inicialmente:

- `GET /health`
- `GET /api/v1/eda/summary`
- `GET /api/v1/modules`
- `GET /api/v1/nlp/results`
- `POST /api/v1/nlp/predict`
- `GET /api/v1/classification/results`
- `POST /api/v1/classification/predict`
- `POST /api/v1/classification/explain?method=shap|lime|perturbacion`
- `GET /api/v1/monitoring/report?refresh=false`

La evaluación comparativa es transversal. Cada integrante compara los algoritmos de su módulo y
publica las métricas bajo el contrato común. El dashboard consolida ganadores y versiones sin comparar
directamente métricas incompatibles como MAE, Silhouette, lift, modularidad y F1.

Docker empaqueta PostgreSQL, Streamlit y FastAPI (ver `deployment/README.md`). La validación, la
detección de deriva y el reentrenamiento no usan orquestador externo: viven en `src/monitoring/`
y se programan con el planificador del sistema.

### Monitoreo, versionado, auditoría y reentrenamiento

```bash
python -m src.monitoring                    # reporte en artifacts/monitoring/ + bitácora
python -m src.monitoring --retrain --dry-run  # plan de reentrenamiento (qué está en Alerta)
python -m src.monitoring --retrain          # reentrena los módulos en Alerta y respalda su manifest
```

El monitoreo recalcula por periodo la métrica principal de cada módulo con las predicciones
fechadas de sus artefactos, mide la deriva de probabilidades/errores (PSI) y de la tasa objetivo,
registra versión, fecha y huella SHA-256 de cada artefacto (`model_registry.json`) y agrega una
fila por módulo a `audit_log.csv` en cada corrida. Umbrales: PSI 0.10 / 0.25; degradación 10 % / 20 %;
antigüedad máxima 90 días. La página **Monitoreo** del dashboard muestra todo esto.

### Explicabilidad

`src/explainability/explain.py` explica predicciones individuales con SHAP (modelos de árboles),
LIME (si existe `explain_background.csv`) o perturbación con valores típicos, agregando las columnas
one-hot a su variable original. La página **Clasificación** muestra el "por qué" de cada pedido
evaluado y la página **Explicabilidad** consolida importancias, el árbol sustituto de clustering,
las variables del modelo de fletes y los términos del modelo de sentimientos.

### Base de datos

`compose.yaml` incluye `db` (PostgreSQL 16) y `deployment/schema.sql` crea las tablas estables.
`python scripts/load_artifacts_to_db.py` carga registro de modelos, métricas, predicciones de cada
módulo, reporte de monitoreo y bitácora; sin `DATABASE_URL` usa SQLite en `data/processed/`.

### Ingeniería de características

`src/features/selection.py` reúne el filtro de colinealidad, la selección por importancia (árboles o
permutación), PCA/LDA y la referencia LDA/QDA; `src/models/classification.py` compara sin
tratamiento, ponderación de clases, SMOTE y ADASYN. La serie temporal pronostica 90 días y resume
ventas esperadas a 7, 30, 60 y 90 días (`generar_artefactos()`).

## Ejecución con Docker

Con Docker Desktop iniciado, levantar backend y frontend desde la raíz:

```bash
docker compose up --build
```

Abrir:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8010/docs`

Para detener los servicios:

```bash
docker compose down
```

Los volúmenes permiten que los cambios del código se reflejen durante el desarrollo. La base de datos
no forma parte todavía del Compose; se añadirá cuando exista el pipeline procesado y se defina el
esquema de PostgreSQL.

## Guía breve para el equipo

Antes de repartir tareas, `eda-front` debe integrarse en `develop`. Cada integrante crea su rama desde
`develop`, nunca directamente desde `main`:

```bash
git switch develop
git pull origin develop
git switch -c feature/nombre-modulo
```

Al terminar:

```bash
git push -u origin feature/nombre-modulo
```

El Pull Request debe apuntar de `feature/nombre-modulo` hacia `develop`.

| Rama | Responsabilidad |
|---|---|
| `feature/association` | Notebook `03_association`, Apriori/FP-Growth/ECLAT, reglas, métricas y `association.py`. |
| `feature/clustering` | Notebook `04_clustering`, RFM, K-Means/DBSCAN, segmentos, Silhouette y `clustering.py`. |
| `feature/timeseries` | Notebook `05_timeseries`, SARIMA/Prophet, backtesting, pronósticos y `timeseries.py`. |
| `feature/graphs` | Notebook `06_graphs`, nodos, aristas, centralidad, Louvain y `graphs.py`. |
| `modelo-regresion-jc` | Notebook `07-regression-fletev3.ipynb`, predicción del costo del flete, sistema híbrido con XGBoost y evaluación mediante MAE, RMSE, R² y WAPE. |
| `feature/classification` | Notebook `08_classification`, reseña negativa, Recall/F1/PR-AUC y `classification.py`. |
| `feature/nlp` | Notebook `09_nlp`, sentimiento de reseñas, TF-IDF/SVM/Naive Bayes/BERTimbau, macro-F1 y `sentiment.py`. |

Flujo obligatorio de cada módulo:

1. Experimentar en su notebook.
2. Pasar la lógica definitiva a `src/models/`.
3. Guardar resultados en `artifacts/<modulo>/`.
4. Crear `manifest.json`.
5. Completar su archivo en `frontend/views/model_pages/`.
6. Agregar pruebas y abrir PR hacia `develop`.

Cada integrante modifica únicamente su módulo. Los cambios en navegación, estilos globales o contratos
se coordinan previamente en `develop`.


## Cada compañero hará:
1. Entrenar y comparar varios algoritmos.
2. Evaluarlos con métricas apropiadas.
3. Seleccionar un único modelo ganador.
4. Guardar ese modelo y sus resultados en artifacts/<modulo>/.
5. Mostrar en Streamlit la comparación y justificar por qué ganó.

| Módulo | Pregunta que responde | Utilidad |
|---|---|---|
| **Asociación** | ¿Qué productos o categorías se compran juntos? | Crear recomendaciones y estrategias de venta cruzada. |
| **Clustering** | ¿Qué tipos de clientes existen y cuáles tienen mayor valor? | Segmentar clientes y diseñar acciones diferenciadas. |
| **Series temporales** | ¿Cuánto se venderá en los próximos periodos? | Planificar inventario, logística y demanda. |
| **Grafos** | ¿Qué productos, categorías o vendedores están más conectados? | Detectar comunidades, productos centrales y relaciones relevantes. |
| **Regresión** | ¿Cuánto costará el flete de un pedido? | Anticipar el costo logístico, proteger la rentabilidad y detectar envíos inusualmente caros. |
| **Clasificación** | ¿Qué pedidos podrían recibir una reseña negativa? | Identificar riesgos y actuar antes de que el cliente quede insatisfecho. |
| **NLP / Sentimientos** | ¿Qué dicen y sienten los clientes en sus reseñas? | Priorizar quejas, detectar temas críticos y monitorear la satisfacción. |