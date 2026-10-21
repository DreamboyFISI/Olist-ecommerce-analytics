# DataSmart · Olist — Plataforma de Minería de Datos

Proyecto Integrador · Minería de Datos · UNMSM FISI · 2026

## Estado
- [x] 1° avance (S8): comprensión del negocio y de los datos — `notebooks/01_eda/`
- [ ] 2° avance (S12): preparación, asociación, clustering, series, grafos, regresión, prototipo
- [ ] Entrega final (S14): clasificación, NLP, explicabilidad, monitoreo, app web

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

## Dataset
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0) — 9 tablas, pedidos de sep-2016 a oct-2018.

## Dashboard analítico

El repositorio incluye una base modular para que seis integrantes trabajen en ramas separadas sin
modificar la navegación o los componentes compartidos. La interfaz usa Streamlit y Plotly con las
pantallas conectadas mediante un menú lateral accesible.

La página **EDA y calidad** consume `docs/hallazgos_eda.json` y presenta KPIs, gráficas interactivas y
las figuras originales del notebook. Las pantallas de modelamiento muestran un estado vacío explícito
hasta que exista un artefacto real; no se publican resultados simulados.

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
- `feature/regression`
- `feature/classification`

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

### Integración y evaluación

La API expone inicialmente:

- `GET /health`
- `GET /api/v1/eda/summary`
- `GET /api/v1/modules`

La evaluación comparativa es transversal. Cada integrante compara los algoritmos de su módulo y
publica las métricas bajo el contrato común. El dashboard consolida ganadores y versiones sin comparar
directamente métricas incompatibles como MAE, Silhouette, lift, modularidad y F1.

Docker y n8n se incorporarán después de estabilizar los modelos y sus contratos. Docker empaquetará
Streamlit, FastAPI, PostgreSQL y n8n; n8n orquestará validación, detección de drift y reentrenamiento,
pero la lógica analítica permanecerá en Python.
