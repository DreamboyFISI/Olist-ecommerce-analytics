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
