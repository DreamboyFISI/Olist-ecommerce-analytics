# Análisis de sentimientos de reseñas — resumen del módulo

Módulo NLP / Sentimientos · rama `feature/nlp`
Proyecto Integrador · Minería de Datos · UNMSM FISI 2026

## Pregunta de negocio

¿Qué sentimiento expresan las reseñas escritas por los clientes y qué temas o experiencias
impulsan la insatisfacción? El objetivo operativo es convertir el texto libre en una señal
accionable para priorizar casos, detectar problemas de producto o logística y monitorear la
satisfacción más allá de la nota promedio.

## Datos y etiqueta

- Fuente: `data/raw/olist_order_reviews_dataset.csv` (Olist, portugués brasileño).
- Se conserva una reseña por pedido (la última respondida).
- Se usan título y mensaje; se exige al menos una letra.
- Se eliminan duplicados exactos del texto normalizado para evitar fuga entre train y test.
- Población final: **35,044 reseñas únicas** de 2016-10 a 2018-08.

La etiqueta se construye desde la nota del cliente:

| Nota | Clase | Rol |
|---:|---|---|
| 1–2 estrellas | Negativo | Clase crítica para el negocio |
| 3 estrellas | Neutro | Clase ambigua y difícil |
| 4–5 estrellas | Positivo | Clase mayoritaria |

Distribución: **29.3% negativo, 9.1% neutro, 61.6% positivo**. Es una etiqueta débil: mide
satisfacción global, no sentimiento puro, pero es la señal más cercana disponible para todo el
dataset y permite entrenar sin anotación manual.

## Por qué esta forma de evaluar

1. **Macro-F1 como métrica principal.** Las clases están desbalanceadas; el accuracy premia
   acertar la clase positiva. Macro-F1 da el mismo peso a negativo, neutro y positivo.
2. **PR-AUC y ROC-AUC de la clase negativa.** El negocio necesita detectar insatisfacción; la
   PR-AUC es más informativa que ROC-AUC cuando la clase negativa es minoría.
3. **Umbral operativo fuera de muestra.** El umbral de negativa se elige maximizando F1 con
   predicciones out-of-fold del entrenamiento, nunca con el test.
4. **QWK y accuracy ±1 para la nota 1–5.** La nota es ordinal; la kappa ponderada cuadrática
   castiga más los errores lejanos y la accuracy ±1 mide si el modelo se acerca.
5. **Deduplicación y benchmark externo.** Se eliminan textos repetidos y el transformer de
   referencia se entrena en otro e-commerce (B2W), no en Olist, para no filtrar información.

## Preprocesamiento

- Normalización Unicode, minúsculas, eliminación de HTML, URLs y correos.
- Se conservan acentos, signos de exclamación/interrogación y negaciones (`não`, `nunca`, `sem`).
- No se eliminan stopwords: en portugués una negación cambia el sentido de toda la frase.
- Representación: **TF-IDF de palabras (1–2 n-gramas, 60k features) + TF-IDF de caracteres
  (3–5 n-gramas, 80k features)**. Los n-gramas de carácter capturan errores de tipeo,
  conjugaciones y variantes morfológicas.

## Modelos comparados

Validación cruzada estratificada de 5 folds sobre el 80% de entrenamiento; el 20% restante se
reserva para la evaluación final. La selección se hace por macro-F1.

| Modelo | Macro-F1 CV | Macro-F1 test | Accuracy test | Balanced accuracy | PR-AUC negativa |
|---|---:|---:|---:|---:|---:|
| **Regresión Logística (C=0.5)** | **0.6770** | **0.6769** | 0.8368 | 0.6815 | 0.8786 |
| SVM lineal calibrado | 0.6061 | 0.6157 | 0.8528 | 0.6340 | 0.8812 |
| Complement Naive Bayes | 0.6080 | 0.6126 | 0.8259 | 0.6346 | 0.8435 |
| Baseline (clase mayoritaria) | 0.2541 | 0.2541 | 0.6159 | 0.3333 | 0.2928 |

El modelo desplegado es la **regresión logística con TF-IDF palabra+carácter**. La búsqueda
pequeña de `C` eligió 0.5. El SVM tiene accuracy global algo mayor, pero su macro-F1 y balanced
accuracy son peores: no compensa el mal desempeño en neutro.

### Desempeño del ganador en test

| Métrica | Valor |
|---|---:|
| Macro-F1 | 0.6769 |
| Weighted-F1 | 0.8349 |
| Accuracy | 0.8368 |
| Balanced accuracy | 0.6815 |
| QWK (3 clases) | 0.8309 |
| PR-AUC negativa | 0.8786 |
| ROC-AUC negativa | 0.9542 |
| F1 negativa (umbral 0.3745) | 0.8363 |
| Precisión negativa | 0.7765 |
| Recall negativa | 0.9059 |

Por clase: negativo F1 **0.833**, neutro F1 **0.279**, positivo F1 **0.918**. La clase neutra es
la más difícil: muchas reseñas de 3 estrellas son ambivalentes o describen el producto sin una
polaridad clara.

### Evaluación ordinal auxiliar (nota 1–5)

| Métrica | Valor |
|---|---:|
| QWK | 0.8410 |
| Nota exacta | 67.44% |
| Acierto ±1 estrella | 88.76% |
| PR-AUC negativa | 0.8781 |

El modelo de 5 clases no se despliega, pero demuestra que el texto conserva información ordinal:
aunque no siempre acierte la estrella exacta, casi 9 de cada 10 predicciones quedan a una nota
de distancia.

## Benchmark con transformer (referencia externa)

Se evaluó **BERTimbau fine-tuneado en reseñas de B2W** (`Eduardo211000/bertimbau-sentimento-b2w`),
otro e-commerce brasileño, en modo zero-shot sobre la misma muestra balanceada de test
(500 reseñas por clase). No se entrenó con Olist ni se despliega en la API.

| Modelo | Macro-F1 | Accuracy | F1 negativo | F1 neutro | F1 positivo |
|---|---:|---:|---:|---:|---:|
| Regresión Logística (clásico) | 0.6476 | — | — | — | — |
| BERTimbau B2W zero-shot | **0.6486** | 0.6873 | 0.746 | 0.383 | 0.817 |

En la muestra balanceada hay un **empate técnico** (diferencia < 0.01). El transformer obtiene un
F1 de neutro de **0.383**, pero su ventaja global es marginal; cuesta órdenes de magnitud más en
inferencia y no es interpretable término a término. Se despliega el clásico por latencia, costo y
explicabilidad. Fine-tunear BERTimbau con Olist en GPU es la mejora natural siguiente.

## Hallazgos de negocio

- **La logística domina la negatividad.** Las reseñas de pedidos con entrega tardía tienen
  **72.8%** de negativas predichas; las entregadas a tiempo, **23.6%**; sin entrega registrada,
  **89.2%**.
- **Categorías prioritarias.** Entre las 15 categorías con más reseñas, `baby` (37.2%),
  `furniture_decor` (35.7%) y `bed_bath_table` (34.9%) concentran la mayor tasa de negativas.
- **Estacionalidad.** Marzo y abril de 2018 muestran picos de negatividad (47.6% y 43.8%),
  coherentes con problemas logísticos de esos meses.
- **Términos interpretables.** El modelo asocia negativo con `nao`, `nao recomendo`,
  `nao gostei`, `pessima`, `nao chegou`, `nao recebi`; positivo con `otimo`, `amei`, `excelente`,
  `muito bom`, `no prazo`, `antes do prazo`.
- **Ruido de etiqueta.** Hay reseñas con texto positivo y nota negativa (y viceversa), lo que
  confirma que la nota no es una anotación perfecta de sentimiento. Esas discrepancias son
  candidatas a revisión manual, no necesariamente errores del modelo. La vista de errores muestra
  la nota exacta (1–5) para distinguir un fallo del modelo de una contradicción texto-nota.

## Limitaciones

- Etiqueta débil: la nota mide satisfacción global; 3 estrellas es ambigua.
- Solo ~41% de las reseñas tienen comentario; hay sesgo de selección.
- Portugués coloquial, ironía, sarcasmo, emojis y errores de tipeo.
- Se eliminan duplicados exactos, pero pueden quedar near-duplicates.
- La partición train/test es aleatoria y estratificada, no temporal; mide generalización a reseñas
  similares, no deriva futura. Para producción conviene añadir un holdout temporal.
- Las tasas por categoría se calculan sobre el test y algunas categorías tienen menos de 200
  reseñas; son señales exploratorias, no estimaciones precisas.
- El transformer es zero-shot y de otro dominio; no se fine-tuneó en Olist.
- Deriva temporal 2016–2018; se requiere monitoreo y reentrenamiento.
- Asociación no causal: retraso y negatividad están correlacionados, no se prueba causalidad.

## Reproducción

```bash
pip install -r requirements.txt

# Pipeline principal (~4 min en CPU)
python -m src.models.sentiment

# Benchmark opcional del transformer (requiere GPU/CPU y requirements-nlp.txt)
pip install -r requirements-nlp.txt
python scripts/benchmark_transformer.py --sample-per-class 500

# Pruebas
python -m pytest tests/test_sentiment.py -q
```

Artefactos generados en `artifacts/nlp/`:

- `manifest.json` — contrato del módulo.
- `results.csv` — muestra de predicciones del test.
- `comparacion_modelos.csv` — comparación de algoritmos.
- `metricas_por_clase.csv` — precisión, recall, F1 y soporte por clase.
- `terminos_por_clase.csv` — términos con mayor peso por clase.
- `chart_data.json` — datos para la vista Streamlit.
- `model.joblib` — pipeline serializado (TF-IDF + regresión logística).
- `transformer_benchmark.json` — benchmark externo opcional.
