# DataSmart · Olist — Informe técnico consolidado

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

## Plataforma

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

## Asociación

**Pregunta de negocio.** ¿Qué productos o categorías se compran conjuntamente?

**Modelo ganador.** FP-Growth - Canasta por Pedido · versión 1.0.0 · generado 2026-11-03T23:50:17

**Periodo.** 2016-09-04 a 2018-10-17

| Métrica | Valor |
|---|---|
| rules_count | 10 |
| max_lift | 15.1457 |
| avg_lift | 4.3316 |
| avg_confidence | 0.0453 |
| min_support_used | 0.0000 |
| exec_time_sec | 0.7105 |

**Conclusiones de negocio.**

- Especializacion extrema: El 99.2% de los pedidos contiene una sola categoria, reflejando compras puntuales de necesidad inmediata y escasa navegacion horizontal.
- Afinidad en moda infantil: La regla fashion_childrens_clothes -> fashion_bags_accessories alcanza Lift de 15.15 y confianza de 28.6%, siendo la mayor oportunidad de venta cruzada en checkout.
- Sinergia en hogar y descanso: home_confort y bed_bath_table muestran asociacion bidireccional (Lift 1.14, confianza 11.0%), ideal para bundles y promociones por segundo articulo.
- Desacoplamiento en recompras: Con solo 3.0% de recompra historica, la estrategia de empaquetamiento debe priorizar la conversion inmediata sobre campanas diferidas.

**Limitaciones.**

- Baja tasa multicategoria: Solo el 0.81% de los pedidos combina categorias, requiriendo soportes minimos bajos para identificar reglas no triviales.
- Granularidad por categoria: El analisis a nivel de categoria agregada no distingue afinidades directas entre SKUs o marcas especificas.
- Simetria temporal: La transaccion no registra el orden de incorporacion al carrito, asumiendo simultaneidad en la compra.
- Sin telemetria de navegacion: Solo se evaluan compras aprobadas, omitiendo sesiones de navegacion, comparaciones o carritos abandonados.

## Clustering

**Pregunta de negocio.** ¿Qué segmentos de clientes existen y cuáles aportan mayor valor?

**Modelo ganador.** K-Means (k=4) - Conjunto A (RFM) · versión 1.0.0 · generado 2026-11-03T17:39:45

**Periodo.** 2016-09-04 a 2018-10-17

| Métrica | Valor |
|---|---|
| silhouette_score | 0.4003 |
| davies_bouldin_index | 0.7537 |
| calinski_harabasz_index | 60,934.70 |
| stability_ari_10_runs | 0.9947 |
| baseline_rfm_quantiles_ari | 0.1617 |
| explainability_tree_accuracy | 0.9696 |

**Conclusiones de negocio.**

- Concentración extrema de recompra: Apenas el 3.0% de los clientes ha realizado más de un pedido ('Recompradores Leales VIP'). La casi-constancia de F debilita el RFM tradicional pero convierte la retención en la palanca de mayor ROI.
- Deserción de clientes de alto gasto: El segmento 'Alto Valor en Riesgo' acumula el 61.7% de los ingresos históricos pero presenta 272 días de inactividad, una satisfacción inferior (4.04★) y la mayor tasa de retrasos (9.46%), confirmando la correlación hallada en el EDA entre retrasos logísticos y abandono.
- Ventana crítica de reactivación: El segmento 'Recientes / Nuevos' (17.3% de la base) posee recencia media de 42 días, satisfacción máxima (4.32★) y mínima tasa de retrasos (6.04%), conformando la cohorte óptima para programas de bienvenida y estímulo a la segunda compra.
- Masa ocasional de ticket bajo: El 45.3% de los clientes pertenece a 'Ocasionales de Bajo Valor' (gasto medio R$ 68.3), para los cuales se requieren incentivos por umbral de envío gratuito.

**Limitaciones.**

- Distribución degenerada en frecuencia: El 97.0% de los clientes registra exactamente una compra, limitando la aplicabilidad de modelos probabilísticos tradicionales como BG/NBD y requiriendo transformaciones asimétricas.
- Horizonte temporal estático (2016-2018): Los datos culminan en octubre de 2018. Sin ingesta en tiempo real, la recencia aumenta artificialmente para toda la base si no se calcula respecto a ventanas relativas de cohorte.
- Inexistencia de telemetría y navegación: La plataforma solo registra pedidos aprobados; la ausencia de eventos de visualización, sesiones web o carritos abandonados impide segmentar según intención previa a la compra.
- Sensibilidad al flete interestatal: En Brasil el costo de envío representa una fricción decisiva que no se deriva puramente de la preferencia del consumidor sino de la distancia geográfica al vendedor.

## Series temporales

**Pregunta de negocio.** ¿Cómo evolucionarán las ventas en los próximos periodos?

**Modelo ganador.** sarima · versión 1.0.0 · generado 2026-11-02T07:41:52

**Periodo.** 2017-01-23 a 2018-08-22 (577 dias)

**Validación.** Walk-forward de 6 pliegues x 30 dias sobre 577 dias

| Métrica | Valor |
|---|---|
| MAE | 6,906.94 |
| RMSE | 8,547.58 |
| sMAPE_% | 19.0800 |
| MAPE_% | 21.5300 |
| MAE_baseline | 7,263.44 |
| mejora_vs_baseline_% | 4.9000 |

**Modelos comparados.**

- {"modelo": "sarima", "MAE": 6906.94, "RMSE": 8547.58, "MAPE": 21.53, "sMAPE": 19.08, "mejora_MAE_%": 4.9}
- {"modelo": "sarimax", "MAE": 7423.94, "RMSE": 9100.35, "MAPE": 23.7, "sMAPE": 20.38, "mejora_MAE_%": -2.2}
- {"modelo": "naive_estacional", "MAE": 7263.44, "RMSE": 9217.12, "MAPE": 21.5, "sMAPE": 20.96, "mejora_MAE_%": 0.0}
- {"modelo": "prophet", "MAE": 8060.97, "RMSE": 9711.97, "MAPE": 25.12, "sMAPE": 22.33, "mejora_MAE_%": -11.0}

**Por qué ganó.**

- Es el unico de los cuatro que supera al baseline naive estacional (6907 frente a 7263 de MAE, 4.9% mejor).
- Captura la estacionalidad semanal, que es real: el sabado vende 39% menos que el martes.
- Modelar log1p en lugar del nivel reduce el error porque la serie es asimetrica y crece a lo largo del periodo.
- El orden se elige por AIC sobre el train de cada pliegue, sin mirar el test.
- SARIMAX no mejora porque sus regresores de calendario valen cero en los seis pliegues de validacion; ademas el dataset tiene un solo Black Friday, por lo que su efecto no puede estimarse y validarse a la vez.
- Prophet quedo por debajo del baseline en las seis configuraciones probadas: 19 meses son pocos para su modelo de tendencia con changepoints.

**Conclusiones de negocio.**

- El pronostico a 30 dias permite anticipar la demanda para planificar inventario y capacidad logistica.
- La estacionalidad semanal es el patron dominante: el fin de semana cae hasta 39% frente al martes, lo que ordena la programacion de personal y despachos.
- El KPI comprometido en el 1er avance (MAPE semanal <= 20%) ya lo cumple un baseline trivial con 16.5%, asi que la vara real es superar ese baseline.

**Limitaciones.**

- La serie util tiene 19 meses: no permite estimar estacionalidad anual, solo la semanal (s=7).
- Un unico Black Friday en todo el periodo impide modelar eventos especiales.
- El arranque de 2017 y los dias posteriores al 22-ago-2018 se excluyen por no ser representativos (lanzamiento y corte de extraccion).
- El dataset termina en 2018: el pronostico es una demostracion metodologica, no una proyeccion operativa vigente.

## Grafos

**Pregunta de negocio.** ¿Qué productos, categorías o vendedores están más conectados?

**Modelo ganador.** Louvain · versión 1.0.0 · generado 2026-11-09T20:56:49

**Periodo.** 2016-09-04 a 2018-09-03

| Métrica | Valor |
|---|---|
| modularidad_ganador | 0.7596 |
| comunidades_ganador | 21 |
| cobertura_ganador | 1.0000 |
| nodos_grafo_principal | 3169 |
| aristas_grafo_principal | 6616 |
| conectividad_grafo_principal_% | 100.00 |
| componente_gigante_producto_vendedor_% | 26.9400 |
| pedidos_con_2_o_mas_productos | 3236 |
| aristas_co_compra | 4058 |

**Conclusiones de negocio.**

- El grafo categoría–vendedor está 100.0% conectado, por lo que permite analizar todas las categorías y vendedores sin descartar nodos.
- health_beauty es la categoría con más vendedores (492), seguida de sports_leisure (481), lo que indica mercados con mayor competencia.
- La co-compra de categorías más frecuente es bed_bath_table + furniture_decor (70 pedidos), una oportunidad de venta cruzada.
- Solo 3,236 pedidos incluyen dos o más productos distintos, por lo que el grafo de co-compra directa es disperso y debe complementarse con reglas de asociación.

**Limitaciones.**

- El grafo de co-compra producto–producto es disperso: el componente gigante cubre 8.15% de sus nodos.
- Las aristas categoría–vendedor representan presencia, no volumen de ventas ni ingresos.
- El periodo analizado (2016-09-04 a 2018-09-03) no refleja el comportamiento actual del mercado.
- Louvain es estocástico; se fijó random_state=42 para reproducibilidad.
- Betweenness se calculó de forma aproximada (k=100) sobre el componente gigante.

## Regresión

**Pregunta de negocio.** ¿Cuánto costará el flete de un pedido?

**Modelo ganador.** Sistema híbrido XGBoost · versión 3.0 · generado 2026-11-23T22:54:28

| Métrica | Valor |
|---|---|
| MAE | 5.4491 |
| RMSE | 15.0572 |
| R2 | 0.6589 |
| sesgo | 2.3061 |
| WAPE | 21.9822 |
| MAE_fletes_altos | 15.0929 |
| sesgo_fletes_altos | 10.0825 |
| precision_clase_alta | 0.8053 |
| recall_clase_alta | 0.8448 |
| F1_clase_alta | 0.8246 |

## Clasificación

**Pregunta de negocio.** ¿Qué pedidos tienen riesgo de recibir una reseña negativa?

**Modelo ganador.** HistGradientBoosting (sin tratamiento, ajustado) · versión 1.0.0 · generado 2026-11-03T20:31:04

**Periodo.** Entrenamiento 2017-01 a 2018-05; prueba 2018-06 a 2018-08

| Métrica | Valor |
|---|---|
| roc_auc | 0.7241 |
| pr_auc | 0.3999 |
| f1 | 0.4124 |
| precision | 0.5925 |
| recall | 0.3162 |
| balanced_accuracy | 0.6451 |
| brier | 0.0783 |
| umbral | 0.2907 |
| prevalencia_test | 0.1067 |
| cv_pr_auc | 0.5543 |
| n_train | 78801 |
| n_test | 18789 |

**Conclusiones de negocio.**

- Priorización accionable: contactar el 10% de pedidos con mayor riesgo captura el 40.0% de las reseñas negativas (lift 4.0x); con el 30% se captura el 59.9%.
- La promesa de entrega es la palanca principal: los pedidos pendientes al enviar la encuesta tienen 74.0% de negativas y los entregados tarde 19.5%, frente a 9.2% de los puntuales. Variables más influyentes: pendiente_en_encuesta, dias_retraso, n_items.
- Con el umbral 0.29 el modelo detecta el 31.6% de las negativas con una precisión del 59.3% (PR-AUC 0.400 frente a una prevalencia de 0.107).
- Al momento de la compra el riesgo es poco anticipable (PR-AUC 0.181 vs 0.400 con seguimiento logístico): la acción preventiva debe apoyarse en el monitoreo del despacho y la entrega, no solo en el perfil del pedido.

**Limitaciones.**

- Momento de predicción: el modelo se aplica al enviar la encuesta; para usarlo antes (al comprar o al despachar) el desempeño cae, como muestra el escenario complementario.
- Deriva temporal: la tasa de negativas varía entre meses (picos en nov-2017 y feb–mar 2018); el modelo debe reentrenarse periódicamente y vigilar la prevalencia (test: 10.7%).
- Brecha validación–test: la PR-AUC baja de 0.554 (CV) a 0.400 (test) sobre todo porque la prevalencia cae de 15.0% a 10.7%; el lift sobre el azar se mantiene, pero la ROC-AUC también desciende, señal de deriva a monitorear.
- Sin variables de texto ni historial del vendedor: el comentario de la reseña es posterior a la predicción y el historial de reputación del vendedor no se incluyó para evitar fuga temporal.

## Sentimientos

**Pregunta de negocio.** ¿Qué sentimiento expresan las reseñas y qué temas explican la insatisfacción?

**Modelo ganador.** Regresión Logística + TF-IDF palabra/carácter · versión 1.0.0 · generado 2026-11-23T20:40:36

**Periodo.** 2016-10-02 a 2018-08-31

| Métrica | Valor |
|---|---|
| macro_f1 | 0.6769 |
| weighted_f1 | 0.8349 |
| accuracy | 0.8368 |
| balanced_accuracy | 0.6815 |
| cohen_kappa | 0.8309 |
| negative_pr_auc | 0.8786 |
| negative_roc_auc | 0.9542 |
| negative_f1 | 0.8363 |
| negative_precision | 0.7765 |
| negative_recall | 0.9059 |
| umbral_negativo | 0.3745 |
| macro_f1_cv | 0.6770 |
| qwk_5_estrellas | 0.8410 |
| exactitud_5_estrellas | 0.6744 |
| off_by_one_5_estrellas | 0.8876 |
| n_total | 35044 |
| n_train | 28035 |
| n_test | 7009 |
| tasa_negativa_test | 0.2928 |
| transformer_macro_f1 | 0.6486 |

**Conclusiones de negocio.**

- El modelo detecta reseñas negativas con PR-AUC 0.879 y ROC-AUC 0.954; con el umbral 0.37 logra precisión 77.6% y recall 90.6%.
- El sentimiento está muy ligado a la logística: las reseñas de pedidos con entrega tardía tienen 72.8% de negativas predichas frente a 23.6% de los entregados a tiempo, y 89.2% cuando no hay entrega registrada.
- Entre las 15 categorías con más reseñas, la de mayor tasa de negativas es baby (37.2% de 191 reseñas), una prioridad para revisar producto, embalaje y expectativas.
- La clase neutra es la más difícil (F1 0.279): muchas reseñas de 3 estrellas son ambivalentes o descriptivas. Para decisiones operativas conviene usar la probabilidad de negativa y no solo la clase argmax.
- El transformer externo zero-shot alcanzó macro-F1 0.649 en la muestra balanceada, frente a 0.648 del mejor modelo clásico. Empate técnico: el clásico se despliega por latencia, costo e interpretabilidad, y fine-tunear BERTimbau en GPU queda como mejora pendiente.

**Limitaciones.**

- La nota de 1–5 estrellas es una etiqueta débil: mide satisfacción global y no sentimiento puro; las reseñas de 3 estrellas son especialmente ambiguas.
- Solo ~41% de las reseñas tienen comentario. La muestra está sesgada hacia clientes que deciden escribir, que suelen ser los muy satisfechos o los muy insatisfechos.
- El texto es portugués coloquial, con errores de tipeo, ironía, emojis y abreviaturas; los n-gramas de carácter ayudan, pero no resuelven la ironía ni el sarcasmo.
- Se eliminaron duplicados exactos de texto para evitar fuga de información; near-duplicates y plantillas casi idénticas pueden seguir inflando levemente la evaluación.
- La partición train/test es aleatoria y estratificada, no temporal: mide generalización a reseñas similares, no deriva futura. Para producción conviene añadir un holdout temporal.
- Las tasas por categoría se calculan sobre el test y algunas categorías tienen menos de 200 reseñas; son señales exploratorias, no estimaciones precisas.
- El benchmark con transformer es zero-shot sobre un modelo entrenado en otro e-commerce (B2W). No se fine-tuneó BERTimbau en Olist por costo computacional; con GPU el techo sería mayor.
- El modelo se entrenó con reseñas de 2016–2018. Cambios de catálogo, logística o lenguaje requieren monitoreo de deriva y reentrenamiento periódico.
- Es un modelo descriptivo, no causal: una asociación entre retraso y negatividad no prueba que el retraso sea la única causa.

## Monitoreo y mantenimiento

Último reporte: 2026-11-24T15:51:40+00:00. Resumen: OK 7, Vigilar 0, Alerta 0, Sin datos 0.

| Módulo | Versión | Métrica | Referencia | Actual | Estado | Motivo |
|---|---|---|---|---|---|---|
| Clasificación | 1.0.0 | PR-AUC | 0.3999 | 0.3683 | OK | Desempeño y distribución estables respecto a la referencia. |
| Regresión | 3.0 | MAE | 5.4491 | 5.4491 | OK | Desempeño y distribución estables respecto a la referencia. |
| Sentimientos | 1.0.0 | macro-F1 | 0.6769 | 0.6245 | OK | Desempeño y distribución estables respecto a la referencia. |
| Series temporales | 1.0.0 | MAE | 6,906.94 | 6,096.71 | OK | Desempeño y distribución estables respecto a la referencia. |
| Clustering | 1.0.0 | — | — | — | OK | Sin predicciones fechadas: se vigila versión y antigüedad del artefacto. |
| Grafos | 1.0.0 | — | — | — | OK | Sin predicciones fechadas: se vigila versión y antigüedad del artefacto. |
| Asociación | 1.0.0 | — | — | — | OK | Sin predicciones fechadas: se vigila versión y antigüedad del artefacto. |

Reglas: PSI < 0.10 estable, 0.10–0.25 vigilar, > 0.25 alerta; degradación de la métrica principal ≥ 10 % vigilar, ≥ 20 % alerta; artefactos de más de 90 días se marcan para revisión. Un módulo en Alerta se reentrena con `python -m src.monitoring --retrain`; cada corrida queda en `artifacts/monitoring/audit_log.csv`.


_Generado automáticamente el 2026-11-24 15:51 UTC con `python scripts/build_report.py`._
