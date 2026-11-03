# Resumen Ejecutivo y Metodológico — Módulo de Clustering
**Proyecto Integrador:** DataSmart · Olist — Plataforma de Minería de Datos  
**Institución:** UNMSM — Facultad de Ingeniería de Sistemas e Informática (2026)  
**Módulo:** Segmentación de Clientes (`feature/clustering`)  
**Responsable:** Científico de Datos / Ingeniero de ML  

---

## 1. Pregunta y Objetivo de Negocio
- **Pregunta central:** ¿Qué tipos de clientes existen en Olist y cuáles aportan el mayor valor económico a la plataforma?
- **Objetivo analítico:** Identificar agrupaciones naturales homogéneas de clientes mediante técnicas no supervisadas sobre su comportamiento transaccional (Recencia, Frecuencia, Gasto Monetario, Logística y Calidad del Servicio), seleccionar con rigor técnico un único modelo ganador validado contra el contrato común (`docs/module_contract.schema.json`) y traducir los segmentos a estrategias comerciales de retención, reactivación y fidelización.

---

## 2. Hechos del Dataset y Auditoría de Datos
En concordancia con los hallazgos del EDA (`docs/hallazgos_eda.json`):

1. **Identificador del Cliente Real:**
   - La tabla `customers` contiene 99,441 filas asociadas a `customer_id` (generado por pedido).
   - Se agrupó **estrictamente por `customer_unique_id`** (93,358 clientes únicos con pedidos entregados válidos), evitando la sobreestimación del tamaño de la base.
2. **Distribución Real de Frecuencia ($F$):**
   - El **97.00%** de los clientes (90,557 usuarios) realizó exactamente **un solo pedido**.
   - Solo el **3.00%** (2,801 clientes) registró recompra ($F \ge 2$), alcanzando un máximo de 15 pedidos.
   - *Implicancia metodológica:* La varianza cuasi-nula en $F$ debilita la segmentación RFM tradicional por cuantiles uniformes. Se implementó la transformación $\log(1 + (F - 1))$, mapeando a los compradores únicos al origen ($0.0$) sin distorsionar el centroide, aislando a los recompradores en su propia dimensión.
3. **Auditoría de LRFM (Longitud / Antigüedad $L$):**
   - $L_{\text{span}}$ (días entre la primera y la última compra) es $0$ para el **97.87%** de los clientes (distribución cuasi-degenerada tipo delta de Dirac).
   - $L_{\text{tenure}}$ (días desde la primera compra hasta la fecha de corte) presenta una correlación de Pearson de **$r = 0.9866$** con la Recencia ($R$).
   - *Decisión técnica:* Se descarta formalmente $L$ del agrupamiento para no introducir colinealidad espuria ni redundancia dimensional.
4. **Consistencia de la Métrica Monetaria ($M$):**
   - Se auditó la suma de `order_payments.payment_value` frente a `price + freight_value`.
   - El **99.68%** de los pedidos presenta una concordancia cercana ($< 1.00$ BRL) y 99.55% coincidencia exacta ($< 0.01$ BRL). Se adoptó `payment_value` como fuente monetaria oficial por representar el flujo de caja neto efectivamente desembolsado por el cliente.
5. **Periodo Temporal:**
   - Compras realizadas entre el **04/09/2016** y el **17/10/2018**.
   - Fecha de referencia calculada: **18/10/2018** (último timestamp observado $+ 1$ día).

---

## 3. Metodología de Preparación y Modelado

### Ingeniería de Características
- **Conjunto A (RFM Transformado):** $\log(1 + R)$, $\log(1 + (F - 1))$ y $\log(1 + M)$. Escalado con `StandardScaler`.
- **Conjunto B (RFM Extendido - 9 variables):** Conjunto A más `review_score`, `freight_ratio` (flete / monetario), $\log(1 + \text{n\_items})$, `installments` (cuotas), `is_late` (indicador de entrega tardía) y $\log(1 + \text{n\_categories})$.
- **CLV:**
  - *Histórico:* Valor acumulado observado $\sum \text{payment\_value}$.
  - *Proyectado Simple (Ilustrativo):* $\text{CLV}_{\text{proy}} = 0.15 \times M$ (asumiendo margen de contribución del 15% sobre horizonte esperado de 2 años).
- **Curva ABC (Pareto):** Clasificación según contribución acumulada a ingresos ($A \le 80\%$, $B \in (80\%, 95\%]$, $C > 95\%$).

### Partición y Algoritmos Evaluados
- Partición estratificada **80% Entrenamiento / 20% Prueba** (`random_state=42`).
- Algoritmos comparados: **K-Means** ($k=2 \dots 6$), **DBSCAN** ($\text{eps} \in [0.6, 1.0]$, $\text{min\_samples}=25$) y **Hierarchical Clustering (Ward)** sobre submuestra de 8,000 clientes con asignación a prueba por `NearestCentroid`.
- Evaluación con **Silhouette Score** (muestreo representativo de 10,000 puntos para optimizar $O(N^2)$), **Davies-Bouldin Index** y **Calinski-Harabasz Index**.

---

## 4. Tabla Comparativa de Algoritmos y Espacios de Características

| Algoritmo | Espacio de Variables | Nº Clusters | Silhouette (Train) | Silhouette (Test) | Davies-Bouldin | Calinski-Harabasz | Ruido (%) | Tiempo (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **K-Means ($k=4$) ★ Ganador** | **Conjunto A (RFM)** | **4** | **0.3919** | **0.4005** | **0.7526** | **61,421.3** | **0.0%** | **0.12** |
| K-Means ($k=3$) | Conjunto A (RFM) | 3 | 0.4185 | 0.4222 | 0.8338 | 45,133.4 | 0.0% | 0.11 |
| K-Means ($k=2$) | Conjunto A (RFM) | 2 | 0.7276 | 0.7178 | 0.4495 | 37,024.0 | 0.0% | 0.10 |
| Ward Jerárquico ($k=4$) | Conjunto A (RFM) | 4 | 0.3322 | 0.3880 | 0.8120 | 48,920.1 | 0.0% | 2.85 |
| Ward Jerárquico ($k=3$) | Conjunto A (RFM) | 3 | 0.3229 | 0.3208 | 0.8450 | 43,105.2 | 0.0% | 2.70 |
| DBSCAN ($\text{eps}=0.6$) | Conjunto A (RFM) | 2 | 0.6210 | N/A | 0.6120 | 3,120.4 | 1.22% | 0.85 |
| DBSCAN ($\text{eps}=1.0$) | Conjunto A (RFM) | 2 | 0.6840 | N/A | 0.5410 | 4,210.1 | 0.49% | 0.80 |
| K-Means ($k=4$) | Conjunto B (Extendido) | 4 | 0.2127 | 0.2151 | 1.3532 | 17,902.4 | 0.0% | 0.25 |
| Ward Jerárquico ($k=4$) | Conjunto B (Extendido) | 4 | 0.1719 | 0.2092 | 1.4820 | 14,210.5 | 0.0% | 3.10 |
| DBSCAN ($\text{eps}=0.6$) | Conjunto B (Extendido) | 6 | 0.1540 | N/A | 1.6210 | 1,890.2 | 41.31% | 1.15 |

---

## 5. Justificación del Modelo Ganador
Se seleccionó de forma unívoca **K-Means ($k=4$) sobre el Conjunto A (RFM transformado)** por las siguientes razones:

1. **Superación de la Maldición de la Dimensionalidad:** Al añadir variables transaccionales secundarias en el Conjunto B (9 dimensiones), la densidad euclidiana se dispersó por ruido en reseñas y demoras (débil correlación con el gasto), provocando que el Silhouette cayera de **0.4005 a 0.2151** y Davies-Bouldin empeorara de **0.75 a 1.35**.
2. **Falla estructural de DBSCAN:** Debido a la concentración del 97% en una sola compra, el espacio no presenta valles de baja densidad; DBSCAN agrupa al 98% en un macro-cluster indistinguible, resultando comercialmente inútil.
3. **Consistencia y Estabilidad Sobresaliente:** K-Means ($k=4$) demostró una estabilidad de **$\text{ARI} = 0.9947$** evaluado a través de 10 semillas aleatorias independientes.
4. **Ruptura frente a Cuantiles Ingenuos:** El $\text{ARI} = 0.1617$ frente a la segmentación tradicional por quintiles confirma que el modelo no supervisado descubre agrupaciones geométricas intrínsecas en vez de forzar particiones uniformes artificiales.

---

## 6. Perfilado de Segmentos y Estrategia Comercial

| Segmento | % Base Clientes | % Ingresos Totales | Recencia Media | Frecuencia Media | Gasto Medio | Satisfacción Promedio | Tasa Entregas Tardías | Acción de Negocio Prioritaria |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Recientes / Nuevos** | 17.3% (16,173) | 14.0% (2.16M R$) | 42.3 días | 1.00 | 133.3 R$ | **4.32 ★** | **6.04%** | **Campaña de Segundo Pedido:** Cupón de bienvenida con caducidad a 30 días para aprovechar la ventana de alta satisfacción reciente. |
| **Alto Valor en Riesgo** | 34.4% (32,111) | **61.7% (9.52M R$)** | 272.4 días | 1.00 | **296.3 R$** | 4.04 ★ | **9.46%** | **Estrategia Win-Back VIP:** Contacto personalizado, disculpas por fricciones logísticas previas y garantía de entrega rápida bonificada. |
| **Recompradores Leales (VIP)** | **3.0% (2,801)** | 5.6% (0.86M R$) | 220.3 días | **2.11** | 308.6 R$ | 4.21 ★ | 7.01% | **Programa de Lealtad y Referidos:** Membresía de envíos prioritarios y recompensas por recomendación (referrals) para elevar LTV. |
| **Ocasionales de Bajo Valor** | **45.3% (42,273)** | 18.7% (2.89M R$) | 287.8 días | 1.00 | 68.3 R$ | 4.17 ★ | 8.04% | **Cross-Selling de Bajo Costo:** Activaciones automáticas por email/push con umbrales de envío gratis para duplicar el tamaño de cesta. |

> **Conexión directa con el EDA:** El segmento *Alto Valor en Riesgo*, que aporta el **61.7% de los ingresos históricos**, exhibe la mayor tasa de entregas tardías (9.46%) y la menor calificación promedio (4.04★). Esto valida empíricamente el hallazgo del EDA: las demoras logísticas castigan la satisfacción (14.69% reseñas negativas globales y 54% en demoras) y destruyen la posibilidad de un segundo pedido en clientes de ticket elevado.

---

## 7. Explicabilidad Analítica (Árbol Sustituto)
Para dotar al modelo de explicabilidad en producción sin perder la capacidad de partición de K-Means, se entrenó un árbol de decisión sustituto (`DecisionTreeClassifier`, `max_depth=3`) sobre las variables naturales ($R, F, M$), alcanzando una **exactitud del 96.96%**.

### Reglas Legibles Extraídas:
```text
|--- Monetary <= 122.29 R$
|   |--- Recency <= 90.50 días  --> Segmento: Recientes / Nuevos
|   |--- Recency >  90.50 días
|   |   |--- Frequency <= 1.50   --> Segmento: Ocasionales de Bajo Valor
|   |   |--- Frequency >  1.50   --> Segmento: Recompradores Leales (VIP)
|--- Monetary >  122.29 R$
|   |--- Recency <= 82.50 días   --> Segmento: Recientes / Nuevos
|   |--- Recency >  82.50 días
|   |   |--- Frequency <= 1.50   --> Segmento: Alto Valor en Riesgo
|   |   |--- Frequency >  1.50   --> Segmento: Recompradores Leales (VIP)
```

---

## 8. Figuras Recomendadas para la Presentación
1. **Gráfico de Barras Doble (Clientes vs Ingresos):** Ilustra visualmente que el 34.4% de clientes (*Alto Valor en Riesgo*) concentra el 61.7% del capital, justificando concentrar el presupuesto en retención.
2. **Dispersión PCA 2D:** Muestra la separación geométrica limpia de los 4 cuadrantes del espacio RFM.
3. **Radar Multidimensional de Centroides:** Muestra el contraste relativo de Recencia invertida, Frecuencia y Monetario.
4. **Boxplots Comparativos de Recencia y Gasto:** Evidencia la separación distributiva no superpuesta entre grupos.

---

## 9. Limitaciones Metodológicas
1. **Asimetría extrema en Frecuencia:** El 97% de compradores únicos inhabilita el uso confiable de modelos bayesianos o probabilísticos tipo BG/NBD para CLV.
2. **Horizonte temporal histórico cerrado:** Pedidos culminan en octubre de 2018. En un despliegue productivo vivo debe recalcularse la recencia con ventanas móviles de cohorte.
3. **Ausencia de telemetría web:** No se dispone de navegación, visitas ni carritos abandonados; solo se observa el checkout exitoso.
