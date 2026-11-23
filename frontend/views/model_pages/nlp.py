"""
Vista Streamlit del módulo de Análisis de Sentimientos — DataSmart Olist.
==========================================================================
Responsable: Módulo NLP / Sentimientos (rama feature/nlp)
Proyecto Integrador · Minería de Datos · UNMSM FISI 2026

La vista consume artefactos precalculados y permite analizar una reseña en
tiempo real. No entrena modelos dentro de Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components import PALETTE, section_intro, style_figure
from frontend.data_client import predict_nlp
from frontend.views.module_template import render_module

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT / "artifacts" / "nlp"

CLASS_COLORS = {
    "Negativo": "#C8553D",
    "Neutro": "#E0A458",
    "Positivo": PALETTE["bondi"],
}
LABEL_ORDER = ["Negativo", "Neutro", "Positivo"]


# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------


@st.cache_data
def _load_chart_data() -> dict | None:
    path = ARTIFACTS_DIR / "chart_data.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as source:
        return json.load(source)


@st.cache_data
def _load_results() -> pd.DataFrame | None:
    path = ARTIFACTS_DIR / "results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _predict(text: str) -> dict:
    """Usa la API si está disponible; si no, ejecuta el modelo local."""
    try:
        return predict_nlp(text)
    except (HTTPError, URLError, TimeoutError, ValueError):
        from backend.services.sentiment import predict_sentiment

        return predict_sentiment(text)


# ---------------------------------------------------------------------------
# Guía de métricas
# ---------------------------------------------------------------------------


def _render_metrics_guide() -> None:
    with st.expander("Cómo leer las métricas de sentimiento", expanded=False):
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(
                """
                **Macro-F1** es la métrica principal: promedia el F1 de negativo, neutro y
                positivo con el mismo peso, así que no permite que la clase positiva (la más
                frecuente) esconda un mal desempeño en las otras dos.

                **Accuracy** es la proporción de aciertos globales; se reporta por transparencia,
                pero con clases desbalanceadas puede ser engañosa. **Balanced accuracy** promedia
                el recall de las tres clases.
                """
            )
        with col_right:
            st.markdown(
                """
                **PR-AUC de la clase negativa** mide qué tan bien se ordenan las reseñas negativas
                cuando son minoría. Es más informativa que ROC-AUC para priorizar casos.

                **QWK (kappa ponderada cuadrática)** evalúa la nota de 1 a 5 como problema ordinal:
                castiga más equivocarse por 4 estrellas que por 1. **±1 estrella** indica cuántas
                predicciones quedan a lo sumo a una nota de la real.
                """
            )


# ---------------------------------------------------------------------------
# Resumen del modelo
# ---------------------------------------------------------------------------


def _render_summary(chart_data: dict) -> None:
    section_intro(
        "Resumen del modelo",
        "Comparación de algoritmos, matriz de confusión y desempeño por clase.",
    )

    distribution = chart_data.get("distribucion_clases", {})
    rows = []
    for split_name, label in (("train", "Entrenamiento"), ("test", "Prueba")):
        for item in distribution.get(split_name, []):
            rows.append(
                {
                    "Partición": label,
                    "Sentimiento": item["clase"],
                    "Reseñas": item["resenas"],
                    "Porcentaje": item["porcentaje"],
                }
            )
    if rows:
        dist_df = pd.DataFrame(rows)
        fig = px.bar(
            dist_df,
            x="Sentimiento",
            y="Reseñas",
            color="Partición",
            barmode="group",
            text="Porcentaje",
            color_discrete_sequence=[PALETTE["sinbad"], PALETTE["bondi"]],
            category_orders={"Sentimiento": LABEL_ORDER},
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig = style_figure(fig, title="Distribución de clases", height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    comparison = pd.DataFrame(chart_data.get("comparacion_modelos", []))
    if not comparison.empty:
        plot_df = comparison.dropna(subset=["macro_f1_balanceado"]).copy()
        plot_df = plot_df.sort_values("macro_f1_balanceado")
        fig = px.bar(
            plot_df,
            x="macro_f1_balanceado",
            y="modelo",
            orientation="h",
            color="tipo",
            text="macro_f1_balanceado",
            color_discrete_sequence=[PALETTE["bondi"], PALETTE["eden"]],
            labels={"macro_f1_balanceado": "Macro-F1 en muestra balanceada", "modelo": ""},
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig = style_figure(fig, title="Comparación de modelos (muestra balanceada)", height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        display = comparison.copy()
        display = display.rename(
            columns={
                "modelo": "Modelo",
                "tipo": "Tipo",
                "macro_f1_cv": "Macro-F1 CV",
                "macro_f1_test": "Macro-F1 test",
                "macro_f1_balanceado": "Macro-F1 balanceado",
                "accuracy_test": "Accuracy test",
                "balanced_accuracy_test": "Balanced accuracy",
                "cohen_kappa_test": "QWK test",
                "negative_pr_auc_test": "PR-AUC negativa",
                "n_test": "n test",
                "tiempo_cv_seg": "Tiempo CV (s)",
            }
        )
        columns = [
            "Modelo",
            "Tipo",
            "Macro-F1 CV",
            "Macro-F1 test",
            "Macro-F1 balanceado",
            "Accuracy test",
            "PR-AUC negativa",
            "n test",
        ]
        display = display[[column for column in columns if column in display.columns]]
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption(
            "El transformer es una referencia externa zero-shot evaluada sobre una muestra "
            "balanceada; no se entrenó con reseñas de Olist ni se despliega en la API."
        )

    col_confusion, col_classes = st.columns([1.1, 1])
    with col_confusion:
        matrix = np.array(chart_data.get("matriz_confusion", []), dtype=int)
        if matrix.size == 9:
            annotations = np.array(
                [
                    [f"{value:,}" for value in row]
                    for row in matrix
                ]
            )
            fig = px.imshow(
                matrix,
                x=LABEL_ORDER,
                y=LABEL_ORDER,
                text_auto=False,
                color_continuous_scale=[
                    [0.0, PALETTE["white"]],
                    [1.0, PALETTE["bondi"]],
                ],
                labels={"x": "Predicción", "y": "Clase real", "color": "Reseñas"},
            )
            fig.update_traces(text=annotations, texttemplate="%{text}")
            fig = style_figure(fig, title="Matriz de confusión (test)", height=360)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with col_classes:
        per_class = chart_data.get("metricas_por_clase", {})
        if per_class:
            class_df = pd.DataFrame(per_class).T.reset_index()
            class_df.columns = ["Clase", "Precisión", "Recall", "F1", "Soporte"]
            class_df["Clase"] = pd.Categorical(class_df["Clase"], LABEL_ORDER, ordered=True)
            class_df = class_df.sort_values("Clase")
            st.dataframe(
                class_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Precisión": st.column_config.NumberColumn(format="%.3f"),
                    "Recall": st.column_config.NumberColumn(format="%.3f"),
                    "F1": st.column_config.NumberColumn(format="%.3f"),
                    "Soporte": st.column_config.NumberColumn(format="%d"),
                },
            )
        threshold = chart_data.get("umbral_negativo", {})
        if threshold:
            st.markdown(
                f"""
                <div class='module-note'>
                    <strong>Umbral operativo de negativa:</strong> {threshold.get('threshold', 0):.3f}.
                    Se eligió maximizando F1 con predicciones out-of-fold del entrenamiento,
                    nunca con el test. PR-AUC OOF: {threshold.get('oof_pr_auc', 0):.3f}.
                </div>
                """,
                unsafe_allow_html=True,
            )

    calibration = pd.DataFrame(chart_data.get("calibracion", []))
    if not calibration.empty:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Calibración perfecta",
                line=dict(color="grey", dash="dash", width=1),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=calibration["probabilidad_media"],
                y=calibration["fraccion_negativas"],
                mode="lines+markers",
                name="Modelo",
                line=dict(color="#C8553D", width=2.4),
                marker=dict(size=8),
            )
        )
        figure = style_figure(
            figure,
            title="Calibración de la probabilidad de negativa",
            height=340,
        )
        figure.update_xaxes(title="Probabilidad predicha (media del bin)", tickformat=".0%")
        figure.update_yaxes(title="Fracción real de negativas", tickformat=".0%")
        st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
        st.caption(
            "Una curva cercana a la diagonal indica que la probabilidad puede leerse como "
            "frecuencia esperada. La ponderación por clase y los umbrales ajustados pueden desviarla."
        )


# ---------------------------------------------------------------------------
# Analizador interactivo
# ---------------------------------------------------------------------------


def _render_analyzer() -> None:
    section_intro(
        "Analizador de reseñas",
        "Escribe una reseña en portugués y consulta el sentimiento, la confianza y las probabilidades.",
    )

    examples = {
        "Negativa": "O produto chegou com um arranhão e a entrega atrasou duas semanas. Muito ruim.",
        "Neutra": "O produto é ok, chegou no prazo, mas a embalagem veio amassada. Ainda não sei se recomendo.",
        "Positiva": "Produto excelente, chegou antes do prazo e em perfeito estado. Recomendo muito!",
    }

    if "nlp_text" not in st.session_state:
        st.session_state["nlp_text"] = examples["Negativa"]

    example_cols = st.columns(3)
    for column, (label, text) in zip(example_cols, examples.items()):
        with column:
            if st.button(f"Ejemplo {label.lower()}", use_container_width=True):
                st.session_state["nlp_text"] = text

    text = st.text_area(
        "Reseña en portugués",
        key="nlp_text",
        height=150,
        placeholder="Ex.: O produto chegou rápido, mas veio com um defeito.",
    )

    if st.button("Analizar sentimiento", type="primary"):
        try:
            with st.spinner("Ejecutando el modelo..."):
                st.session_state["nlp_result"] = _predict(text)
        except Exception as exc:  # pragma: no cover - depende del entorno
            st.error(f"No se pudo analizar la reseña: {exc}")

    result = st.session_state.get("nlp_result")
    if not result:
        st.info("Elige un ejemplo o escribe una reseña para ver la predicción.")
        return

    col_label, col_confidence = st.columns(2)
    col_label.metric("Sentimiento predicho", result["sentimiento"])
    col_confidence.metric("Confianza", f"{result['confianza'] * 100:.1f}%")

    probabilities = result.get("probabilidades", {})
    if probabilities:
        proba_df = pd.DataFrame(
            [
                {"Sentimiento": label, "Probabilidad": probabilities.get(label, 0.0)}
                for label in LABEL_ORDER
            ]
        )
        fig = px.bar(
            proba_df,
            x="Probabilidad",
            y="Sentimiento",
            orientation="h",
            color="Sentimiento",
            color_discrete_map=CLASS_COLORS,
            text="Probabilidad",
            range_x=[0, 1],
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig = style_figure(fig, title="Probabilidades por clase", height=300)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.caption(
        f"Modelo: {result.get('modelo', '—')} · versión {result.get('version', '—')}. "
        "La predicción se ejecuta en la API o, si no está disponible, con el modelo local."
    )


# ---------------------------------------------------------------------------
# Términos por clase
# ---------------------------------------------------------------------------


def _render_terms(chart_data: dict) -> None:
    section_intro(
        "Términos que explican cada sentimiento",
        "Pesos del modelo lineal sobre el vocabulario de palabras. No son reglas causales: "
        "son asociaciones aprendidas del texto.",
    )
    terms = chart_data.get("terminos", {})
    if not terms:
        st.info("No hay términos interpretables en los artefactos.")
        return

    col_class, col_group = st.columns(2)
    selected_class = col_class.selectbox("Sentimiento", LABEL_ORDER, index=0)
    selected_group = col_group.radio(
        "Dirección del peso",
        ["asociados", "opuestos"],
        horizontal=True,
        format_func=lambda value: "Asociados a la clase" if value == "asociados" else "Opuestos a la clase",
    )
    values = terms.get(selected_class, {}).get(selected_group, [])
    if not values:
        st.info("No hay términos para esta selección.")
        return

    frame = pd.DataFrame(values).sort_values("peso")
    fig = px.bar(
        frame,
        x="peso",
        y="termino",
        orientation="h",
        color_discrete_sequence=[CLASS_COLORS.get(selected_class, PALETTE["bondi"])],
        labels={"peso": "Peso en el modelo", "termino": "Término"},
        text="peso",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig = style_figure(
        fig,
        title=f"Términos {selected_group} a {selected_class.lower()}",
        height=520,
    )
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Cruces de negocio
# ---------------------------------------------------------------------------


def _render_business(chart_data: dict) -> None:
    section_intro(
        "Sentimiento y operación",
        "Tasa de reseñas negativas predichas por estado de entrega, categoría y mes. "
        "El sentimiento se predice con el texto; los cruces usan el tablón de pedidos.",
    )
    insights = chart_data.get("insights_negocio", {})
    summary = insights.get("resumen", {})
    if summary:
        cols = st.columns(3)
        cols[0].metric("Reseñas analizadas", f"{summary.get('resenas', 0):,}")
        cols[1].metric("Negativas predichas", f"{summary.get('tasa_negativa_predicha', 0) * 100:.1f}%")
        cols[2].metric("Confianza media", f"{summary.get('confianza_media', 0) * 100:.1f}%")

    col_delivery, col_category = st.columns(2)
    with col_delivery:
        delivery = pd.DataFrame(insights.get("por_estado_entrega", []))
        if not delivery.empty:
            delivery = delivery.sort_values("tasa_negativa")
            fig = px.bar(
                delivery,
                x="tasa_negativa",
                y="estado_entrega",
                orientation="h",
                color="estado_entrega",
                color_discrete_sequence=[PALETTE["bondi"], PALETTE["sinbad"], "#C8553D"],
                text="tasa_negativa",
                labels={"tasa_negativa": "Tasa de negativas", "estado_entrega": ""},
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig = style_figure(fig, title="Negatividad por entrega", height=340)
            fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with col_category:
        categories = pd.DataFrame(insights.get("por_categoria", []))
        if not categories.empty:
            categories = categories.sort_values("tasa_negativa").tail(12)
            fig = px.bar(
                categories,
                x="tasa_negativa",
                y="categoria_principal",
                orientation="h",
                color="tasa_negativa",
                color_continuous_scale=[
                    [0.0, PALETTE["sinbad"]],
                    [1.0, "#C8553D"],
                ],
                text="tasa_negativa",
                labels={"tasa_negativa": "Tasa de negativas", "categoria_principal": ""},
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig = style_figure(fig, title="Categorías con más negatividad", height=420)
            fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    monthly = pd.DataFrame(insights.get("por_mes", []))
    if not monthly.empty:
        monthly = monthly.sort_values("mes")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly["mes"],
                y=monthly["tasa_negativa"],
                mode="lines+markers",
                name="Tasa de negativas",
                line=dict(color="#C8553D", width=2.4),
            )
        )
        fig = style_figure(fig, title="Evolución mensual de la negatividad predicha", height=340)
        fig.update_yaxes(tickformat=".0%", title="Tasa de negativas")
        fig.update_xaxes(title=None)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Nota 1–5 y errores
# ---------------------------------------------------------------------------


def _render_ordinal(chart_data: dict) -> None:
    section_intro(
        "La nota de 1 a 5 como problema ordinal",
        "Evaluación auxiliar: qué tan cerca queda la nota reconstruida por el texto. "
        "No es el modelo desplegado, pero muestra la riqueza de la señal.",
    )
    ordinal = chart_data.get("ordinal_5_estrellas", {})
    if not ordinal:
        st.info("No hay evaluación ordinal en los artefactos.")
        return

    cols = st.columns(4)
    cols[0].metric("QWK", f"{ordinal.get('quadratic_weighted_kappa', 0):.3f}")
    cols[1].metric("Nota exacta", f"{ordinal.get('exact_accuracy', 0) * 100:.1f}%")
    cols[2].metric("±1 estrella", f"{ordinal.get('off_by_one_accuracy', 0) * 100:.1f}%")
    cols[3].metric("PR-AUC negativa", f"{ordinal.get('negative_pr_auc', 0):.3f}")

    matrix = np.array(ordinal.get("confusion_matrix", []), dtype=int)
    classes = [str(value) for value in ordinal.get("classes", [1, 2, 3, 4, 5])]
    if matrix.size == 25:
        fig = px.imshow(
            matrix,
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale=[
                [0.0, PALETTE["white"]],
                [1.0, PALETTE["eden"]],
            ],
            labels={"x": "Nota predicha", "y": "Nota real", "color": "Reseñas"},
        )
        fig = style_figure(fig, title="Matriz de confusión de la nota (1–5)", height=420)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption(
        "QWK penaliza más los errores lejanos. Un QWK alto con accuracy exacta moderada indica "
        "que el modelo suele acercarse a la nota, aunque no siempre la acierte."
    )


def _render_errors(chart_data: dict) -> None:
    section_intro(
        "Análisis de errores",
        "Casos de test donde la predicción no coincide con la clase construida desde la nota. "
        "Útil para detectar ironía, ambigüedad y reseñas mal etiquetadas.",
    )
    errors = pd.DataFrame(chart_data.get("errores", []))
    if errors.empty:
        st.info("No hay errores registrados.")
        return
    selected = st.selectbox("Filtrar por sentimiento real", ["Todos"] + LABEL_ORDER)
    if selected != "Todos":
        errors = errors[errors["real"] == selected]
    display = errors.rename(
        columns={
            "texto": "Reseña",
            "real": "Real",
            "prediccion": "Predicción",
            "confianza": "Confianza",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Confianza": st.column_config.NumberColumn(format="%.3f"),
            "Reseña": st.column_config.TextColumn(width="large"),
        },
    )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def render(module: dict) -> None:
    render_module(
        module,
        "Sentimiento predicho, probabilidad, términos influyentes y cruces con logística.",
    )
    if not module.get("manifest_available"):
        return

    chart_data = _load_chart_data()
    results = _load_results()
    if not chart_data or results is None or results.empty:
        st.warning("No se encontraron los artefactos necesarios en artifacts/nlp/.")
        return

    metrics = module.get("metrics", {})
    columns = st.columns(5)
    columns[0].metric("Macro-F1", f"{metrics.get('macro_f1', 0):.3f}")
    columns[1].metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
    columns[2].metric("QWK 5 estrellas", f"{metrics.get('qwk_5_estrellas', 0):.3f}")
    columns[3].metric("PR-AUC negativa", f"{metrics.get('negative_pr_auc', 0):.3f}")
    columns[4].metric("Reseñas", f"{metrics.get('n_total', 0):,}")

    _render_metrics_guide()

    tabs = st.tabs(
        [
            "Resumen",
            "Analizador",
            "Términos",
            "Negocio",
            "Nota 1–5",
            "Errores",
        ]
    )
    with tabs[0]:
        _render_summary(chart_data)
    with tabs[1]:
        _render_analyzer()
    with tabs[2]:
        _render_terms(chart_data)
    with tabs[3]:
        _render_business(chart_data)
    with tabs[4]:
        _render_ordinal(chart_data)
    with tabs[5]:
        _render_errors(chart_data)

    st.markdown("---")
    col_conclusions, col_limitations = st.columns(2)
    with col_conclusions:
        section_intro("Conclusiones de negocio", "Qué puede hacer Olist con esta señal.")
        for index, item in enumerate(module.get("business_conclusions", []), start=1):
            st.markdown(f"**{index}.** {item}")
    with col_limitations:
        section_intro("Limitaciones", "Qué no debe inferirse de estos resultados.")
        for item in module.get("limitations", []):
            st.markdown(f"- {item}")

    with st.expander("Decisiones metodológicas y modelo ganador", expanded=False):
        st.markdown(
            f"""
            <div class='module-note'>
                <strong>Modelo desplegado:</strong> {module.get('winner', '—')}. Combina TF-IDF
                de palabras (1–2 n-gramas) y de caracteres (3–5 n-gramas) con una regresión
                logística ponderada por clase.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            La etiqueta se construye desde la nota: 1–2 estrellas = negativo, 3 = neutro,
            4–5 = positivo. No se eliminan stopwords porque las negaciones ("não", "nunca",
            "sem") invierten el sentido. Se deduplican textos exactos para evitar fuga entre
            train y test. La selección se hace con validación cruzada estratificada y macro-F1;
            el test se usa una sola vez para reportar el desempeño final.

            El transformer de referencia se evalúa zero-shot con un BERTimbau entrenado en otro
            e-commerce brasileño (B2W). No se fine-tuneó en Olist para no gastar horas de CPU ni
            contaminar la evaluación con modelos ya entrenados en este mismo dataset.
            """
        )
