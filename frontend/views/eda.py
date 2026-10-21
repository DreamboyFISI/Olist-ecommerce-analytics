from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components import PALETTE, page_header, section_intro, style_figure

ROOT = Path(__file__).resolve().parents[2]


def _review_chart(eda: dict):
    source = eda["review_score_%"]
    frame = pd.DataFrame({"Estrellas": [int(k) for k in source], "Porcentaje": list(source.values())})
    fig = px.bar(frame, x="Estrellas", y="Porcentaje", text_auto=".1f", color="Estrellas", color_discrete_sequence=[PALETTE["eden"], PALETTE["bondi"], PALETTE["sinbad"], PALETTE["san_marino"], "#2E7D64"])
    fig.update_traces(hovertemplate="%{x} estrellas: %{y:.2f}%<extra></extra>")
    fig.update_layout(showlegend=False)
    return style_figure(fig, "Distribución de calificaciones")


def _category_chart(eda: dict, top_n: int, metric: str):
    frame = pd.DataFrame.from_dict(eda["top10_categorias"], orient="index").reset_index(names="Categoría").head(top_n)
    y_value = "ventas" if metric == "Ventas" else "%_review_negativa"
    labels = {"ventas": "Ventas (BRL)", "%_review_negativa": "Reseñas negativas (%)"}
    fig = px.bar(frame.sort_values(y_value), x=y_value, y="Categoría", orientation="h", color=y_value, color_continuous_scale=[PALETTE["sinbad"], PALETTE["bondi"], PALETTE["eden"]], labels={y_value: labels[y_value]})
    fig.update_layout(coloraxis_showscale=False)
    return style_figure(fig, f"Top {top_n} categorías por {labels[y_value].lower()}", height=460)


def _quality_chart(eda: dict):
    missing = pd.Series(eda["nulos"], name="Porcentaje").sort_values().tail(10).rename_axis("Variable").reset_index()
    fig = px.bar(missing, x="Porcentaje", y="Variable", orientation="h", color_discrete_sequence=[PALETTE["bondi"]])
    fig.update_traces(hovertemplate="%{y}: %{x:.2f}%<extra></extra>")
    return style_figure(fig, "Variables con mayor porcentaje de valores faltantes", height=440)


def _correlation_chart(eda: dict):
    corr = pd.Series(eda["corr_review_score"], name="Correlación").sort_values().rename_axis("Variable").reset_index()
    corr["Dirección"] = corr["Correlación"].apply(lambda value: "Positiva" if value >= 0 else "Negativa")
    fig = px.bar(corr, x="Correlación", y="Variable", orientation="h", color="Dirección", color_discrete_map={"Positiva": PALETTE["san_marino"], "Negativa": PALETTE["bondi"]})
    fig.update_traces(hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    return style_figure(fig, "Correlación de Spearman con review_score", height=470)


def render(eda: dict, source_label: str) -> None:
    page_header("EDA completado", "Qué cuentan los pedidos de Olist", "Explora los resultados consolidados del análisis inicial. Los controles permiten cambiar la lectura sin recalcular el notebook.")
    st.caption(f"Periodo: {eda['periodo'][0]} a {eda['periodo'][1]} · Fuente: {source_label}")
    section_intro("Indicadores clave", "Una vista rápida de satisfacción, logística y comportamiento comercial.")
    row = st.columns(4)
    row[0].metric("Pedidos", f"{eda['tablas']['orders']['filas']:,}")
    row[1].metric("Clientes únicos", f"{eda['clientes_unicos']:,}")
    row[2].metric("Entrega mediana", f"{eda['dias_entrega_mediana']:.2f} días")
    row[3].metric("Recompra", f"{eda['%_clientes_recompra']:.2f}%")
    tab_satisfaction, tab_commerce, tab_quality = st.tabs(["Satisfacción y logística", "Comercio", "Calidad de datos"])
    with tab_satisfaction:
        left, right = st.columns([1.3, 1])
        with left:
            st.plotly_chart(_review_chart(eda), use_container_width=True, config={"displaylogo": False})
        with right:
            st.subheader("El retraso cambia la experiencia")
            st.metric("Negativas · entrega a tiempo", f"{eda['%_negativas_a_tiempo']:.2f}%")
            st.metric("Negativas · entrega tardía", f"{eda['%_negativas_tardias']:.2f}%")
            st.metric("MAE de la promesa de Olist", f"{eda['MAE_estimacion_olist_dias']:.2f} días")
            st.info("La puntualidad es la señal más accionable del EDA y orienta regresión y clasificación.")
        st.plotly_chart(_correlation_chart(eda), use_container_width=True, config={"displaylogo": False})
    with tab_commerce:
        controls = st.columns([1, 1, 2])
        top_n = controls[0].slider("Número de categorías", 5, 10, 10, help="Ajusta cuántas categorías aparecen en la gráfica.")
        metric = controls[1].selectbox("Ordenar por", ["Ventas", "Reseñas negativas"])
        st.plotly_chart(_category_chart(eda, top_n, metric), use_container_width=True, config={"displaylogo": False})
        payment = pd.DataFrame({"Medio": list(eda["payment_type_%"]), "Porcentaje": list(eda["payment_type_%"].values())})
        pay_fig = px.pie(payment, names="Medio", values="Porcentaje", hole=.58, color_discrete_sequence=[PALETTE["bondi"], PALETTE["san_marino"], PALETTE["eden"], PALETTE["sinbad"], PALETTE["janna"]])
        pay_fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value:.2f}%<extra></extra>")
        st.plotly_chart(style_figure(pay_fig, "Distribución de medios de pago"), use_container_width=True, config={"displaylogo": False})
        stats = st.columns(3)
        stats[0].metric("Ticket promedio", f"R$ {eda['ticket_promedio_BRL']:,.2f}")
        stats[1].metric("Categorías para 80% de ventas", f"{eda['categorias_80%_ventas']} de {eda['n_categorias']}")
        stats[2].metric("Pedidos multicategoría", f"{eda['%_pedidos_multi_categoria']:.2f}%")
    with tab_quality:
        st.plotly_chart(_quality_chart(eda), use_container_width=True, config={"displaylogo": False})
        left, right = st.columns(2)
        with left:
            st.subheader("Inconsistencias detectadas")
            st.dataframe(pd.DataFrame({"Control": list(eda["inconsistencias"]), "Registros": list(eda["inconsistencias"].values())}), hide_index=True, use_container_width=True)
        with right:
            st.subheader("Outliers por IQR")
            st.dataframe(pd.DataFrame.from_dict(eda["outliers"], orient="index").reset_index(names="Variable"), hide_index=True, use_container_width=True)
    with st.expander("Ver figuras originales del notebook"):
        selected = st.selectbox("Figura", ["e02_ventas_mensuales.png", "e05_entrega_vs_review.png", "e07_entrega_por_estado.png", "e09_correlaciones.png"])
        st.image(str(ROOT / "docs" / "figures" / selected), caption="Salida original del EDA", use_container_width=True)
