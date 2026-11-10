"""
Vista Streamlit del modulo de Reglas de Asociacion y Venta Cruzada - DataSmart Olist.
===================================================================================
Responsable: Modulo de Asociacion (rama feature/association)
Proyecto Integrador - Mineria de Datos - UNMSM FISI 2026

Visualiza los resultados del modelo ganador a partir de artefactos precalculados:
  - Metricas clave y explicacion teorica detallada de cada una.
  - Simulador de venta cruzada interactivo en tiempo real (formulario de inferencia sin reentrenar).
  - Justificacion tecnica de la seleccion del modelo ganador (FP-Growth).
  - Tabla comparativa de algoritmos evaluados (Apriori, FP-Growth, ECLAT).
  - Graficos interactivos con paleta corporativa:
      * Dispersion Soporte vs Confianza (escalado por Lift).
      * Ranking de Top 10 Reglas por Lift.
      * Heatmap de co-ocurrencia entre categorias principales.
  - Explorador interactivo filtrable por categoria y umbral de Lift.
  - Conclusiones de negocio y limitaciones metodologicas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components import (
    PALETTE,
    section_intro,
    style_figure,
)
from frontend.views.module_template import render_module

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT / "artifacts" / "association"


# ---------------------------------------------------------------------------
# Carga de Artefactos mediante st.cache_data
# ---------------------------------------------------------------------------

@st.cache_data
def _load_manifest() -> dict | None:
    path = ARTIFACTS_DIR / "manifest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _load_chart_data() -> dict | None:
    path = ARTIFACTS_DIR / "chart_data.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _load_results() -> pd.DataFrame | None:
    path = ARTIFACTS_DIR / "results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_comparison() -> pd.DataFrame | None:
    path = ARTIFACTS_DIR / "algorithm_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Explicacion de Metricas de Asociacion
# ---------------------------------------------------------------------------

def _render_metrics_guide() -> None:
    with st.expander("Interpretación y fundamentación de las métricas de asociación", expanded=True):
        st.markdown(
            """
            <div class='module-note'>
                <strong>Guía para analistas de negocio:</strong> Interpretación práctica de los indicadores de reglas de asociación en comercio electrónico.
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                """
                El **Soporte (Support)** representa la popularidad o masa crítica de una combinación de artículos en la plataforma. Para el analista de negocio, responde a la pregunta de qué proporción del total de pedidos globales (96,478 entregas) incluyó ambos productos simultáneamente. En un catálogo donde el 99.2% de los pedidos son monoproducto, valores de soporte reducidos (como 0.01%) reflejan segmentos de nicho muy específicos pero con suficiente volumen para activar promociones dirigidas sin canibalizar otras categorías.

                La **Confianza (Confidence)** evalúa la fiabilidad predictiva de la recomendación al momento del pago. Expresa la probabilidad condicional de que el comprador incorpore el segundo artículo dado que ya tiene el primero en el carrito. Una confianza del 28.57% en artículos infantiles y accesorios indica que casi 1 de cada 3 clientes aceptará la sugerencia cruzada, ofreciendo una tasa de conversión óptima para pop-ups o módulos de recomendación durante el checkout.
                """
            )
        with col_m2:
            st.markdown(
                """
                El **Lift (Elevación de Afinidad)** es el indicador rector para priorizar estrategias comerciales, ya que cuantifica cuántas veces más probable es la compra conjunta frente a lo esperado por pura casualidad estadística. Un Lift de 1.0 refleja independencia total, mientras que valores superiores a 1.0 confirman complementariedad genuina. Por ejemplo, un Lift de 15.15x valida una afinidad sobresaliente entre categorías, asegurando un retorno de inversión muy superior al de cualquier descuento al azar.

                El **Apalancamiento (Leverage)** y la **Convicción (Conviction)** complementan la toma de decisiones al evaluar volumen y direccionalidad. El apalancamiento mide el número neto de ventas adicionales obtenidas por encima de compras fortuitas. Por su parte, la convicción mide la asimetría de la regla: valores elevados advierten que el producto sugerido casi nunca se adquiere de forma aislada sin el antecedente, lo que orienta directamente la planificación de stock y empaquetamiento conjunto.
                """
            )


# ---------------------------------------------------------------------------
# Simulador de Carrito y Venta Cruzada en Tiempo Real (Inferencia)
# ---------------------------------------------------------------------------

def _render_interactive_simulator(results: pd.DataFrame, chart_data: dict) -> None:
    section_intro(
        "Simulador de Venta Cruzada en Tiempo Real",
        "Selecciona un artículo en el carrito para consultar en tiempo real la recomendación del modelo sin necesidad de reentrenar.",
    )

    cooc_info = chart_data.get("cooccurrence_matrix", {})
    cooc_categories = cooc_info.get("categories", [])
    matrix = cooc_info.get("matrix", [])

    rule_antecedents = sorted(results["antecedents"].unique().tolist())
    all_options = rule_antecedents + [c for c in cooc_categories if c not in rule_antecedents]

    col_form1, col_form2 = st.columns([2, 1])

    with col_form1:
        selected_item = st.selectbox(
            "Artículo o categoría en el carrito del cliente:",
            options=all_options,
            index=0,
            help="Selecciona una categoría para evaluar la recomendación predictiva.",
        )

    with col_form2:
        st.markdown("<br>", unsafe_allow_html=True)
        is_direct_rule = selected_item in rule_antecedents
        tag_text = "Regla Directa (Lift > 1.05)" if is_direct_rule else "Afinidad de Mercado"
        st.markdown(f"<div class='module-note'><strong>Tipo de patrón:</strong> {tag_text}</div>", unsafe_allow_html=True)

    # 1. Caso A: La categoría tiene regla de asociación directa (Lift > 1.05)
    matched_rules = results[results["antecedents"] == selected_item]

    if not matched_rules.empty:
        best_rule = matched_rules.sort_values(["lift", "confidence"], ascending=[False, False]).iloc[0]
        rec_product = best_rule["consequents"]
        conf_val = float(best_rule["confidence"])
        lift_val = float(best_rule["lift"])
        supp_val = float(best_rule["support"])
        conv_val = float(best_rule["conviction"])

        st.markdown(
            f"""
            <div class='module-note'>
                <strong>Recomendación óptima en checkout:</strong> Para compras de <code>{selected_item}</code>, 
                recomendar <code>{rec_product}</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("Lift Multiplicador", f"{lift_val:.2f}x", help="Factor de afinidad frente a compras al azar")
        r_col2.metric("Confianza Estimada", f"{conf_val * 100:.2f}%", help="Probabilidad de adquirir el consecuente dado el antecedente")
        r_col3.metric("Soporte Global", f"{supp_val:.6f}", help="Frecuencia relativa en los 96,478 pedidos")
        r_col4.metric("Convicción", f"{conv_val:.3f}", help="Dependencia direccional de la regla")

        if lift_val >= 5.0:
            strategy_text = "Estrategia prioritaria de alto impacto: Mostrar banner o bundle directo en el carrito con 15% de descuento al añadir el segundo producto."
        elif lift_val >= 1.5:
            strategy_text = "Estrategia de afinidad directa: Sugerir el artículo complementario como pop-up antes de proceder a la pasarela de pagos."
        else:
            strategy_text = "Estrategia de incentivo: Ofrecer envío unificado gratis por compra combinada de ambas categorías."

        st.info(strategy_text)

    # 2. Caso B: La categoría no tiene regla directa con Lift > 1.05 (patrón monoproducto del 99.2%)
    else:
        top_partner = None
        top_count = 0

        if selected_item in cooc_categories and matrix:
            idx = cooc_categories.index(selected_item)
            row_counts = matrix[idx]
            off_diag = [(cooc_categories[j], row_counts[j]) for j in range(len(cooc_categories)) if j != idx]
            off_diag.sort(key=lambda x: x[1], reverse=True)
            if off_diag and off_diag[0][1] > 0:
                top_partner, top_count = off_diag[0]

        if top_partner:
            st.markdown(
                f"""
                <div class='module-note'>
                    <strong>Recomendación asistida por afinidad de mercado:</strong> Para compras de <code>{selected_item}</code>, 
                    la mayor co-ocurrencia histórica registrada es <code>{top_partner}</code> ({top_count} pedidos conjuntos).
                </div>
                """,
                unsafe_allow_html=True,
            )
            c_col1, c_col2, c_col3 = st.columns(3)
            c_col1.metric("Pedidos Conjuntos", f"{top_count} órdenes", help="Frecuencia absoluta en pedidos entregados")
            c_col2.metric("Comportamiento Base", "99.2% Compra Única", help="Hallazgo del EDA de concentración monoproducto")
            c_col3.metric("Potencial de Bundle", "Moderado", help="Oportunidad para incentivar primera compra cruzada")
            st.info("Estrategia sugerida: Ofrecer promoción de envío combinado o descuento progresivo para incentivar la exploración de categorías afines.")
        else:
            st.markdown(
                f"""
                <div class='module-note'>
                    <strong>Diagnóstico para {selected_item}:</strong> Artículo adquirido exclusivamente en transacciones unitarias.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info("Estrategia sugerida: Al pertenecer al 99.2% de pedidos de categoría única, se recomienda incentivar cupones para compras posteriores en lugar de empaquetamiento directo.")


# ---------------------------------------------------------------------------
# Componentes Graficos Plotly (Paleta Uniforme)
# ---------------------------------------------------------------------------

def _render_scatter_support_confidence(chart_data: dict) -> None:
    section_intro(
        "Dispersión de Reglas (Soporte vs Confianza)",
        "Distribución de reglas de asociación. El tamaño y color representan la intensidad del Lift.",
    )
    scatter_pts = chart_data.get("scatter_data", [])
    if not scatter_pts:
        st.info("No hay datos disponibles para el gráfico de dispersión.")
        return

    df = pd.DataFrame(scatter_pts)
    color_scale = [
        [0.0, PALETTE["sinbad"]],
        [0.5, PALETTE["bondi"]],
        [1.0, PALETTE["eden"]],
    ]

    fig = px.scatter(
        df,
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_name="rule_label",
        hover_data={
            "support": ":.6f",
            "confidence": ":.4f",
            "lift": ":.2f",
            "leverage": ":.6f",
            "conviction": ":.4f",
        },
        labels={
            "support": "Soporte",
            "confidence": "Confianza",
            "lift": "Lift",
        },
        color_continuous_scale=color_scale,
        size_max=18,
    )
    fig = style_figure(fig, title="Espacio Soporte vs Confianza", height=420)
    fig.update_layout(coloraxis_colorbar=dict(title="Lift", len=0.8))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_top_rules_bar(chart_data: dict) -> None:
    section_intro(
        "Top 10 Reglas por Lift",
        "Reglas con mayor factor de complementariedad frente a compras aleatorias independientes.",
    )
    top_rules = chart_data.get("top_10_rules", [])
    if not top_rules:
        st.info("No hay reglas disponibles para el ranking.")
        return

    df = pd.DataFrame(top_rules)
    df["rule_name"] = df["antecedents"] + " -> " + df["consequents"]
    df = df.sort_values("lift", ascending=True)

    fig = px.bar(
        df,
        x="lift",
        y="rule_name",
        orientation="h",
        color_discrete_sequence=[PALETTE["bondi"]],
        labels={"lift": "Lift", "rule_name": "Regla"},
        text=df["lift"].apply(lambda v: f"{v:.2f}"),
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Lift: %{x:.2f}<br>Confianza: %{customdata[0]:.2%}<extra></extra>",
        customdata=df[["confidence"]],
    )
    fig = style_figure(fig, title="Ranking de Reglas por Lift", height=420)
    fig.update_layout(yaxis_title=None, xaxis_title="Lift")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_cooccurrence_heatmap(chart_data: dict) -> None:
    section_intro(
        "Matriz de Co-ocurrencia entre Categorías Top",
        "Frecuencia conjunta de compra entre pares de categorías principales en pedidos entregados.",
    )
    cooc_info = chart_data.get("cooccurrence_matrix", {})
    categories = cooc_info.get("categories", [])
    matrix = cooc_info.get("matrix", [])

    if not categories or not matrix:
        st.info("No hay datos disponibles para la matriz de co-ocurrencia.")
        return

    arr = np.array(matrix, dtype=int, copy=True)
    np.fill_diagonal(arr, 0)
    matrix_viz = pd.DataFrame(arr, index=categories, columns=categories)

    color_scale = [
        [0.0, PALETTE["white"]],
        [0.15, PALETTE["sinbad"]],
        [0.55, PALETTE["bondi"]],
        [1.0, PALETTE["eden"]],
    ]

    fig = px.imshow(
        matrix_viz,
        x=categories,
        y=categories,
        color_continuous_scale=color_scale,
        labels={"color": "Co-ocurrencia"},
        aspect="auto",
    )
    fig.update_traces(
        hovertemplate="<b>%{y} + %{x}</b><br>Co-ocurrencias: %{z}<extra></extra>"
    )
    fig = style_figure(fig, title="Co-ocurrencia Cruzada de Categorías (Top 15)", height=450)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Punto de Entrada de la Vista
# ---------------------------------------------------------------------------

def render(module: dict) -> None:
    """Punto de entrada principal para la vista del módulo de Asociación."""
    # 1. Cabecera y verificación del contrato común
    render_module(module, "Reglas de asociación, support, confidence, lift y algoritmo ganador.")

    if not module.get("manifest_available"):
        return

    st.markdown("---")

    # 2. Cargar artefactos precalculados
    manifest = _load_manifest()
    chart_data = _load_chart_data()
    results = _load_results()
    comparison_df = _load_comparison()

    if not chart_data or results is None or results.empty:
        st.warning("No se encontraron los artefactos necesarios en artifacts/association/.")
        return

    # 3. Métricas clave del modelo ganador
    metrics = module.get("metrics", {})
    rules_count = metrics.get("rules_count", len(results))
    max_lift = metrics.get("max_lift")
    avg_lift = metrics.get("avg_lift")
    avg_conf = metrics.get("avg_confidence")
    exec_time = metrics.get("exec_time_sec")

    section_intro("Desempeño del Modelo Ganador", "Evaluación cuantitativa de las reglas extraídas mediante FP-Growth.")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Reglas Generadas", f"{rules_count}", help="Reglas válidas con Lift > 1.05")
    col2.metric("Lift Máximo", f"{max_lift:.2f}" if isinstance(max_lift, (int, float)) else "N/A", help="Factor multiplicador pico")
    col3.metric("Lift Promedio", f"{avg_lift:.2f}" if isinstance(avg_lift, (int, float)) else "N/A", help="Media de Lift en reglas válidas")
    col4.metric("Confianza Media", f"{avg_conf * 100:.2f}%" if isinstance(avg_conf, (int, float)) else "N/A", help="P(Consecuente | Antecedente)")
    col5.metric("Tiempo de Ejecución", f"{exec_time:.2f} s" if isinstance(exec_time, (int, float)) else "N/A", help="Tiempo de ajuste del modelo ganador")

    # Explicación detallada de cada métrica
    _render_metrics_guide()

    st.markdown("---")

    # 4. Simulador interactivo de Venta Cruzada (Inferencia en Tiempo Real)
    _render_interactive_simulator(results, chart_data)

    st.markdown("---")

    # 5. Justificación técnica integrada al diseño
    with st.expander("Criterios de selección del algoritmo ganador", expanded=False):
        st.markdown(
            """
            <div class='module-note'>
                <strong>Modelo seleccionado:</strong> FP-Growth sobre el espacio de Canasta por Pedido (order_id).
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_just1, col_just2 = st.columns(2)
        with col_just1:
            st.markdown(
                """
                La estructura en árbol enlazado (FP-Tree) de FP-Growth procesa la matriz transaccional con un consumo pico de 0.14 MB de memoria RAM, frente a los 557.25 MB requeridos por Apriori para generar y podar candidatos. Esta diferencia de tres órdenes de magnitud evita la sobrecarga combinatoria en entornos de alta dispersión.
                
                Frente a ECLAT, que opera mediante intersección de listas verticales (TID-lists), FP-Growth mantiene un tiempo de cómputo predecible y deriva soporte, confianza y lift de forma directa sin reconstruir índices intermedios.
                """
            )
        with col_just2:
            st.markdown(
                """
                La elección de la canasta por pedido responde a la naturaleza del comercio electrónico: el objetivo de negocio es la venta cruzada simultánea en el momento del pago. El 99.2% de los pedidos en Olist concentra una sola categoría, por lo que las reglas obtenidas reflejan patrones reales de compra conjunta en una misma sesión.
                
                Agrupar transacciones a nivel de cliente histórico acumula compras realizadas a lo largo de meses. Dado que la tasa de recompra histórica es de apenas 3.0%, la asociación diferida pierde aplicabilidad práctica para el diseño de paquetes y recomendaciones en tiempo real.
                """
            )

    st.markdown("---")

    # 6. Tabla Comparativa de Algoritmos Evaluados
    if comparison_df is not None and not comparison_df.empty:
        section_intro("Evaluación Comparativa de Algoritmos", "Benchmark de algoritmos evaluados en ambos espacios de canasta.")
        display_comp = comparison_df.copy()
        display_comp.columns = [
            "Algoritmo", "Espacio de Canasta", "Soporte Mínimo",
            "Nº Itemsets", "Nº Reglas (Lift > 1.05)", "Tiempo (s)", "Memoria (MB)",
        ]
        st.dataframe(display_comp, hide_index=True, use_container_width=True)
        st.markdown("---")

    # 7. Visualizaciones Interactivas Plotly
    tab_scatter, tab_bar, tab_heatmap = st.tabs([
        "Dispersión Soporte vs Confianza",
        "Top 10 Reglas por Lift",
        "Matriz de Co-ocurrencia",
    ])

    with tab_scatter:
        _render_scatter_support_confidence(chart_data)

    with tab_bar:
        _render_top_rules_bar(chart_data)

    with tab_heatmap:
        _render_cooccurrence_heatmap(chart_data)

    st.markdown("---")

    # 8. Explorador Interactivo de Reglas
    section_intro("Explorador Interactivo de Reglas", "Filtrado por categoría antecedente y umbral de Lift mínimo.")

    all_antecedents = ["Todas"] + sorted(results["antecedents"].unique().tolist())
    col_filter1, col_filter2 = st.columns([2, 2])

    selected_ant = col_filter1.selectbox("Filtrar por Categoría Antecedente:", all_antecedents)
    min_lift_filter = col_filter2.slider("Lift Mínimo:", min_value=1.0, max_value=float(results["lift"].max()), value=1.05, step=0.05)

    filtered_rules = results.copy()
    if selected_ant != "Todas":
        filtered_rules = filtered_rules[filtered_rules["antecedents"] == selected_ant]
    filtered_rules = filtered_rules[filtered_rules["lift"] >= min_lift_filter]

    st.dataframe(
        filtered_rules,
        hide_index=True,
        use_container_width=True,
        column_config={
            "antecedents": st.column_config.TextColumn("Antecedente (Si compra...)", width="medium"),
            "consequents": st.column_config.TextColumn("Consecuente (...también compra)", width="medium"),
            "support": st.column_config.NumberColumn("Soporte", format="%.6f"),
            "confidence": st.column_config.NumberColumn("Confianza", format="%.4f"),
            "lift": st.column_config.NumberColumn("Lift", format="%.3f"),
            "leverage": st.column_config.NumberColumn("Leverage", format="%.6f"),
            "conviction": st.column_config.NumberColumn("Convicción", format="%.4f"),
        },
    )

    st.markdown("---")

    # 9. Conclusiones de Negocio y Limitaciones (Breves y Estructuradas)
    conclusions = module.get("business_conclusions", [])
    limitations = module.get("limitations", [])

    col_conc, col_lim = st.columns(2)

    with col_conc:
        section_intro("Conclusiones de Negocio", "Acciones comerciales para venta cruzada y empaquetamiento.")
        for idx, item in enumerate(conclusions, start=1):
            st.markdown(f"**{idx}.** {item}")

    with col_lim:
        section_intro("Limitaciones Metodológicas", "Restricciones analíticas identificadas en el dataset.")
        for idx, item in enumerate(limitations, start=1):
            st.markdown(f"- {item}")
