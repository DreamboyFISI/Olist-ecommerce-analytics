"""
Vista Streamlit del modulo de Reglas de Asociacion y Venta Cruzada - DataSmart Olist.
===================================================================================
Responsable: Modulo de Asociacion (rama feature/association)
Proyecto Integrador - Mineria de Datos - UNMSM FISI 2026

Visualiza los resultados del modelo ganador a partir de artefactos precalculados:
  - Metricas clave de desempeño de las reglas.
  - Justificacion tecnica resumida de la seleccion de FP-Growth.
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

    # Construir arreglo explicito y modificable para evitar errores de arreglos read-only
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

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Justificación técnica integrada al diseño
    with st.expander("Criterios de selección del algoritmo ganador", expanded=True):
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

    # 5. Tabla Comparativa de Algoritmos Evaluados
    if comparison_df is not None and not comparison_df.empty:
        section_intro("Evaluación Comparativa de Algoritmos", "Benchmark de algoritmos evaluados en ambos espacios de canasta.")
        display_comp = comparison_df.copy()
        display_comp.columns = [
            "Algoritmo", "Espacio de Canasta", "Soporte Mínimo",
            "Nº Itemsets", "Nº Reglas (Lift > 1.05)", "Tiempo (s)", "Memoria (MB)",
        ]
        st.dataframe(display_comp, hide_index=True, use_container_width=True)
        st.markdown("---")

    # 6. Visualizaciones Interactivas Plotly
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

    # 7. Explorador Interactivo de Reglas
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

    # 8. Conclusiones de Negocio y Limitaciones (Breves y Estructuradas)
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
