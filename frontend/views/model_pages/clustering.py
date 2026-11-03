"""
Vista Streamlit del módulo de Clustering y Segmentación de Clientes — DataSmart Olist.
===================================================================================
Responsable: Módulo de Clustering (rama feature/clustering)
Proyecto Integrador · Minería de Datos · UNMSM FISI 2026

Visualiza los resultados del modelo ganador a partir exclusivamente de los artefactos
precalculados en artifacts/clustering/ mediante st.cache_data:
  - Métricas de validación y estabilidad.
  - Justificación técnica y de negocio del modelo ganador.
  - Tabla comparativa de algoritmos (K-Means, DBSCAN, Ward Jerárquico en Conjuntos A y B).
  - Distribución de clientes y contribución acumulada a ingresos por segmento.
  - Gráficos interactivos Plotly:
      * Proyección PCA bidimensional (muestra <= 3,000 puntos).
      * Dispersión 3D RFM interactiva.
      * Perfil multidimensional relativo (Radar chart).
      * Boxplots de distribución de Recencia, Frecuencia y Monetario por cluster.
  - Explicabilidad analítica mediante árbol de decisión superficial (reglas y exactitud).
  - Muestra estratificada filtrable de clientes (R, F, M, CLV, ABC, Silhouette).
  - Conclusiones y limitaciones metodológicas.
"""

from __future__ import annotations

import json
from pathlib import Path

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
ARTIFACTS_DIR = ROOT / "artifacts" / "clustering"

# Paleta accesible corporativa
CLUSTER_COLORS = [
    PALETTE["bondi"],       # #0799B6 - Recientes / Nuevos
    "#E07B54",              # Terracota - Alto Valor en Riesgo
    PALETTE["eden"],        # #114C5F - Recompradores Leales (VIP)
    PALETTE["sinbad"],      # #9CD2D3 - Ocasionales de Bajo Valor
    PALETTE["san_marino"],  # #4A6EB0
    PALETTE["janna"],       # #F2E6CF
]


# ═══════════════════════════════════════════════════════════════════════════
# Carga de Artefactos mediante st.cache_data
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Componentes Gráficos Plotly
# ═══════════════════════════════════════════════════════════════════════════

def _render_distribution_bars(chart_data: dict) -> None:
    counts = chart_data.get("cluster_counts", [])
    if not counts:
        return

    df = pd.DataFrame(counts)

    col1, col2 = st.columns(2)

    with col1:
        fig_clients = px.bar(
            df,
            x="cluster_name",
            y="pct_clients",
            color="cluster_name",
            color_discrete_sequence=CLUSTER_COLORS[: len(df)],
            labels={"cluster_name": "Segmento", "pct_clients": "% de Clientes"},
            text=df["pct_clients"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_clients = style_figure(fig_clients, title="Distribución de la Base de Clientes (%)", height=350)
        fig_clients.update_traces(textposition="outside")
        fig_clients.update_layout(showlegend=False, xaxis_title=None, yaxis_title="% Clientes")
        st.plotly_chart(fig_clients, use_container_width=True, config={"displaylogo": False})

    with col2:
        fig_rev = px.bar(
            df,
            x="cluster_name",
            y="pct_revenue",
            color="cluster_name",
            color_discrete_sequence=CLUSTER_COLORS[: len(df)],
            labels={"cluster_name": "Segmento", "pct_revenue": "% de Ingresos"},
            text=df["pct_revenue"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_rev = style_figure(fig_rev, title="Contribución a Ingresos Totales (%)", height=350)
        fig_rev.update_traces(textposition="outside")
        fig_rev.update_layout(showlegend=False, xaxis_title=None, yaxis_title="% Ingresos")
        st.plotly_chart(fig_rev, use_container_width=True, config={"displaylogo": False})


def _render_pca_scatter(chart_data: dict) -> None:
    section_intro(
        "Proyección Bidimensional PCA (Muestra Estratificada)",
        "Reducción dimensional mediante componentes principales sobre el espacio RFM transformado logarítmicamente.",
    )
    pca_pts = chart_data.get("pca_sample", [])
    if not pca_pts:
        st.info("No hay datos disponibles para la proyección PCA.")
        return

    df_pca = pd.DataFrame(pca_pts)
    var_ratio = chart_data.get("pca_variance_ratio", [0.0, 0.0])
    v1_pct = var_ratio[0] * 100 if len(var_ratio) > 0 else 0
    v2_pct = var_ratio[1] * 100 if len(var_ratio) > 1 else 0

    fig = px.scatter(
        df_pca,
        x="pca_1",
        y="pca_2",
        color="cluster_name",
        color_discrete_sequence=CLUSTER_COLORS[: df_pca["cluster_name"].nunique()],
        hover_data=["Recency", "Frequency", "Monetary"],
        labels={
            "pca_1": f"Componente Principal 1 ({v1_pct:.1f}% var)",
            "pca_2": f"Componente Principal 2 ({v2_pct:.1f}% var)",
            "cluster_name": "Segmento",
            "Recency": "Recencia (días)",
            "Frequency": "Frecuencia",
            "Monetary": "Gasto Total (R$)",
        },
        opacity=0.75,
    )
    fig = style_figure(fig, height=480)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_scatter_3d(results: pd.DataFrame) -> None:
    section_intro(
        "Espacio Tridimensional RFM",
        "Dispersión sobre Recencia (días), Frecuencia (nº pedidos) y Monetario (R$).",
    )
    fig = px.scatter_3d(
        results,
        x="R",
        y="M",
        z="F",
        color="cluster_name",
        color_discrete_sequence=CLUSTER_COLORS[: results["cluster_name"].nunique()],
        hover_data=["customer_unique_id", "R", "F", "M", "cluster_name"],
        labels={
            "R": "Recencia (días)",
            "M": "Monetario (R$)",
            "F": "Frecuencia",
            "cluster_name": "Segmento",
        },
        opacity=0.8,
    )
    fig = style_figure(fig, height=520)
    fig.update_layout(
        scene=dict(
            xaxis_title="Recencia (días)",
            yaxis_title="Monetario (R$)",
            zaxis_title="Frecuencia",
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_radar(chart_data: dict) -> None:
    section_intro(
        "Perfil Multidimensional Relativo (Radar)",
        "Variables estandarizadas de 0 a 1 entre centroides. La recencia invertida premia la inmediatez de compra.",
    )
    centroids = chart_data.get("centroids_normalized", [])
    if not centroids:
        return

    categories = [
        "Recencia\n(invertida)",
        "Frecuencia\n(recompra)",
        "Monetario\n(gasto total)",
        "Ticket\npromedio",
    ]
    fig = go.Figure()

    for idx, c in enumerate(centroids):
        values = [
            c.get("Recency_norm", 0),
            c.get("Frequency_norm", 0),
            c.get("Monetary_norm", 0),
            c.get("AvgTicket_norm", 0),
        ]
        values.append(values[0])
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=c["cluster_name"],
                line=dict(color=color, width=2),
                opacity=0.70,
            )
        )

    fig = style_figure(fig, height=440)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_boxplots(results: pd.DataFrame) -> None:
    section_intro(
        "Distribución y Dispersión por Segmento",
        "Boxplots de Recencia y Gasto Monetario observados en la muestra de resultados.",
    )
    col1, col2 = st.columns(2)

    with col1:
        fig_r = px.box(
            results,
            x="cluster_name",
            y="R",
            color="cluster_name",
            color_discrete_sequence=CLUSTER_COLORS[: results["cluster_name"].nunique()],
            labels={"cluster_name": "Segmento", "R": "Recencia (días)"},
        )
        fig_r = style_figure(fig_r, title="Distribución de Recencia (días)", height=380)
        fig_r.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig_r, use_container_width=True, config={"displaylogo": False})

    with col2:
        fig_m = px.box(
            results,
            x="cluster_name",
            y="M",
            color="cluster_name",
            color_discrete_sequence=CLUSTER_COLORS[: results["cluster_name"].nunique()],
            labels={"cluster_name": "Segmento", "M": "Gasto Monetario (R$)"},
            log_y=True,
        )
        fig_m = style_figure(fig_m, title="Distribución de Gasto Monetario (escala log R$)", height=380)
        fig_m.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig_m, use_container_width=True, config={"displaylogo": False})


def _render_quantitative_profile(chart_data: dict) -> None:
    section_intro("Perfil Cuantitativo y Plan de Acción", "Promedios observados por cluster y estrategia sugerida.")
    centroids = chart_data.get("centroids_normalized", [])
    counts_map = {c["cluster"]: c for c in chart_data.get("cluster_counts", [])}

    if not centroids:
        return

    profile_rows = []
    for c in centroids:
        cid = c["cluster"]
        meta = counts_map.get(cid, {})
        profile_rows.append({
            "Segmento": c["cluster_name"],
            "% Clientes": f"{meta.get('pct_clients', 0):.1f}%",
            "% Ingresos": f"{meta.get('pct_revenue', 0):.1f}%",
            "Recencia media": f"{c.get('raw_Recency', 0):.1f} d",
            "Frecuencia media": f"{c.get('raw_Frequency', 0):.2f}",
            "Gasto medio": f"R$ {c.get('raw_Monetary', 0):.2f}",
            "Ticket prom.": f"R$ {c.get('raw_AvgTicket', 0):.2f}",
            "Satisfacción": f"{c.get('raw_review_score', 0):.2f} ★",
            "% Entregas tardías": f"{c.get('raw_is_late_pct', 0):.1f}%",
            "Acción de Negocio Prioritaria": meta.get("business_action", ""),
        })

    profile_df = pd.DataFrame(profile_rows)
    st.dataframe(profile_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Punto de Entrada de la Vista
# ═══════════════════════════════════════════════════════════════════════════

def render(module: dict) -> None:
    """Punto de entrada principal para la vista del módulo de Clustering."""
    # 1. Cabecera y verificación del contrato común
    render_module(module, "Cliente, cluster, métricas RFM, perfil del segmento y Silhouette Score.")

    if not module.get("manifest_available"):
        return

    st.markdown("---")

    # 2. Cargar artefactos precalculados
    chart_data = _load_chart_data()
    results = _load_results()
    comparison_df = _load_comparison()

    if not chart_data or results is None or results.empty:
        st.warning("No se encontraron los artefactos necesarios en artifacts/clustering/.")
        return

    # 3. Métricas clave del particionamiento ganador
    metrics = module.get("metrics", {})
    sil = metrics.get("silhouette_score")
    db = metrics.get("davies_bouldin_index")
    ch = metrics.get("calinski_harabasz_index")
    ari_stab = metrics.get("stability_ari_10_runs")
    ari_base = metrics.get("baseline_rfm_quantiles_ari")

    section_intro("Desempeño y Validación Técnica", "Evaluación sobre el 20% de prueba y análisis de estabilidad.")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Silhouette (Test)", f"{sil:.4f}" if isinstance(sil, (int, float)) else "N/A", help="Objetivo: >= 0.40")
    col2.metric("Davies-Bouldin", f"{db:.4f}" if isinstance(db, (int, float)) else "N/A", help="Menor es mejor")
    col3.metric("Calinski-Harabasz", f"{ch:,.1f}" if isinstance(ch, (int, float)) else "N/A", help="Mayor es mejor")
    col4.metric("Estabilidad ARI (10 seeds)", f"{ari_stab:.4f}" if isinstance(ari_stab, (int, float)) else "N/A", help="Consistencia entre corridas")
    col5.metric("ARI vs Cuantiles RFM", f"{ari_base:.4f}" if isinstance(ari_base, (int, float)) else "N/A", help="Diferenciación frente a regla ingenua")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Justificación del Modelo Ganador
    with st.expander("📌 Justificación de Selección del Modelo Ganador", expanded=True):
        st.markdown(
            """
            **Modelo Seleccionado:** `K-Means (k=4) sobre Conjunto A (RFM transformado con log1p)`
            
            1. **Maldición de la Dimensionalidad en Conjunto B:** Al incorporar 9 variables (reseñas, flete, cuotas, demoras), el Silhouette Score se desplomó de **0.4005 a 0.2151** y Davies-Bouldin empeoró de **0.75 a 1.35**, confirmando que el ruido transaccional dispersa la densidad euclidiana.
            2. **Falla de DBSCAN ante Frecuencia casi constante:** Con 97% de compradores únicos, la densidad del espacio no presenta valles claros de separación. DBSCAN clasificó a más del 98% de la base en un solo macro-cluster homogéneo más ruido disperso, resultando inoperable comercialmente.
            3. **Estabilidad y Coherencia de K-Means (k=4):** Logró un **ARI de 0.9947** en 10 corridas independientes, aislando de forma nítida al núcleo de clientes leales (Frecuencia > 1, 3.0%) y particionando a los compradores únicos entre recientes, inactivos de alto valor y ocasionales de bajo ticket.
            4. **Superación de Cuantiles Ingenuos:** El ARI de 0.1617 frente a la segmentación por quintiles confirma que el agrupamiento no supervisado descubre fronteras de comportamiento reales en lugar de forzar cortes uniformes artificiales.
            """
        )

    st.markdown("---")

    # 5. Tabla Comparativa de Algoritmos
    if comparison_df is not None and not comparison_df.empty:
        section_intro("Evaluación Comparativa de Modelos", "Comparación de algoritmos evaluados bajo idéntica partición 80/20 train/test.")
        display_comp = comparison_df[[
            "algorithm", "feature_set", "n_clusters", "silhouette_train",
            "silhouette_test", "davies_bouldin", "calinski_harabasz", "noise_pct", "train_time_sec"
        ]].copy()
        display_comp.columns = [
            "Algoritmo", "Conjunto de Variables", "Nº Clusters", "Sil. Train",
            "Sil. Test", "Davies-Bouldin", "Calinski-Harabasz", "% Ruido", "Tiempo (s)"
        ]
        st.dataframe(display_comp, hide_index=True, use_container_width=True)
        st.markdown("---")

    # 6. Distribución de clientes y contribución
    section_intro("Distribución y Concentración de Valor", "Participación en clientes frente al volumen monetario aportado.")
    _render_distribution_bars(chart_data)

    st.markdown("---")

    # 7. Visualizaciones interactivas
    tab_pca, tab_3d, tab_radar, tab_box = st.tabs([
        "Proyección PCA (2D)",
        "Espacio RFM (3D)",
        "Perfil Radar Relativo",
        "Dispersión y Boxplots",
    ])

    with tab_pca:
        _render_pca_scatter(chart_data)

    with tab_3d:
        _render_scatter_3d(results)

    with tab_radar:
        _render_radar(chart_data)

    with tab_box:
        _render_boxplots(results)

    st.markdown("---")

    # 8. Perfil cuantitativo y acciones
    _render_quantitative_profile(chart_data)

    st.markdown("---")

    # 9. Explicabilidad mediante Árbol de Decisión Sustituto
    tree_acc = chart_data.get("tree_accuracy", 0.0)
    tree_rules = chart_data.get("tree_rules", "")
    section_intro(
        "Explicabilidad del Modelo (Árbol de Decisión Sustituto)",
        f"Aproximación interpretable con un árbol de decisión (max_depth=3) que replica los clusters con exactitud de {tree_acc * 100:.2f}%.",
    )
    with st.expander("Ver Reglas de Decisión Extraídas (R, F, M)", expanded=False):
        st.code(tree_rules, language="text")

    st.markdown("---")

    # 10. Muestra estratificada de resultados
    section_intro(
        "Muestra Estratificada de Clientes",
        "Muestra de 500 clientes con asignación de cluster, RFM, CLV, clasificación ABC y Silhouette individual.",
    )

    filter_col1, filter_col2 = st.columns([2, 2])
    available_clusters = ["Todos"] + sorted(results["cluster_name"].unique().tolist())
    selected_cluster = filter_col1.selectbox("Filtrar por segmento:", available_clusters)

    available_abc = ["Todos"] + sorted(results["ABC"].dropna().unique().tolist())
    selected_abc = filter_col2.selectbox("Filtrar por categoría ABC:", available_abc)

    filtered_df = results.copy()
    if selected_cluster != "Todos":
        filtered_df = filtered_df[filtered_df["cluster_name"] == selected_cluster]
    if selected_abc != "Todos":
        filtered_df = filtered_df[filtered_df["ABC"] == selected_abc]

    st.dataframe(
        filtered_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "customer_unique_id": st.column_config.TextColumn("ID Cliente Único", width="large"),
            "R": st.column_config.NumberColumn("Recencia (días)", format="%d"),
            "F": st.column_config.NumberColumn("Frecuencia", format="%d"),
            "M": st.column_config.NumberColumn("Monetario (R$)", format="%.2f"),
            "AvgTicket": st.column_config.NumberColumn("Ticket Prom. (R$)", format="%.2f"),
            "CLV_historical": st.column_config.NumberColumn("CLV Histórico (R$)", format="%.2f"),
            "CLV_projected": st.column_config.NumberColumn("CLV Proyectado (R$)", format="%.2f"),
            "ABC": st.column_config.TextColumn("Curva ABC"),
            "cluster_name": st.column_config.TextColumn("Segmento"),
            "silhouette": st.column_config.NumberColumn("Silhouette Indiv.", format="%.4f"),
        },
    )

    st.markdown("---")

    # 11. Conclusiones de Negocio y Limitaciones
    conclusions = module.get("business_conclusions", [])
    limitations = module.get("limitations", [])

    col_conc, col_lim = st.columns(2)

    with col_conc:
        section_intro("Conclusiones de Negocio", "Hallazgos accionables derivados de la segmentación.")
        for idx, item in enumerate(conclusions, start=1):
            st.markdown(f"**{idx}.** {item}")

    with col_lim:
        section_intro("Limitaciones Metodológicas", "Restricciones observadas en los datos y modelos.")
        for idx, item in enumerate(limitations, start=1):
            st.markdown(f"- {item}")
