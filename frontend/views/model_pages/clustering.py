"""
Vista Streamlit del módulo de Clustering RFM — DataSmart Olist.
===============================================================

Consume el contrato base mediante ``render_module`` y despliega el análisis
analítico a partir de ``chart_data.json`` y ``results.csv``:
  - Distribución de clientes por segmento.
  - Espacio tridimensional RFM y dispersión bidimensional.
  - Perfil multidimensional relativo (Radar).
  - Perfil cuantitativo de centroides.
  - Tabla de resultados estratificada y filtrable.
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

CLUSTER_COLORS = [
    PALETTE["bondi"],       # #0799B6
    PALETTE["san_marino"],  # #4A6EB0
    PALETTE["eden"],        # #114C5F
    PALETTE["sinbad"],      # #9CD2D3
    PALETTE["janna"],       # #F2E6CF
    "#E07B54",              # Terracota
]


def _load_chart_data() -> dict | None:
    path = ARTIFACTS_DIR / "chart_data.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_results() -> pd.DataFrame | None:
    path = ARTIFACTS_DIR / "results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "AvgTicket" not in df.columns and "M" in df.columns and "F" in df.columns:
        df["AvgTicket"] = (df["M"] / df["F"]).round(2)
    return df


def _render_bar_counts(chart_data: dict) -> None:
    counts = chart_data.get("cluster_counts", [])
    if not counts:
        return

    df = pd.DataFrame(counts)
    total = df["count"].sum()
    df["pct"] = (df["count"] / total * 100).round(1)
    df["label"] = df["count"].apply(lambda x: f"{x:,}") + " (" + df["pct"].astype(str) + "%)"

    fig = px.bar(
        df,
        x="cluster_name",
        y="count",
        color="cluster_name",
        color_discrete_sequence=CLUSTER_COLORS[: len(df)],
        labels={"cluster_name": "Segmento", "count": "Clientes"},
        text="label",
    )
    fig = style_figure(fig, title="Distribución de clientes por segmento", height=360)
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_scatter_3d(results: pd.DataFrame) -> None:
    section_intro(
        "Espacio tridimensional RFM",
        "Dispersión según Recencia, Gasto Monetario y Ticket Promedio.",
    )

    z_col = "AvgTicket" if "AvgTicket" in results.columns else "F"
    z_label = "Ticket promedio (R$)" if z_col == "AvgTicket" else "Frecuencia"

    fig = px.scatter_3d(
        results,
        x="R",
        y="M",
        z=z_col,
        color="cluster_name",
        color_discrete_sequence=CLUSTER_COLORS[: results["cluster_name"].nunique()],
        hover_data=["customer_unique_id", "R", "F", "M", "cluster_name"],
        labels={
            "R": "Recencia (días)",
            "M": "Monetario (R$)",
            z_col: z_label,
            "cluster_name": "Segmento",
        },
        opacity=0.8,
    )
    fig = style_figure(fig, height=540)
    fig.update_layout(
        scene=dict(
            xaxis_title="Recencia (días)",
            yaxis_title="Monetario (R$)",
            zaxis_title=z_label,
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_scatter_2d(results: pd.DataFrame) -> None:
    fig = px.scatter(
        results,
        x="R",
        y="M",
        color="cluster_name",
        color_discrete_sequence=CLUSTER_COLORS[: results["cluster_name"].nunique()],
        hover_data=["customer_unique_id", "F", "M"],
        labels={
            "R": "Recencia (días)",
            "M": "Monetario (R$)",
            "cluster_name": "Segmento",
        },
        opacity=0.75,
    )
    fig = style_figure(fig, title="Dispersión: Recencia vs. Monetario", height=420)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_radar(chart_data: dict) -> None:
    section_intro(
        "Perfil relativo por segmento",
        "Variables normalizadas entre 0 y 1. La recencia invertida premia compras recientes.",
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
                opacity=0.75,
            )
        )

    fig = style_figure(fig, height=420)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_quantitative_profile(chart_data: dict, results: pd.DataFrame) -> None:
    section_intro("Perfil cuantitativo", "Promedios transaccionales observados por cluster.")

    centroids = chart_data.get("centroids_normalized", [])
    if centroids and "raw_Recency" in centroids[0]:
        profile_rows = []
        for c in centroids:
            profile_rows.append({
                "Segmento": c["cluster_name"],
                "Recencia media": f"{c['raw_Recency']:.1f} días",
                "Frecuencia media": f"{c['raw_Frequency']:.2f} pedidos",
                "Monetario medio": f"R$ {c['raw_Monetary']:.2f}",
                "Ticket promedio": f"R$ {c['raw_AvgTicket']:.2f}",
            })
        profile_df = pd.DataFrame(profile_rows)
    else:
        profile_df = (
            results.groupby("cluster_name")
            .agg(
                Recencia_media=("R", "mean"),
                Frecuencia_media=("F", "mean"),
                Monetario_medio=("M", "mean"),
            )
            .round(2)
            .reset_index()
            .rename(columns={"cluster_name": "Segmento"})
        )
    st.dataframe(profile_df, hide_index=True, use_container_width=True)


def render(module: dict) -> None:
    """Punto de entrada principal para la vista de Clustering."""
    # 1. Cabecera y contrato común del módulo
    render_module(module, "Cliente, cluster, métricas RFM, perfil del segmento y Silhouette Score.")

    if not module.get("manifest_available"):
        return

    st.markdown("---")

    # 2. Cargar datos preparados
    chart_data = _load_chart_data()
    results = _load_results()

    if not chart_data or results is None or results.empty:
        st.warning("No se encontraron chart_data.json o results.csv en artifacts/clustering/.")
        return

    # 3. Métricas clave
    section_intro("Desempeño del modelo", "Métricas del particionamiento ganador frente al objetivo.")
    metrics = module.get("metrics", {})
    sil = metrics.get("silhouette_score")
    db = metrics.get("davies_bouldin_index")
    ch = metrics.get("calinski_harabasz_index")

    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{sil:.4f}" if isinstance(sil, (int, float)) else "N/A", help="Objetivo: >= 0.40")
    col2.metric("Davies-Bouldin", f"{db:.4f}" if isinstance(db, (int, float)) else "N/A", help="Menor es mejor")
    col3.metric("Calinski-Harabasz", f"{ch:,.1f}" if isinstance(ch, (int, float)) else "N/A", help="Mayor es mejor")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Distribución de clientes
    _render_bar_counts(chart_data)

    st.markdown("---")

    # 5. Pestañas de visualización
    tab_3d, tab_2d, tab_radar = st.tabs(["Espacio 3D", "Dispersión 2D", "Perfil radar"])

    with tab_3d:
        _render_scatter_3d(results)

    with tab_2d:
        _render_scatter_2d(results)

    with tab_radar:
        _render_radar(chart_data)

    st.markdown("---")

    # 6. Tabla de perfil cuantitativo
    _render_quantitative_profile(chart_data, results)

    st.markdown("---")

    # 7. Tabla de resultados de muestra
    section_intro("Muestra de resultados", "Muestra estratificada de 500 clientes con asignación de segmento.")

    filter_col1, _ = st.columns([2, 2])
    available_clusters = ["Todos"] + sorted(results["cluster_name"].unique().tolist())
    selected_cluster = filter_col1.selectbox("Filtrar por segmento:", available_clusters)

    filtered_df = results if selected_cluster == "Todos" else results[results["cluster_name"] == selected_cluster]

    st.dataframe(
        filtered_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "customer_unique_id": st.column_config.TextColumn("ID Cliente", width="large"),
            "R": st.column_config.NumberColumn("Recencia (días)", format="%d"),
            "F": st.column_config.NumberColumn("Frecuencia", format="%d"),
            "M": st.column_config.NumberColumn("Monetario (R$)", format="%.2f"),
            "cluster": st.column_config.NumberColumn("Cluster"),
            "cluster_name": st.column_config.TextColumn("Segmento"),
            "AvgTicket": st.column_config.NumberColumn("Ticket promedio (R$)", format="%.2f"),
        },
    )

    st.markdown("---")

    # 8. Conclusiones y limitaciones sintéticas
    conclusions = module.get("business_conclusions", [])
    limitations = module.get("limitations", [])

    col_conc, col_lim = st.columns(2)

    with col_conc:
        section_intro("Conclusiones de negocio", "Hallazgos accionables derivados de la segmentación.")
        for idx, item in enumerate(conclusions, start=1):
            st.markdown(f"**{idx}.** {item}")

    with col_lim:
        section_intro("Limitaciones", "Restricciones metodológicas del modelo.")
        for idx, item in enumerate(limitations, start=1):
            st.markdown(f"- {item}")
