import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import (PALETTE, page_header, render_empty_state, section_intro,
                                 status_badge, style_figure)
from frontend.views.module_template import render_module

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts" / "timeseries"


def _cargar(nombre: str):
    ruta = ARTIFACTS / nombre
    if not ruta.exists():
        return None
    with ruta.open(encoding="utf-8") as fuente:
        return json.load(fuente)


def _grafico(chart: dict, dias_historia: int) -> go.Figure:
    serie = pd.DataFrame(chart["serie"]).tail(dias_historia)
    backtest = pd.DataFrame(chart["backtest"])
    futuro = pd.DataFrame(chart["futuro"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=futuro.fecha, y=futuro.superior, mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=futuro.fecha, y=futuro.inferior, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(74,110,176,0.18)", name="Intervalo 95%",
        hovertemplate="%{y:,.0f} BRL<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=serie.fecha, y=serie.ventas, mode="lines", name="Ventas reales",
        line=dict(color=PALETTE["eden"], width=1.4),
        hovertemplate="%{x}: %{y:,.0f} BRL<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=backtest.fecha, y=backtest.pronostico, mode="lines", name="Backtest",
        line=dict(color=PALETTE["bondi"], width=2.2, dash="dot"),
        hovertemplate="%{x}: %{y:,.0f} BRL<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=futuro.fecha, y=futuro.pronostico, mode="lines", name="Pronóstico 30 días",
        line=dict(color=PALETTE["san_marino"], width=2.6),
        hovertemplate="%{x}: %{y:,.0f} BRL<extra></extra>"))
    fig.update_yaxes(title="Ventas diarias (BRL)")
    return style_figure(fig, "Ventas reales, backtest y pronóstico", height=440)


def render(module: dict) -> None:
    manifest = _cargar("manifest.json")
    chart = _cargar("chart_data.json")

    if not manifest or not chart:
        render_module(module, "Fecha, valor real, pronóstico, intervalos de confianza "
                              "y error de backtesting.")
        return

    page_header("Módulo de modelamiento", module["title"], module["question"])
    st.markdown(f"{status_badge(manifest['status'])} &nbsp; <code>{module['branch']}</code> "
                f"&nbsp; v{manifest['version']}", unsafe_allow_html=True)

    m = manifest["metrics"]
    fila = st.columns(4)
    fila[0].metric("Modelo ganador", manifest["winner"].upper())
    fila[1].metric("MAE", f"R$ {m['MAE']:,.0f}", f"{m['mejora_vs_baseline_%']:+.1f}% vs baseline")
    fila[2].metric("sMAPE", f"{m['sMAPE_%']:.1f}%")
    fila[3].metric("Baseline", f"R$ {m['MAE_baseline']:,.0f}")

    st.caption(f"{manifest['model_name']} · {manifest['period']} · {manifest['validation']}")

    dias = st.slider("Días de historia en la gráfica", 60, 580, 180, step=20,
                     help="Controla cuánta serie histórica se muestra junto al pronóstico.")
    st.plotly_chart(_grafico(chart, dias), use_container_width=True,
                    config={"displaylogo": False})

    section_intro("Comparación de modelos",
                  "Los cuatro evaluados con el mismo walk-forward de 6 pliegues × 30 días.")
    comparacion = pd.DataFrame(chart["comparacion"]).rename(columns={
        "modelo": "Modelo", "MAE": "MAE", "RMSE": "RMSE", "MAPE": "MAPE %",
        "sMAPE": "sMAPE %", "mejora_MAE_%": "Mejora vs baseline %"})
    st.dataframe(comparacion, hide_index=True, use_container_width=True)

    section_intro("Por qué ganó", "La justificación que acompaña al modelo seleccionado.")
    for razon in manifest["why_winner"]:
        st.markdown(f"- {razon}")

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.subheader("Conclusiones de negocio")
        for linea in manifest["business_conclusions"]:
            st.markdown(f"- {linea}")
    with derecha:
        st.subheader("Limitaciones")
        for linea in manifest["limitations"]:
            st.markdown(f"- {linea}")

    with st.expander("Detalle por pliegue y descarga"):
        detalle = ARTIFACTS / "comparacion_por_pliegue.csv"
        if detalle.exists():
            st.dataframe(pd.read_csv(detalle).round(2), hide_index=True,
                         use_container_width=True)
        forecast = ARTIFACTS / "forecast.csv"
        if forecast.exists():
            st.download_button("Descargar forecast.csv", forecast.read_bytes(),
                               file_name="forecast.csv", mime="text/csv")
