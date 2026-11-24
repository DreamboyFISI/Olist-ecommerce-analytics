"""Página de Explicabilidad: por qué cada modelo decide lo que decide."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import PALETTE, page_header, render_empty_state, section_intro, style_figure

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
TERRACOTA = "#E07B54"


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _barras(df: pd.DataFrame, x: str, y: str, titulo: str, color=None, height: int = 420) -> go.Figure:
    df = df.sort_values(x)
    fig = go.Figure(go.Bar(x=df[x], y=df[y], orientation="h",
                           marker_color=color if color is not None else PALETTE["bondi"],
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    return style_figure(fig, titulo, height=height)


def _clasificacion() -> None:
    chart = _json(ARTIFACTS / "classification" / "chart_data.json")
    if not chart:
        render_empty_state("Clasificación sin artefactos", "Ejecuta python -m src.models.classification.")
        return
    st.markdown("**Importancia global por permutación** (cuánto cae la PR-AUC al desordenar cada variable) y, en la "
                "página Clasificación, la explicación de cada pedido evaluado con SHAP, LIME o perturbación.")
    imp = pd.DataFrame(chart["feature_importance"])
    colores = [TERRACOTA if f.startswith("Log") else PALETTE["bondi"] for f in imp.sort_values("importancia").familia]
    st.plotly_chart(_barras(imp, "importancia", "variable",
                            "Clasificación · naranja: logística observada, azul: conocida al comprar", colores),
                    use_container_width=True, config={"displaylogo": False})
    esc = pd.DataFrame(chart.get("scenario_comparison", []))
    if not esc.empty:
        st.caption("Solo con lo que se sabe al comprar, la PR-AUC cae de "
                   f"{esc.pr_auc.iloc[0]:.3f} a {esc.pr_auc.iloc[1]:.3f}: la explicación del riesgo está en la logística.")


def _clustering() -> None:
    chart = _json(ARTIFACTS / "clustering" / "chart_data.json")
    if not chart:
        render_empty_state("Clustering sin artefactos", "Ejecuta python -m src.models.clustering.")
        return
    st.markdown(f"**Árbol de decisión sustituto** (profundidad 3) que reproduce el {chart.get('tree_accuracy', 0) * 100:.1f}% "
                "de las asignaciones de K-Means. Son las reglas con las que cualquiera puede ubicar a un cliente:")
    st.code(chart.get("tree_rules", ""), language="text")
    perfiles = pd.DataFrame(chart.get("centroids_normalized", []))
    if not perfiles.empty:
        cols = [c for c in perfiles.columns if c.startswith("raw_")] + ["cluster_name"]
        tabla = perfiles[cols].rename(columns=lambda c: c.replace("raw_", ""))
        st.dataframe(tabla, hide_index=True, use_container_width=True)


def _regresion() -> None:
    ruta = ARTIFACTS / "regression" / "high_feature_importance.csv"
    if not ruta.exists():
        render_empty_state("Regresión sin importancias", "El notebook de regresión exporta high_feature_importance.csv.")
        return
    imp = pd.read_csv(ruta)
    col = next((c for c in imp.columns if c.startswith("importancia")), imp.columns[2])
    top = imp.nsmallest(15, "posicion") if "posicion" in imp else imp.head(15)
    st.markdown("**Importancia por permutación en el modelo de fletes altos**: cuánto crece el error (MAE) al desordenar la variable.")
    st.plotly_chart(_barras(top.rename(columns={col: "importancia"}), "importancia", "variable",
                            "Regresión · variables que más cambian el costo del flete estimado"),
                    use_container_width=True, config={"displaylogo": False})
    if "descripcion" in imp:
        st.dataframe(top[["variable", "descripcion", "origen"]].rename(columns=str.capitalize),
                     hide_index=True, use_container_width=True)


def _sentimientos() -> None:
    ruta = ARTIFACTS / "nlp" / "terminos_por_clase.csv"
    if not ruta.exists():
        render_empty_state("Sentimientos sin términos", "Ejecuta python -m src.models.sentiment.")
        return
    terminos = pd.read_csv(ruta)
    st.markdown("**Términos que más pesan** en el modelo lineal de sentimientos: lo que el modelo lee para decidir.")
    clase = st.selectbox("Clase", sorted(terminos.clase.unique()), index=0)
    grupo_opts = sorted(terminos.grupo.unique())
    grupo = st.radio("Grupo de términos", grupo_opts, horizontal=True)
    sel = terminos[(terminos.clase == clase) & (terminos.grupo == grupo)].nlargest(15, "peso")
    st.plotly_chart(_barras(sel, "peso", "termino", f"Sentimientos · {clase} · {grupo}", TERRACOTA, 420),
                    use_container_width=True, config={"displaylogo": False})


def _series_y_grafos() -> None:
    ts = _json(ARTIFACTS / "timeseries" / "manifest.json")
    gr = _json(ARTIFACTS / "graphs" / "manifest.json")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Series temporales · por qué ganó SARIMA**")
        for linea in (ts or {}).get("why_winner", ["Sin manifiesto."]):
            st.markdown(f"- {linea}")
    with c2:
        st.markdown("**Grafos · qué explica la red**")
        for linea in (gr or {}).get("business_conclusions", ["Sin manifiesto."]):
            st.markdown(f"- {linea}")


def render() -> None:
    page_header("Explicabilidad", "Por qué los modelos deciden así",
                "No basta con saber qué predice un modelo: aquí se muestra qué variables, reglas o términos "
                "sostienen cada resultado.")
    section_intro("Métodos usados",
                  "Importancia por permutación (agnóstica al modelo), SHAP y LIME para explicar pedidos "
                  "individuales, un árbol sustituto para los segmentos y los pesos del modelo lineal de texto.")
    tabs = st.tabs(["Clasificación", "Clustering", "Regresión", "Sentimientos", "Series y grafos"])
    with tabs[0]:
        _clasificacion()
    with tabs[1]:
        _clustering()
    with tabs[2]:
        _regresion()
    with tabs[3]:
        _sentimientos()
    with tabs[4]:
        _series_y_grafos()
