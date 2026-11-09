import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import (
    page_header,
    render_empty_state,
    section_intro,
    status_badge,
    style_figure,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHART_DATA = ROOT / "artifacts" / "graphs" / "chart_data.json"
ITEMS = ROOT / "data" / "processed" / "grafos" / "items_pedidos.csv"
RAW_DIR = ROOT / "data" / "raw"


def _chart_path(module: dict) -> Path:
    """Respeta chart_data_path del manifest; usa la ruta habitual como respaldo."""
    configured = module.get("chart_data_path")
    if not configured:
        return DEFAULT_CHART_DATA
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def _newest_mtime(paths: Iterable[Path]) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else None


def _artifact_status(chart_path: Path) -> tuple[str, str] | None:
    """Indica si los artefactos quedaron atrás respecto a los datos."""
    if not chart_path.exists():
        return None

    raw_mtime = _newest_mtime(RAW_DIR.glob("*.csv"))
    items_mtime = ITEMS.stat().st_mtime if ITEMS.exists() else None
    chart_mtime = chart_path.stat().st_mtime

    if raw_mtime is not None and (items_mtime is None or raw_mtime > items_mtime):
        return (
            "data",
            "Los CSV de `data/raw/` son más recientes que `items_pedidos.csv`. "
            "Ejecuta `python -m src.models.graphs --refresh` para reconstruir "
            "los datos y los artefactos.",
        )
    if items_mtime is not None and items_mtime > chart_mtime:
        return (
            "chart",
            "`items_pedidos.csv` es más reciente que `chart_data.json`. "
            "Ejecuta `python -m src.models.graphs` para actualizar el diagrama.",
        )
    return None


def _refresh_command(kind: str) -> list[str]:
    command = [sys.executable, "-m", "src.models.graphs"]
    if kind == "data":
        command.append("--refresh")
    return command


def _can_refresh() -> bool:
    return (ROOT / "src" / "models" / "graphs.py").exists() and (
        ITEMS.exists() or RAW_DIR.exists()
    )


def _network_figure(data: dict) -> go.Figure:
    nodes = pd.DataFrame(data["grafo"]["nodos"])
    edges = pd.DataFrame(data["grafo"]["aristas"])
    graph = nx.Graph()
    graph.add_nodes_from(nodes.nodo)
    graph.add_edges_from(edges[["origen", "destino"]].itertuples(index=False, name=None))

    positions = nx.spring_layout(graph, seed=42, k=0.35, iterations=60)
    nodes["x"] = nodes.nodo.map(lambda node: positions[node][0])
    nodes["y"] = nodes.nodo.map(lambda node: positions[node][1])

    edge_x, edge_y = [], []
    for row in edges.itertuples(index=False):
        edge_x.extend([positions[row.origen][0], positions[row.destino][0], None])
        edge_y.extend([positions[row.origen][1], positions[row.destino][1], None])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 0.7, "color": "rgba(17,76,95,0.25)"},
            hoverinfo="none",
            showlegend=False,
        )
    )
    for node_type, subset in nodes.groupby("tipo"):
        figure.add_trace(
            go.Scatter(
                x=subset.x,
                y=subset.y,
                mode="markers",
                name=node_type,
                marker={
                    "size": subset.grado.clip(lower=3).pow(0.55) * 3,
                    "color": subset.comunidad,
                    "colorscale": "Viridis",
                    "showscale": node_type == "categoria",
                    "colorbar": {"title": "Comunidad", "thickness": 12},
                    "line": {"width": 0.5, "color": "white"},
                },
                customdata=subset[
                    ["etiqueta", "nodo", "grado", "pagerank", "comunidad"]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"Tipo: {node_type}<br>"
                    "ID: %{customdata[1]}<br>"
                    "Grado: %{customdata[2]}<br>"
                    "PageRank: %{customdata[3]:.6f}<br>"
                    "Comunidad: %{customdata[4]}<extra></extra>"
                ),
            )
        )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return style_figure(figure, "Red categoría–vendedor (aristas de mayor peso)", height=520)


def _table(
    rows: list[dict],
    columns: list[str],
    labels: dict[str, str],
    key: str | None = None,
):
    frame = pd.DataFrame(rows)[columns].rename(columns=labels)
    if key is None:
        st.dataframe(frame, hide_index=True, use_container_width=True, height="auto")
        return
    with st.container(key=f"graphs_fit_{key}"):
        st.dataframe(frame, hide_index=True, use_container_width=True, height="auto")


def render(module: dict) -> None:
    page_header("Módulo de grafos", module["title"], module["question"])
    st.markdown(
        f"{status_badge(module.get('status', 'Pendiente'))} &nbsp; "
        f"<code>{module['branch']}</code>",
        unsafe_allow_html=True,
    )

    chart_path = _chart_path(module)
    chart_reference = module.get("chart_data_path") or "artifacts/graphs/chart_data.json"
    if not module.get("manifest_available") or not chart_path.exists():
        render_empty_state(
            "Módulo preparado para su rama",
            "La navegación y el espacio visual ya están listos. Esta pantalla se activará "
            f"al agregar artifacts/graphs/manifest.json y {chart_reference}.",
        )
        return

    stale = _artifact_status(chart_path)
    if stale:
        kind, message = stale
        st.warning(message)
        if _can_refresh():
            if st.button("Actualizar grafos ahora", key="graphs_refresh", type="primary"):
                try:
                    with st.spinner("Reconstruyendo datos y artefactos..."):
                        subprocess.run(
                            _refresh_command(kind),
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                except subprocess.CalledProcessError as exc:
                    st.error("No se pudieron actualizar los grafos.")
                    st.code(((exc.stderr or "") + (exc.stdout or ""))[-4000:])
                else:
                    st.success("Artefactos actualizados. Recargando la vista...")
                    st.rerun()

    data = json.loads(chart_path.read_text(encoding="utf-8"))
    metrics = module.get("metrics", {})

    st.caption(f"Periodo: {module.get('period')} · Dataset: {module.get('dataset')}")

    section_intro(
        "Indicadores clave",
        "Un resumen de la red construida: tamaño, conectividad y calidad de las comunidades.",
    )
    row = st.columns(4)
    row[0].metric("Nodos", f"{int(metrics.get('nodos_grafo_principal', 0)):,}")
    row[1].metric("Aristas", f"{int(metrics.get('aristas_grafo_principal', 0)):,}")
    row[2].metric("Comunidades", f"{int(metrics.get('comunidades_ganador', 0)):,}")
    row[3].metric("Modularidad", f"{metrics.get('modularidad_ganador', 0):.4f}")
    st.markdown(
        "<div class='module-note'><strong>Modelo ganador:</strong> "
        f"{module.get('winner', 'Sin informar')} · "
        f"Cobertura de nodos: {metrics.get('cobertura_ganador', 0):.0%} · "
        f"Conectividad del grafo: {metrics.get('conectividad_grafo_principal_%', 0):.1f}%</div>",
        unsafe_allow_html=True,
    )
    shown_nodes = len(data["grafo"]["nodos"])
    shown_edges = len(data["grafo"]["aristas"])
    total_nodes = int(metrics.get("nodos_grafo_principal", 0))
    total_edges = int(metrics.get("aristas_grafo_principal", 0))
    results_path = module.get("results_path", "artifacts/graphs/nodos.csv")
    st.caption(
        f"El diagrama muestra {shown_edges:,} de {total_edges:,} aristas y "
        f"{shown_nodes:,} de {total_nodes:,} nodos, seleccionados por peso para "
        "mantenerlo legible."
    )
    st.caption(
        "Las etiquetas muestran los primeros 8 caracteres del ID. "
        f"Los IDs completos se conservan en {results_path}."
    )

    st.markdown(
        """
        <style>
        [class*="st-key-graphs_tables"] [data-testid="stDataFrame"] {
            width: 100%;
            box-sizing: border-box;
        }
        [class*="st-key-graphs_tables"] [data-testid="stDataFrameResizable"] {
            width: 100% !important;
        }
        [class*="st-key-graphs_fit"] [data-testid="stDataFrame"] {
            width: 100%;
            box-sizing: border-box;
            padding: .4rem .4rem .1rem;
        }
        [class*="st-key-graphs_fit"] [data-testid="stDataFrameResizable"] {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="graphs_tables"):
        tab_red, tab_centralidad, tab_co_compra, tab_conclusiones = st.tabs(
            ["Red y comunidades", "Centralidad", "Co-compra", "Conclusiones"]
        )
    
        with tab_red:
            st.plotly_chart(
                _network_figure(data), use_container_width=True, config={"displaylogo": False}
            )
            section_intro(
                "Comparación de algoritmos",
                "Modularidad, número de comunidades, cobertura y tiempo de cómputo. "
                "El ganador se elige por modularidad; si hay empate técnico, por tiempo.",
            )
            comparacion = pd.DataFrame(data["comparacion"]).sort_values(
                "modularidad", ascending=False
            )
            st.dataframe(
                comparacion.rename(
                    columns={
                        "algoritmo": "Algoritmo",
                        "modularidad": "Modularidad",
                        "comunidades": "Comunidades",
                        "cobertura": "Cobertura",
                        "tiempo_s": "Tiempo (s)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            section_intro(
                "Comunidades detectadas",
                "Cada comunidad mezcla categorías y vendedores con un patrón de surtido similar.",
            )
            _table(
                data["comunidades"],
                [
                    "comunidad",
                    "nodos",
                    "categorias",
                    "vendedores",
                    "grado_promedio",
                    "top_nodos",
                ],
                {
                    "comunidad": "Comunidad",
                    "nodos": "Nodos",
                    "categorias": "Categorías",
                    "vendedores": "Vendedores",
                    "grado_promedio": "Grado promedio",
                    "top_nodos": "Nodos principales",
                },
            )
    
        with tab_centralidad:
            left, right = st.columns(2, gap="small")
            with left:
                st.subheader("Categorías con más vendedores")
                _table(
                    sorted(data["top_categorias"], key=lambda r: -r["grado"]),
                    ["etiqueta", "grado"],
                    {"etiqueta": "Categoría", "grado": "Vendedores"},
                    key="categorias",
                )
            with right:
                st.subheader("Vendedores con más categorías")
                _table(
                    sorted(data["top_vendedores"], key=lambda r: -r["grado"]),
                    ["etiqueta", "grado"],
                    {"etiqueta": "Vendedor", "grado": "Categorías"},
                    key="vendedores",
                )
    
            left, right = st.columns(2, gap="small")
            with left:
                st.subheader("Productos con mayor PageRank")
                _table(
                    sorted(data["top_productos"], key=lambda r: -r["pagerank"]),
                    ["etiqueta", "categoria", "pagerank"],
                    {
                        "etiqueta": "Producto",
                        "categoria": "Categoría",
                        "pagerank": "PageRank",
                    },
                    key="productos",
                )
            with right:
                st.subheader("Nodos puente")
                _table(
                    sorted(data["top_betweenness"], key=lambda r: -r["betweenness"]),
                    ["etiqueta", "betweenness"],
                    {"etiqueta": "Nodo", "betweenness": "Betweenness"},
                    key="betweenness",
                )
    
        with tab_co_compra:
            st.info(
                f"Solo {metrics.get('pedidos_con_2_o_mas_productos', 0):,} pedidos tienen dos o más "
                "productos distintos. La co-compra se usa como análisis complementario, no como "
                "base del módulo."
            )
            left, right = st.columns(2, gap="small")
            with left:
                st.subheader("Categorías que se compran juntas")
                _table(
                    sorted(data["co_compra_categorias"], key=lambda r: -r["pedidos"]),
                    ["categoria_a", "categoria_b", "pedidos"],
                    {
                        "categoria_a": "Categoría A",
                        "categoria_b": "Categoría B",
                        "pedidos": "Pedidos",
                    },
                    key="co_categorias",
                )
            with right:
                st.subheader("Productos que se compran juntos")
                _table(
                    sorted(data["co_compra_productos"], key=lambda r: -r["pedidos"]),
                    ["etiqueta_a", "etiqueta_b", "pedidos"],
                    {
                        "etiqueta_a": "Producto A",
                        "etiqueta_b": "Producto B",
                        "pedidos": "Pedidos",
                    },
                    key="co_productos",
                )
    
        with tab_conclusiones:
            left, right = st.columns(2)
            with left:
                st.subheader("Conclusiones de negocio")
                for conclusion in module.get("business_conclusions", []):
                    st.markdown(f"- {conclusion}")
            with right:
                st.subheader("Limitaciones")
                for limitation in module.get("limitations", []):
                    st.markdown(f"- {limitation}")
            with st.expander("Detalle técnico y trazabilidad"):
                detalle = pd.Series(metrics, name="Valor").rename_axis("Métrica").reset_index()
                st.dataframe(detalle, hide_index=True, use_container_width=True)
                st.caption(
                    f"Resultados: {module.get('results_path')} · "
                    f"Datos de gráficos: {module.get('chart_data_path')}"
                )
                st.caption(
                    "Modelos previstos para el módulo: "
                    + ", ".join(module.get("expected_models", []))
                    + "."
                )
