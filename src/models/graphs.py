"""Módulo de grafos de Olist (fase 4 de CRISP-DM).

    python -m src.models.graphs

Grafo principal: bipartito categoría–vendedor (100% conexo).
Grafos complementarios: producto–vendedor y co-compra producto–producto.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import community as community_louvain
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "data" / "processed" / "grafos" / "items_pedidos.csv"
ARTIFACTS = ROOT / "artifacts" / "graphs"
PERIODO = "2016-09-04 a 2018-09-03"


def load_items(path: Path = ITEMS) -> pd.DataFrame:
    dtype = {"order_id": "string", "product_id": "string", "seller_id": "string"}
    df = pd.read_csv(path, dtype=dtype)
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    return df


def build_bipartite(
    df: pd.DataFrame, left: str, right: str, left_type: str, right_type: str
) -> nx.Graph:
    """Grafo bipartito con peso = cantidad de ítems que conectan ambos nodos."""
    edges = (
        df.groupby([left, right], observed=True).size().reset_index(name="peso")
    )
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(getattr(row, left), getattr(row, right), weight=int(row.peso))

    left_nodes = set(edges[left])
    right_nodes = set(edges[right]) - left_nodes
    nx.set_node_attributes(graph, {node: left_type for node in left_nodes}, "tipo")
    nx.set_node_attributes(graph, {node: right_type for node in right_nodes}, "tipo")
    return graph


def build_cooccurrence(df: pd.DataFrame, entity: str, entity_type: str) -> nx.Graph:
    """Grafo de co-ocurrencia: nodos comprados/vendidos en el mismo pedido."""
    pairs: Counter[tuple[str, str]] = Counter()
    for _, group in df.groupby("order_id", observed=True)[entity]:
        values = sorted(set(group.dropna()))
        if len(values) > 1:
            pairs.update(combinations(values, 2))
    graph = nx.Graph()
    for (a, b), peso in pairs.items():
        graph.add_edge(a, b, weight=int(peso))
    nx.set_node_attributes(graph, entity_type, "tipo")
    return graph


def graph_stats(graph: nx.Graph) -> dict:
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    giant = components[0] if components else set()
    degrees = [degree for _, degree in graph.degree()]
    return {
        "nodos": graph.number_of_nodes(),
        "aristas": graph.number_of_edges(),
        "componentes": len(components),
        "componente_gigante": len(giant),
        "componente_gigante_%": round(len(giant) / graph.number_of_nodes() * 100, 2)
        if graph.number_of_nodes()
        else 0.0,
        "grado_promedio": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
        "grado_maximo": max(degrees) if degrees else 0,
        "nodos_aislados": sum(1 for degree in degrees if degree == 0),
    }


def compare_communities(graph: nx.Graph) -> tuple[dict[str, dict], str]:
    """Compara Louvain, Greedy Modularity y Label Propagation por modularidad."""
    results: dict[str, dict] = {}

    start = time.perf_counter()
    louvain = community_louvain.best_partition(
        graph, weight="weight", random_state=42
    )
    results["Louvain"] = {
        "particion": louvain,
        "modularidad": nx.community.modularity(graph, _groups(louvain), weight="weight"),
        "comunidades": len(set(louvain.values())),
        "cobertura": round(len(louvain) / graph.number_of_nodes(), 4),
        "tiempo_s": round(time.perf_counter() - start, 2),
    }

    start = time.perf_counter()
    greedy = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    results["Greedy Modularity"] = {
        "particion": {
            node: index for index, group in enumerate(greedy) for node in group
        },
        "modularidad": nx.community.modularity(graph, greedy, weight="weight"),
        "comunidades": len(greedy),
        "cobertura": round(sum(map(len, greedy)) / graph.number_of_nodes(), 4),
        "tiempo_s": round(time.perf_counter() - start, 2),
    }

    start = time.perf_counter()
    lpa = list(nx.community.label_propagation_communities(graph))
    results["Label Propagation"] = {
        "particion": {
            node: index for index, group in enumerate(lpa) for node in group
        },
        "modularidad": nx.community.modularity(graph, lpa, weight="weight"),
        "comunidades": len(lpa),
        "cobertura": round(sum(map(len, lpa)) / graph.number_of_nodes(), 4),
        "tiempo_s": round(time.perf_counter() - start, 2),
    }

    for result in results.values():
        result["modularidad"] = round(float(result["modularidad"]), 4)

    # Empate técnico: si la diferencia de modularidad es despreciable, gana el más rápido.
    best = max(result["modularidad"] for result in results.values())
    candidates = [
        name
        for name, result in results.items()
        if best - result["modularidad"] <= 0.001
    ]
    winner = min(candidates, key=lambda name: results[name]["tiempo_s"])
    return results, winner


def _groups(partition: dict[str, int]) -> list[set[str]]:
    groups: dict[int, set[str]] = {}
    for node, community in partition.items():
        groups.setdefault(community, set()).add(node)
    return list(groups.values())


def top_degree(graph: nx.Graph, node_type: str, limit: int = 10) -> list[dict]:
    rows = [
        {"nodo": node, "grado": degree}
        for node, degree in graph.degree()
        if graph.nodes[node].get("tipo") == node_type
    ]
    rows.sort(key=lambda row: (-row["grado"], row["nodo"]))
    return rows[:limit]


def product_catalog(df: pd.DataFrame) -> dict[str, str]:
    catalog = df.drop_duplicates("product_id")[["product_id", "categoria_en"]]
    return dict(catalog.itertuples(index=False, name=None))


def category_projection(graph: nx.Graph) -> tuple[nx.Graph, list[dict]]:
    """Proyecta categorías conectadas por vendedores compartidos."""
    categories = [
        node for node, data in graph.nodes(data=True) if data.get("tipo") == "categoria"
    ]
    projection = nx.Graph()
    projection.add_nodes_from(categories)
    rows = []
    for a, b in combinations(categories, 2):
        shared = len(set(graph[a]) & set(graph[b]))
        if shared:
            projection.add_edge(a, b, weight=shared)
            rows.append({"categoria_a": a, "categoria_b": b, "vendedores_compartidos": shared})
    rows.sort(key=lambda row: (-row["vendedores_compartidos"], row["categoria_a"]))
    return projection, rows[:10]


def betweenness_top(graph: nx.Graph, limit: int = 10, k: int = 100) -> list[dict]:
    """Betweenness aproximada sobre el componente gigante (k muestras)."""
    giant = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(giant)
    scores = nx.betweenness_centrality(subgraph, k=min(k, len(subgraph)), seed=42)
    rows = [
        {"nodo": node, "betweenness": round(score, 6)}
        for node, score in scores.items()
    ]
    rows.sort(key=lambda row: (-row["betweenness"], row["nodo"]))
    return rows[:limit]


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = load_items()

    # 1. Grafo principal: categoría–vendedor.
    categorias = build_bipartite(
        df, "categoria_en", "seller_id", "categoria", "vendedor"
    )
    cat_stats = graph_stats(categorias)
    comparacion, winner = compare_communities(categorias)
    winning_partition = comparacion[winner]["particion"]

    pagerank_cat = nx.pagerank(categorias, weight="weight")
    pagerank_productos = None

    nodes = pd.DataFrame(
        [
            {
                "nodo": node,
                "tipo": data.get("tipo"),
                "grado": categorias.degree(node),
                "pagerank": round(pagerank_cat.get(node, 0.0), 6),
                "comunidad": winning_partition.get(node),
            }
            for node, data in categorias.nodes(data=True)
        ]
    ).sort_values(["comunidad", "grado"], ascending=[True, False])
    nodes.to_csv(ARTIFACTS / "nodos.csv", index=False)

    edges = pd.DataFrame(
        [
            {"origen": a, "destino": b, "peso": data["weight"]}
            for a, b, data in categorias.edges(data=True)
        ]
    ).sort_values("peso", ascending=False)
    edges.to_csv(ARTIFACTS / "aristas.csv", index=False)

    community_rows = []
    for community, group in nodes.groupby("comunidad", dropna=False):
        subset = group.sort_values("grado", ascending=False)
        community_rows.append(
            {
                "comunidad": int(community),
                "nodos": len(group),
                "categorias": int((group.tipo == "categoria").sum()),
                "vendedores": int((group.tipo == "vendedor").sum()),
                "grado_promedio": round(group.grado.mean(), 2),
                "top_nodos": ", ".join(subset.nodo.head(5)),
            }
        )
    communities = pd.DataFrame(community_rows).sort_values("nodos", ascending=False)
    communities.to_csv(ARTIFACTS / "comunidades.csv", index=False)

    # 2. Grafo producto–vendedor (centralidad de productos y vendedores).
    producto_vendedor = build_bipartite(
        df, "product_id", "seller_id", "producto", "vendedor"
    )
    pv_stats = graph_stats(producto_vendedor)
    pagerank_productos = nx.pagerank(producto_vendedor, weight="weight")
    catalogo = product_catalog(df)

    top_productos_pagerank = sorted(
        (
            {
                "producto": node,
                "categoria": catalogo.get(node, "sin_categoria"),
                "pagerank": round(score, 6),
                "grado": producto_vendedor.degree(node),
            }
            for node, score in pagerank_productos.items()
            if producto_vendedor.nodes[node].get("tipo") == "producto"
        ),
        key=lambda row: -row["pagerank"],
    )[:10]
    top_vendedores_grado = top_degree(producto_vendedor, "vendedor")
    betweenness = betweenness_top(producto_vendedor)

    # 3. Co-compra producto–producto y categoría–categoría (complementario).
    co_compra = build_cooccurrence(df, "product_id", "producto")
    co_compra_stats = graph_stats(co_compra)
    co_compra_pairs = sorted(
        (
            {"producto_a": a, "producto_b": b, "pedidos": data["weight"]}
            for a, b, data in co_compra.edges(data=True)
        ),
        key=lambda row: (-row["pedidos"], row["producto_a"]),
    )[:20]
    pd.DataFrame(co_compra_pairs).to_csv(ARTIFACTS / "pares_co_compra.csv", index=False)

    co_categoria = build_cooccurrence(df, "categoria_en", "categoria")
    co_categoria_pairs = sorted(
        (
            {"categoria_a": a, "categoria_b": b, "pedidos": data["weight"]}
            for a, b, data in co_categoria.edges(data=True)
        ),
        key=lambda row: (-row["pedidos"], row["categoria_a"]),
    )[:20]
    pd.DataFrame(co_categoria_pairs).to_csv(
        ARTIFACTS / "pares_categorias.csv", index=False
    )

    # 4. Proyección categoría–categoría.
    projection, projection_pairs = category_projection(categorias)

    comparison_rows = [
        {
            "algoritmo": name,
            "modularidad": result["modularidad"],
            "comunidades": result["comunidades"],
            "cobertura": result["cobertura"],
            "tiempo_s": result["tiempo_s"],
        }
        for name, result in comparacion.items()
    ]
    pd.DataFrame(comparison_rows).to_csv(
        ARTIFACTS / "comparacion_algoritmos.csv", index=False
    )

    metrics = {
        "grafo_principal": {
            "nombre": "bipartito categoria-vendedor",
            **cat_stats,
        },
        "grafo_producto_vendedor": {"nombre": "bipartito producto-vendedor", **pv_stats},
        "grafo_co_compra": {"nombre": "co-compra producto-producto", **co_compra_stats},
        "grafo_categoria_categoria": {
            "nombre": "co-compra categoria-categoria",
            **graph_stats(co_categoria),
        },
        "proyeccion_categoria": {
            "nodos": projection.number_of_nodes(),
            "aristas": projection.number_of_edges(),
            "top_pares": projection_pairs,
        },
        "comparacion_comunidades": comparison_rows,
        "ganador": winner,
        "criterio_ganador": "mayor modularidad; empate técnico (Δ≤0.001) se resuelve por menor tiempo",
        "top_categorias_por_vendedores": top_degree(categorias, "categoria"),
        "top_vendedores_por_categorias": top_degree(categorias, "vendedor"),
        "top_productos_pagerank": top_productos_pagerank,
        "top_vendedores_por_productos": top_vendedores_grado,
        "betweenness_aproximada": betweenness,
        "top_pares_co_compra": co_compra_pairs[:10],
        "top_pares_categorias": co_categoria_pairs[:10],
    }
    with (ARTIFACTS / "metricas.json").open("w", encoding="utf-8") as salida:
        json.dump(metrics, salida, ensure_ascii=False, indent=2)

    # 5. Datos para la vista (subconjunto legible).
    # Etiquetas solo visuales: los artefactos conservan los IDs originales.
    top_edges = edges.head(120)
    keep = set(top_edges.origen) | set(top_edges.destino)
    seller_labels = {
        node: f"Vendedor {str(node)[:8]}"
        for node in categorias
        if categorias.nodes[node].get("tipo") == "vendedor"
    }
    product_labels = {
        node: f"Producto {str(node)[:8]}"
        for node in producto_vendedor
        if producto_vendedor.nodes[node].get("tipo") == "producto"
    }

    def display_label(node: str, node_type: str | None = None) -> str:
        if node_type == "vendedor":
            return seller_labels.get(node, node)
        if node_type == "producto":
            return product_labels.get(node, node)
        return node

    chart_nodes = nodes[nodes.nodo.isin(keep)].copy()
    chart_nodes["etiqueta"] = [
        display_label(node, node_type)
        for node, node_type in zip(chart_nodes.nodo, chart_nodes.tipo)
    ]

    communities_display = communities.head(10).copy()
    communities_display["top_nodos"] = communities_display.top_nodos.map(
        lambda value: ", ".join(
            display_label(node.strip(), categorias.nodes[node.strip()].get("tipo"))
            for node in value.split(",")[:2]
        )
    )

    top_categorias_display = [
        {**row, "etiqueta": row["nodo"]}
        for row in top_degree(categorias, "categoria")
    ]
    top_vendedores_display = [
        {**row, "etiqueta": seller_labels.get(row["nodo"], row["nodo"])}
        for row in top_degree(categorias, "vendedor")
    ]
    top_productos_display = [
        {**row, "etiqueta": product_labels.get(row["producto"], row["producto"])}
        for row in top_productos_pagerank
    ]
    betweenness_display = [
        {
            **row,
            "etiqueta": display_label(
                row["nodo"], producto_vendedor.nodes[row["nodo"]].get("tipo")
            ),
        }
        for row in betweenness
    ]
    co_compra_display = [
        {
            **row,
            "etiqueta_a": product_labels.get(row["producto_a"], row["producto_a"]),
            "etiqueta_b": product_labels.get(row["producto_b"], row["producto_b"]),
        }
        for row in co_compra_pairs[:10]
    ]

    chart_data = {
        "grafo": {
            "nodos": chart_nodes.to_dict(orient="records"),
            "aristas": top_edges.to_dict(orient="records"),
        },
        "comunidades": communities_display.to_dict(orient="records"),
        "top_categorias": top_categorias_display,
        "top_vendedores": top_vendedores_display,
        "top_productos": top_productos_display,
        "top_betweenness": betweenness_display,
        "co_compra_categorias": co_categoria_pairs[:10],
        "co_compra_productos": co_compra_display,
        "comparacion": comparison_rows,
    }
    with (ARTIFACTS / "chart_data.json").open("w", encoding="utf-8") as salida:
        json.dump(chart_data, salida, ensure_ascii=False, indent=2)

    # 6. Manifest del contrato común.
    mejor = comparacion[winner]
    manifest = {
        "status": "Completado",
        "model_name": f"{winner} sobre grafo bipartito categoría–vendedor",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "items_pedidos.csv (Olist, 112,650 ítems)",
        "period": PERIODO,
        "metrics": {
            "modularidad_ganador": mejor["modularidad"],
            "comunidades_ganador": mejor["comunidades"],
            "cobertura_ganador": mejor["cobertura"],
            "nodos_grafo_principal": cat_stats["nodos"],
            "aristas_grafo_principal": cat_stats["aristas"],
            "conectividad_grafo_principal_%": cat_stats["componente_gigante_%"],
            "componente_gigante_producto_vendedor_%": pv_stats["componente_gigante_%"],
            "pedidos_con_2_o_mas_productos": int(
                (df.groupby("order_id").product_id.nunique() >= 2).sum()
            ),
            "aristas_co_compra": co_compra_stats["aristas"],
        },
        "winner": winner,
        "results_path": "artifacts/graphs/nodos.csv",
        "chart_data_path": "artifacts/graphs/chart_data.json",
        "business_conclusions": [
            f"El grafo categoría–vendedor está {cat_stats['componente_gigante_%']}% conectado, "
            f"por lo que permite analizar todas las categorías y vendedores sin descartar nodos.",
            f"{metrics['top_categorias_por_vendedores'][0]['nodo']} es la categoría con más "
            f"vendedores ({metrics['top_categorias_por_vendedores'][0]['grado']}), seguida de "
            f"{metrics['top_categorias_por_vendedores'][1]['nodo']} "
            f"({metrics['top_categorias_por_vendedores'][1]['grado']}), lo que indica mercados "
            "con mayor competencia.",
            f"La co-compra de categorías más frecuente es "
            f"{co_categoria_pairs[0]['categoria_a']} + {co_categoria_pairs[0]['categoria_b']} "
            f"({co_categoria_pairs[0]['pedidos']} pedidos), una oportunidad de venta cruzada.",
            f"Solo {int((df.groupby('order_id').product_id.nunique() >= 2).sum()):,} pedidos "
            "incluyen dos o más productos distintos, por lo que el grafo de co-compra directa "
            "es disperso y debe complementarse con reglas de asociación.",
        ],
        "limitations": [
            "El grafo de co-compra producto–producto es disperso: el componente gigante cubre "
            f"{co_compra_stats['componente_gigante_%']}% de sus nodos.",
            "Las aristas categoría–vendedor representan presencia, no volumen de ventas ni ingresos.",
            "El periodo analizado (2016-09 a 2018-09) no refleja el comportamiento actual del mercado.",
            "Louvain es estocástico; se fijó random_state=42 para reproducibilidad.",
            "Betweenness se calculó de forma aproximada (k=100) sobre el componente gigante.",
        ],
    }
    with (ARTIFACTS / "manifest.json").open("w", encoding="utf-8") as salida:
        json.dump(manifest, salida, ensure_ascii=False, indent=2)

    print(f"Grafo principal: {cat_stats['nodos']:,} nodos, {cat_stats['aristas']:,} aristas")
    print(f"Ganador: {winner} | modularidad={mejor['modularidad']} | "
          f"comunidades={mejor['comunidades']}")
    print(f"Co-compra: {co_compra_stats['nodos']:,} nodos, "
          f"{co_compra_stats['aristas']:,} aristas, "
          f"gigante={co_compra_stats['componente_gigante_%']}%")
    print(f"Artefactos en {ARTIFACTS}")


if __name__ == "__main__":
    main()
