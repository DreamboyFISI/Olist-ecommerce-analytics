"""Pruebas del módulo de grafos.

Se omiten las pruebas de artefactos si todavía no se ejecutó
``python -m src.models.graphs``.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.graphs import (  # noqa: E402
    build_bipartite,
    build_cooccurrence,
    compare_communities,
    graph_stats,
)

ARTIFACTS = ROOT / "artifacts" / "graphs"
ITEMS = ROOT / "data" / "processed" / "grafos" / "items_pedidos.csv"


@pytest.fixture
def items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3"],
            "product_id": ["p1", "p2", "p3", "p1"],
            "seller_id": ["s1", "s1", "s2", "s2"],
            "categoria_en": ["c1", "c2", "c1", "c1"],
        }
    )


def test_bipartito_conserva_tipos_y_pesos(items):
    graph = build_bipartite(items, "categoria_en", "seller_id", "categoria", "vendedor")

    assert graph.nodes["c1"]["tipo"] == "categoria"
    assert graph.nodes["s1"]["tipo"] == "vendedor"
    assert graph["c1"]["s1"]["weight"] == 1
    assert graph["c1"]["s2"]["weight"] == 2
    assert graph.number_of_nodes() == 4


def test_coocurrencia_solo_conecta_pedidos_con_dos_items(items):
    graph = build_cooccurrence(items, "product_id", "producto")

    assert graph.has_edge("p1", "p2")
    assert not graph.has_edge("p1", "p3")
    assert graph["p1"]["p2"]["weight"] == 1


def test_estadisticas_reportan_componentes(items):
    graph = build_cooccurrence(items, "product_id", "producto")
    stats = graph_stats(graph)

    assert stats["nodos"] == 2
    assert stats["aristas"] == 1
    assert stats["componentes"] == 1
    assert stats["componente_gigante_%"] == pytest.approx(100.0)


def test_comparacion_elige_un_ganador_con_cobertura_total(items):
    graph = build_bipartite(items, "categoria_en", "seller_id", "categoria", "vendedor")
    results, winner = compare_communities(graph)

    assert winner in results
    assert results[winner]["cobertura"] == 1.0
    assert set(results) == {"Louvain", "Greedy Modularity", "Label Propagation"}


@pytest.mark.skipif(not ITEMS.exists(), reason="falta data/processed/grafos/items_pedidos.csv")
@pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="faltan artefactos")
def test_manifest_cumple_campos_obligatorios():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    requeridos = {
        "status",
        "model_name",
        "version",
        "generated_at",
        "dataset",
        "period",
        "metrics",
        "winner",
        "results_path",
        "chart_data_path",
        "business_conclusions",
        "limitations",
    }

    assert requeridos <= set(manifest)
    assert manifest["status"] == "Completado"
    assert (ROOT / manifest["results_path"]).exists()
    assert (ROOT / manifest["chart_data_path"]).exists()
