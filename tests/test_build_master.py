"""Pruebas de la tabla maestra (fase 3 de CRISP-DM).

Se omiten si `data/raw/` no tiene los CSV, para que el repo siga siendo clonable
sin el dataset.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.build_master import FILES, RAW, build_master, load_raw, quality_report

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not all((RAW / f).exists() for f in FILES.values()),
    reason="data/raw/ sin los CSV de Olist",
)


@pytest.fixture(scope="module")
def master():
    return build_master(load_raw())


@pytest.fixture(scope="module")
def hallazgos():
    with (ROOT / "docs" / "hallazgos_eda.json").open(encoding="utf-8") as f:
        return json.load(f)


def test_una_fila_por_pedido(master):
    assert len(master) == 99441
    assert not master.order_id.duplicated().any()


def test_columnas_derivadas_presentes(master):
    esperadas = {"dias_entrega", "dias_retraso", "entrega_tardia", "ratio_flete",
                 "ticket", "review_negativa", "tiene_comentario", "registro_valido"}
    assert esperadas <= set(master.columns)


def test_banderas_son_binarias(master):
    for col in [c for c in master.columns if c.startswith("flag_")]:
        assert set(master[col].unique()) <= {0, 1}, col


def test_no_hay_entregas_antes_de_la_compra(master):
    entregados = master.dropna(subset=["dias_entrega"])
    assert (entregados.dias_entrega >= 0).all()


def test_peso_nulo_no_se_convierte_en_cero(master):
    """Un pedido sin peso conocido queda nulo; solo los pedidos sin ítems lo están."""
    assert master.peso_total_g.isna().sum() == master.n_items.isna().sum()
    con_items = master[master.n_items.notna()]
    assert con_items.peso_total_g.notna().all()


def test_reproduce_cifras_del_primer_avance(master, hallazgos):
    entregados = master[master.order_status == "delivered"].dropna(
        subset=["dias_entrega", "review_score"])

    assert master.review_negativa.mean() * 100 == pytest.approx(
        hallazgos["review_negativa_%(<=2)"], abs=0.01)
    assert entregados.entrega_tardia.mean() * 100 == pytest.approx(
        hallazgos["%_entregas_tardias"], abs=0.01)
    assert (entregados.dias_estimados - entregados.dias_entrega).abs().mean() == pytest.approx(
        hallazgos["MAE_estimacion_olist_dias"], abs=0.01)
    assert master.loc[master.order_status != "canceled", "payment_value"].mean() == pytest.approx(
        hallazgos["ticket_promedio_BRL"], abs=0.01)


def test_reporte_de_calidad(master):
    reporte = quality_report(master)
    assert reporte["filas"] == len(master)
    assert reporte["registros_validos"] + reporte["registros_marcados"] == len(master)
    assert reporte["banderas"]["flag_sin_items"] == 775
