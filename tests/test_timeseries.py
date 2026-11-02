"""Pruebas de la serie temporal de ventas."""

import numpy as np
import pandas as pd
import pytest

from src.models.timeseries import (FIN, INICIO, MASTER, build_series, metricas,
                                   naive_estacional, smape, walk_forward_splits)

pytestmark = pytest.mark.skipif(
    not MASTER.exists(),
    reason="falta data/processed/master_pedidos.csv (python -m src.data.build_master)",
)


@pytest.fixture(scope="module")
def serie():
    return build_series()


def test_serie_diaria_sin_huecos(serie):
    assert serie.index.freqstr == "D"
    assert not serie.isna().any()
    assert len(serie) == (pd.Timestamp(FIN) - pd.Timestamp(INICIO)).days + 1


def test_rango_recortado(serie):
    assert serie.index[0] == pd.Timestamp(INICIO)
    assert serie.index[-1] == pd.Timestamp(FIN)
    # El arranque y la cola contaminados quedan fuera: ningún día baja del 10%
    # de la mediana, frente al 2.9% del rango sin recortar.
    assert serie.min() / serie.median() > 0.10


def test_los_pliegues_no_miran_el_futuro(serie):
    for train, test in walk_forward_splits(serie):
        assert train.index[-1] < test.index[0]
        assert len(test) == 30


def test_los_pliegues_avanzan_y_no_se_solapan(serie):
    tests = [test for _, test in walk_forward_splits(serie)]
    for anterior, siguiente in zip(tests, tests[1:]):
        assert anterior.index[-1] < siguiente.index[0]


def test_naive_estacional_repite_la_semana(serie):
    train = serie.iloc[:-30]
    pred = naive_estacional(train, 30)
    assert len(pred) == 30
    assert pred.iloc[0] == train.iloc[-7]
    assert pred.iloc[7] == train.iloc[-7]


def test_metricas_perfectas_son_cero():
    real = np.array([10.0, 20.0, 30.0])
    assert metricas(real, real)["MAE"] == 0
    assert smape(real, real) == 0


def test_smape_no_explota_cerca_de_cero():
    assert np.isfinite(smape(np.array([0.01, 100.0]), np.array([50.0, 100.0])))


def test_sarima_le_gana_al_baseline(serie):
    """El modelo ganador debe superar al naive estacional; si no, no hay modelo."""
    from src.models.timeseries import evaluar, naive_estacional, sarima

    corto = serie.iloc[-200:]
    base = evaluar(corto, naive_estacional, horizonte=30, n_pliegues=2).MAE.mean()
    modelo = evaluar(corto, sarima, horizonte=30, n_pliegues=2).MAE.mean()
    assert modelo < base


def test_los_predictores_devuelven_el_horizonte_pedido(serie):
    from src.models.timeseries import sarima, sarimax

    train = serie.iloc[:-30]
    for predictor in (sarima, sarimax):
        pred = predictor(train, 30)
        assert len(pred) == 30
        assert pred.index[0] == train.index[-1] + pd.Timedelta(days=1)
        assert (pred > 0).all()
