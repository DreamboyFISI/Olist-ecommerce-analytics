"""Serie temporal de ventas diarias de Olist.

Construye la serie, define la validación con origen móvil y las métricas de
error que comparan los modelos del módulo.

Rango: 2017-01-23 a 2018-08-22. Fuera de ese rango la serie no es representativa:

- 2016 y las tres primeras semanas de 2017 son el lanzamiento de la plataforma
  (noviembre de 2016 no existe y la media móvil se cuadruplica en dos semanas).
- Desde el 23 de agosto de 2018 las ventas se desangran de forma monótona hasta
  el 7% de la mediana: es el corte de extracción, no una caída real.

Con este recorte el salto entre el primer y el último mes baja de ×9.6 a ×4.0,
que sí es crecimiento orgánico.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "master_pedidos.csv"

INICIO = "2017-01-23"
FIN = "2018-08-22"
ESTACIONALIDAD = 7


def load_master(path: Path = MASTER) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Genérala con: python -m src.data.build_master"
        )
    return pd.read_csv(path, parse_dates=["order_purchase_timestamp"], low_memory=False)


def build_series(df: pd.DataFrame | None = None, objetivo: str = "ventas",
                 inicio: str = INICIO, fin: str = FIN) -> pd.Series:
    """Serie diaria de ventas (BRL) o de pedidos, ya recortada."""
    if df is None:
        df = load_master()

    v = df[(df.order_status != "canceled") & (df.registro_valido == 1)]
    dia = v.order_purchase_timestamp.dt.normalize()
    serie = (v.groupby(dia).payment_value.sum() if objetivo == "ventas"
             else v.groupby(dia).order_id.count())

    serie = serie.loc[inicio:fin].asfreq("D")
    if serie.isna().any():
        faltan = serie[serie.isna()].index.date.tolist()
        raise ValueError(f"La serie tiene huecos en {faltan}; revisa el rango.")
    serie.name = objetivo
    return serie


def calendario(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Regresores exógenos: Black Friday y feriados brasileños relevantes."""
    exog = pd.DataFrame(index=index)
    exog["black_friday"] = ((index >= "2017-11-23") & (index <= "2017-11-27")).astype(int)
    exog["navidad"] = ((index.month == 12) & (index.day >= 15)).astype(int)
    exog["carnaval"] = (((index >= "2017-02-25") & (index <= "2017-03-01")) |
                        ((index >= "2018-02-10") & (index <= "2018-02-14"))).astype(int)
    return exog


def walk_forward_splits(serie: pd.Series, horizonte: int = 30, n_pliegues: int = 6):
    """Validación con origen móvil: cada pliegue entrena solo con el pasado.

    Devuelve pares (train, test). Nunca se baraja el orden: hacerlo filtraría
    información del futuro hacia el pasado.
    """
    total = len(serie)
    minimo = total - horizonte * n_pliegues
    if minimo < horizonte * 2:
        raise ValueError("Serie demasiado corta para ese horizonte y número de pliegues.")

    for i in range(n_pliegues):
        corte = minimo + i * horizonte
        yield serie.iloc[:corte], serie.iloc[corte:corte + horizonte]


def smape(real: np.ndarray, pred: np.ndarray) -> float:
    """Error porcentual simétrico: no explota cuando el real se acerca a cero."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    denom = (np.abs(real) + np.abs(pred)) / 2
    return float(np.mean(np.abs(real - pred) / np.where(denom == 0, np.nan, denom)) * 100)


def metricas(real, pred) -> dict:
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    error = real - pred
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE": float(np.mean(np.abs(error / np.where(real == 0, np.nan, real))) * 100),
        "sMAPE": smape(real, pred),
    }


def naive_estacional(train: pd.Series, horizonte: int, s: int = ESTACIONALIDAD) -> pd.Series:
    """Baseline obligatorio: repite el mismo día de la semana anterior.

    Si un modelo no le gana a esto, no hay modelo.
    """
    ultimos = train.iloc[-s:].to_numpy()
    pred = [ultimos[i % s] for i in range(horizonte)]
    index = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq="D")
    return pd.Series(pred, index=index, name="naive_estacional")


def evaluar(serie: pd.Series, predictor, horizonte: int = 30, n_pliegues: int = 6) -> pd.DataFrame:
    """Aplica `predictor(train, horizonte) -> Series` sobre cada pliegue."""
    filas = []
    for i, (train, test) in enumerate(walk_forward_splits(serie, horizonte, n_pliegues), 1):
        pred = predictor(train, horizonte)
        fila = {"pliegue": i, "desde": test.index[0].date(), "hasta": test.index[-1].date()}
        fila.update(metricas(test.to_numpy(), np.asarray(pred)))
        filas.append(fila)
    return pd.DataFrame(filas)


ORDENES = [((1, 1, 1), (1, 1, 1, 7)), ((2, 1, 1), (1, 1, 1, 7)),
           ((1, 1, 2), (1, 1, 1, 7)), ((0, 1, 1), (0, 1, 1, 7))]


def _ajustar_sarimax(y, order, seasonal, exog=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    return SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False, maxiter=200)


def _mejor_orden(y, exog=None):
    """Elige el orden por AIC sobre el train del pliegue, sin mirar el test."""
    mejor, mejor_aic = ORDENES[0], np.inf
    for order, seasonal in ORDENES:
        try:
            aic = _ajustar_sarimax(y, order, seasonal, exog).aic
        except Exception:
            continue
        if aic < mejor_aic:
            mejor, mejor_aic = (order, seasonal), aic
    return mejor


def sarima(train: pd.Series, horizonte: int) -> pd.Series:
    y = np.log1p(train)
    order, seasonal = _mejor_orden(y)
    pred = _ajustar_sarimax(y, order, seasonal).forecast(horizonte)
    return pd.Series(np.expm1(pred.to_numpy()), index=_futuro(train, horizonte), name="sarima")


def sarimax(train: pd.Series, horizonte: int) -> pd.Series:
    y = np.log1p(train)
    index = _futuro(train, horizonte)
    exog_tr, exog_te = calendario(train.index), calendario(index)
    order, seasonal = _mejor_orden(y, exog_tr)
    pred = _ajustar_sarimax(y, order, seasonal, exog_tr).forecast(horizonte, exog=exog_te)
    return pd.Series(np.expm1(np.asarray(pred)), index=index, name="sarimax")


def prophet(train: pd.Series, horizonte: int) -> pd.Series:
    from prophet import Prophet

    modelo = Prophet(weekly_seasonality=True, yearly_seasonality=False,
                     daily_seasonality=False, seasonality_mode="multiplicative")
    modelo.add_country_holidays(country_name="BR")
    modelo.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))

    index = _futuro(train, horizonte)
    pred = modelo.predict(pd.DataFrame({"ds": index}))["yhat"].to_numpy()
    return pd.Series(np.clip(pred, 0, None), index=index, name="prophet")


def _futuro(train: pd.Series, horizonte: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq="D")


MODELOS = {"naive_estacional": naive_estacional, "sarima": sarima,
           "sarimax": sarimax, "prophet": prophet}


def comparar(serie: pd.Series, horizonte: int = 30, n_pliegues: int = 6) -> pd.DataFrame:
    filas = []
    for nombre, predictor in MODELOS.items():
        r = evaluar(serie, predictor, horizonte, n_pliegues).mean(numeric_only=True)
        filas.append({"modelo": nombre, **r.drop("pliegue").to_dict()})
    tabla = pd.DataFrame(filas).sort_values("sMAPE").reset_index(drop=True)
    base = tabla.loc[tabla.modelo == "naive_estacional", "MAE"].iat[0]
    tabla["mejora_MAE_%"] = ((base - tabla.MAE) / base * 100).round(1)
    return tabla
