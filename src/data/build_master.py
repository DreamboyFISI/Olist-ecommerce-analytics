"""Tabla maestra de pedidos de Olist — fase 3 de CRISP-DM.

Integra los nueve CSV crudos en una tabla a nivel `order_id` aplicando la matriz
de tratamiento del 1° avance.

Las inconsistencias se marcan con banderas `flag_*` en lugar de eliminarse: cada
módulo decide qué excluir según su problema.

    python -m src.data.build_master
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_tr": "product_category_name_translation.csv",
}

DATE_COLS = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

SEGUNDOS_POR_DIA = 86400


def load_raw(raw_dir: Path = RAW) -> dict[str, pd.DataFrame]:
    missing = [f for f in FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan CSV en {raw_dir}: {missing}. Descárgalos con kagglehub o con el "
            "repositorio alterno indicado en el README."
        )
    dfs = {k: pd.read_csv(raw_dir / f, encoding="utf-8-sig") for k, f in FILES.items()}
    for table, cols in DATE_COLS.items():
        for col in cols:
            dfs[table][col] = pd.to_datetime(dfs[table][col], errors="coerce")
    return dfs


def dedup_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Una reseña por pedido, la respuesta más reciente (814 ids duplicados)."""
    return (reviews.sort_values("review_answer_timestamp")
                   .drop_duplicates("order_id", keep="last"))


def dedup_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    """Centroide por prefijo postal (261,831 filas exactamente duplicadas)."""
    return (geo.drop_duplicates()
               .groupby("geolocation_zip_code_prefix")
               .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"),
                    estado=("geolocation_state", lambda s: s.mode().iat[0]))
               .reset_index())


def build_items(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prod = dfs["products"].merge(dfs["category_tr"], on="product_category_name", how="left")
    prod["category"] = (prod["product_category_name_english"]
                        .fillna(prod["product_category_name"])
                        .fillna("sin_categoria"))
    prod["product_weight_g"] = prod.product_weight_g.fillna(
        prod.groupby("category").product_weight_g.transform("median"))

    items_full = (dfs["order_items"]
                  .merge(prod[["product_id", "category", "product_weight_g"]], on="product_id", how="left")
                  .merge(dfs["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left"))

    agg = items_full.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_productos=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_categorias=("category", "nunique"),
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        categoria_principal=("category", "first"),
        seller_state=("seller_state", "first"),
    )
    # min_count=1 evita que una suma de solo nulos se convierta en 0: un pedido
    # sin peso conocido debe quedar nulo, no pesar cero.
    agg["peso_total_g"] = items_full.groupby("order_id").product_weight_g.sum(min_count=1)
    return agg.reset_index()


def build_payments(payments: pd.DataFrame) -> pd.DataFrame:
    return payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        n_cuotas=("payment_installments", "max"),
        payment_type=("payment_type", lambda s: s.mode().iat[0]),
    ).reset_index()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas, con las mismas definiciones del notebook de EDA."""
    df = df.copy()

    df["dias_entrega"] = (df.order_delivered_customer_date - df.order_purchase_timestamp).dt.total_seconds() / SEGUNDOS_POR_DIA
    df["dias_estimados"] = (df.order_estimated_delivery_date - df.order_purchase_timestamp).dt.total_seconds() / SEGUNDOS_POR_DIA
    df["dias_retraso"] = (df.order_delivered_customer_date - df.order_estimated_delivery_date).dt.total_seconds() / SEGUNDOS_POR_DIA
    df["dias_aprobacion"] = (df.order_approved_at - df.order_purchase_timestamp).dt.total_seconds() / SEGUNDOS_POR_DIA

    df["entrega_tardia"] = (df.dias_retraso > 0).astype(int).where(df.order_delivered_customer_date.notna())
    df["mismo_estado"] = (df.customer_state == df.seller_state).astype(int)
    df["ratio_flete"] = df.freight_total / df.price_total
    df["ticket"] = df.price_total + df.freight_total

    df["tiene_comentario"] = df.review_comment_message.notna().astype(int)
    df["review_negativa"] = (df.review_score <= 2).astype(int).where(df.review_score.notna())

    ts = df.order_purchase_timestamp
    df["fecha"] = ts.dt.date
    df["mes"] = ts.dt.to_period("M").astype(str)
    df["anio"] = ts.dt.year
    df["dia_semana"] = ts.dt.dayofweek
    df["es_fin_semana"] = (df.dia_semana >= 5).astype(int)
    df["hora"] = ts.dt.hour
    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flag_sin_items"] = df.n_items.isna().astype(int)
    df["flag_sin_pago"] = df.payment_value.isna().astype(int)
    df["flag_pago_cero"] = (df.payment_value == 0).astype(int)
    df["flag_entregado_sin_fecha"] = ((df.order_status == "delivered") &
                                      df.order_delivered_customer_date.isna()).astype(int)
    df["flag_cancelado_con_entrega"] = ((df.order_status == "canceled") &
                                        df.order_delivered_customer_date.notna()).astype(int)
    df["flag_entrega_antes_transportista"] = (
        df.order_delivered_customer_date < df.order_delivered_carrier_date).astype(int)
    df["flag_transportista_antes_aprobacion"] = (
        df.order_delivered_carrier_date < df.order_approved_at).astype(int)
    df["flag_pago_descuadrado"] = ((df.payment_value - df.ticket).abs() > 1).astype(int)

    # El descuadre de pago y el adelanto del transportista quedan fuera: son
    # explicables por voucher y por aprobación tardía, no son errores.
    errores = ["flag_sin_items", "flag_pago_cero", "flag_entregado_sin_fecha",
               "flag_cancelado_con_entrega", "flag_entrega_antes_transportista"]
    df["registro_valido"] = (df[errores].sum(axis=1) == 0).astype(int)
    return df


def build_master(dfs: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    if dfs is None:
        dfs = load_raw()

    df = (dfs["orders"]
          .merge(dfs["customers"], on="customer_id", how="left")
          .merge(build_items(dfs), on="order_id", how="left")
          .merge(build_payments(dfs["payments"]), on="order_id", how="left")
          .merge(dedup_reviews(dfs["reviews"])[["order_id", "review_score", "review_comment_message"]],
                 on="order_id", how="left"))

    return add_quality_flags(add_features(df))


def quality_report(df: pd.DataFrame) -> dict:
    flags = [c for c in df.columns if c.startswith("flag_")]
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "periodo": [str(df.order_purchase_timestamp.min().date()),
                    str(df.order_purchase_timestamp.max().date())],
        "registros_validos": int(df.registro_valido.sum()),
        "registros_marcados": int((df.registro_valido == 0).sum()),
        "banderas": {f: int(df[f].sum()) for f in flags},
        "nulos_%": df.isna().mean().mul(100).round(2).loc[lambda s: s > 0].to_dict(),
    }


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_master()
    report = quality_report(df)

    df.to_csv(PROCESSED / "master_pedidos.csv", index=False)
    with (PROCESSED / "reporte_calidad.json").open("w", encoding="utf-8") as salida:
        json.dump(report, salida, ensure_ascii=False, indent=2)

    print(f"Tabla maestra: {PROCESSED / 'master_pedidos.csv'}")
    print(f"{report['filas']:,} filas × {report['columnas']} columnas")
    print(f"Periodo: {report['periodo'][0]} a {report['periodo'][1]}")
    print(f"Registros válidos: {report['registros_validos']:,} "
          f"({report['registros_validos'] / report['filas'] * 100:.2f}%)")


if __name__ == "__main__":
    main()
