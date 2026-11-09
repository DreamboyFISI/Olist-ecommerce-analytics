"""Prepara la tabla a nivel de ítem para el módulo de grafos.

    python -m src.data.build_graphs_data

A diferencia de ``build_master.py``, esta tabla NO agrega por pedido: conserva
``product_id`` y ``seller_id`` para poder construir las aristas del grafo.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "grafos"

FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_tr": "product_category_name_translation.csv",
}

DATE_COLS = {
    "orders": ["order_purchase_timestamp"],
}

SIN_CATEGORIA = "sin_categoria"

COLUMNS = [
    "order_id",
    "order_item_id",
    "product_id",
    "seller_id",
    "price",
    "freight_value",
    "order_status",
    "order_purchase_timestamp",
    "customer_state",
    "categoria_pt",
    "categoria_en",
]


def load_raw(raw_dir: Path = RAW) -> dict[str, pd.DataFrame]:
    missing = [f for f in FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan CSV en {raw_dir}: {missing}. Descárgalos con kagglehub o con el "
            "repositorio alterno indicado en el README."
        )

    string_cols = {
        "orders": ["order_id", "customer_id"],
        "customers": ["customer_id"],
        "items": ["order_id", "product_id", "seller_id"],
        "products": ["product_id", "product_category_name"],
        "category_tr": ["product_category_name", "product_category_name_english"],
    }
    dfs = {
        key: pd.read_csv(
            raw_dir / filename, dtype={col: "string" for col in string_cols[key]}
        )
        for key, filename in FILES.items()
    }
    for table, cols in DATE_COLS.items():
        for col in cols:
            dfs[table][col] = pd.to_datetime(dfs[table][col], errors="coerce")
    return dfs


def build_items(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Une ítems con pedido, cliente, producto y traducción de categoría."""
    orders = dfs["orders"][
        ["order_id", "customer_id", "order_status", "order_purchase_timestamp"]
    ]
    customers = dfs["customers"][["customer_id", "customer_state"]]
    products = dfs["products"][["product_id", "product_category_name"]]
    translation = dfs["category_tr"]

    items = dfs["items"].copy()
    duplicated = int(items.duplicated(subset=["order_id", "order_item_id"]).sum())
    if duplicated:
        items = items.drop_duplicates(subset=["order_id", "order_item_id"], keep="last")

    items["price"] = pd.to_numeric(items["price"], errors="coerce")
    items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce")

    df = (
        items.merge(orders, on="order_id", how="left")
        .merge(customers, on="customer_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(translation, on="product_category_name", how="left")
    )

    # No se descartan productos sin categoría: se etiquetan para conservar sus aristas.
    df["categoria_pt"] = df["product_category_name"].fillna(SIN_CATEGORIA)
    df["categoria_en"] = df["product_category_name_english"].fillna(df["categoria_pt"])

    return df[COLUMNS].reset_index(drop=True), duplicated


def quality_report(df: pd.DataFrame, duplicated: int) -> dict:
    return {
        "filas": int(len(df)),
        "pedidos": int(df.order_id.nunique()),
        "productos": int(df.product_id.nunique()),
        "vendedores": int(df.seller_id.nunique()),
        "categorias": int(df.categoria_en.nunique()),
        "duplicados_eliminados": duplicated,
        "items_sin_categoria": int((df.categoria_pt == SIN_CATEGORIA).sum()),
        "productos_sin_categoria": int(
            df.loc[df.categoria_pt == SIN_CATEGORIA, "product_id"].nunique()
        ),
        "pedidos_por_estado": {
            str(k): int(v) for k, v in df.groupby("order_status").order_id.nunique().items()
        },
        "fechas_invalidas": int(df.order_purchase_timestamp.isna().sum()),
        "nulos_%": df.isna().mean().mul(100).round(2).loc[lambda s: s > 0].to_dict(),
        "periodo": [
            str(df.order_purchase_timestamp.min().date()),
            str(df.order_purchase_timestamp.max().date()),
        ],
    }


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    dfs = load_raw()
    df, duplicated = build_items(dfs)
    report = quality_report(df, duplicated)

    df.to_csv(PROCESSED / "items_pedidos.csv", index=False)
    with (PROCESSED / "reporte_preparacion.json").open("w", encoding="utf-8") as salida:
        json.dump(report, salida, ensure_ascii=False, indent=2)

    print(f"Tabla de ítems: {PROCESSED / 'items_pedidos.csv'}")
    print(f"{report['filas']:,} filas | {report['pedidos']:,} pedidos | "
          f"{report['productos']:,} productos | {report['vendedores']:,} vendedores")
    print(f"Periodo: {report['periodo'][0]} a {report['periodo'][1]}")
    print(f"Ítems sin categoría: {report['items_sin_categoria']:,} "
          f"({report['productos_sin_categoria']:,} productos)")


if __name__ == "__main__":
    main()
