"""Construye catálogos geográficos ligeros para el simulador de regresión.

Este script no entrena ni ejecuta el cuaderno. Solo cruza los catálogos públicos
de clientes y vendedores con la coordenada promedio de cada CEP de Olist.
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "artifacts" / "regression"


def _normalize_city(value: object, state: str) -> str:
    """Normaliza escritura sin inventar equivalencias entre municipios."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().strip()
    text = re.sub(r"[_.,;:]+", " ", text)
    text = re.sub(r"\s*[-/]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" /-")

    state_code = state.lower()
    suffix_patterns = (
        rf"\s+{re.escape(state_code)}$",
    )
    for pattern in suffix_patterns:
        text = re.sub(pattern, "", text).strip(" /-")
    return text


def main() -> None:
    geolocation = pd.read_csv(
        RAW_DIR / "olist_geolocation_dataset.csv",
        usecols=[
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
        ],
    )
    zip_coordinates = (
        geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
        .rename(columns={"geolocation_zip_code_prefix": "zip_prefix"})
    )

    specifications = {
        "customer": {
            "source": "olist_customers_dataset.csv",
            "columns": [
                "customer_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state",
            ],
            "rename": {
                "customer_zip_code_prefix": "zip_prefix",
                "customer_city": "city",
                "customer_state": "state",
            },
            "output": "customer_locations.csv",
        },
        "seller": {
            "source": "olist_sellers_dataset.csv",
            "columns": [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
            ],
            "rename": {
                "seller_zip_code_prefix": "zip_prefix",
                "seller_city": "city",
                "seller_state": "state",
            },
            "output": "seller_locations.csv",
        },
    }

    catalogs = {}
    for role, spec in specifications.items():
        catalog = pd.read_csv(RAW_DIR / spec["source"], usecols=spec["columns"])
        catalog = catalog.rename(columns=spec["rename"])
        catalog["state"] = catalog["state"].astype(str).str.upper()
        catalog["city"] = [
            _normalize_city(city, state)
            for city, state in zip(catalog["city"], catalog["state"])
        ]
        catalogs[role] = catalog

    # Un CEP pertenece normalmente a una ciudad. Para absorber errores como
    # "sao paulo sp" o "sao paulop", elegimos el nombre más frecuente para
    # cada combinación estado + CEP usando clientes y vendedores en conjunto.
    combined = pd.concat(
        [catalog[["state", "zip_prefix", "city"]] for catalog in catalogs.values()],
        ignore_index=True,
    )
    canonical_city = (
        combined.groupby(["state", "zip_prefix"])["city"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("canonical_city")
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for role, spec in specifications.items():
        catalog = catalogs[role].merge(
            canonical_city,
            on=["state", "zip_prefix"],
            how="left",
            validate="many_to_one",
        )
        catalog["city"] = catalog["canonical_city"].fillna(catalog["city"])
        catalog = catalog.drop(columns="canonical_city").merge(
            zip_coordinates, on="zip_prefix", how="inner", validate="many_to_one"
        )
        catalog = catalog.dropna(subset=["lat", "lng", "state", "city"])
        catalog.to_csv(OUTPUT_DIR / spec["output"], index=False)
        print(f"{role}: {len(catalog):,} ubicaciones")


if __name__ == "__main__":
    main()
