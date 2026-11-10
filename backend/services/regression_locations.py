from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "regression"
CATALOG_PATHS = {
    "customer": ARTIFACT_DIR / "customer_locations.csv",
    "seller": ARTIFACT_DIR / "seller_locations.csv",
}


@lru_cache(maxsize=2)
def _load_catalog(role: str) -> pd.DataFrame:
    if role not in CATALOG_PATHS:
        raise ValueError("El tipo de ubicación debe ser 'customer' o 'seller'.")
    return pd.read_csv(CATALOG_PATHS[role])


def _filtered(role: str, state: str, city: str | None = None) -> pd.DataFrame:
    catalog = _load_catalog(role)
    filtered = catalog.loc[catalog["state"].eq(state.strip().upper())]
    if city:
        filtered = filtered.loc[filtered["city"].eq(city)]
    return filtered


def list_cities(role: str, state: str) -> list[str]:
    return sorted(_filtered(role, state)["city"].dropna().unique().tolist())


def list_locations(role: str, state: str, city: str) -> list[dict]:
    filtered = _filtered(role, state, city).copy()
    identifier = "customer_id" if role == "customer" else "seller_id"
    filtered = filtered.sort_values(["zip_prefix", identifier])
    records = []
    for row in filtered.itertuples(index=False):
        short_id = getattr(row, identifier)[-8:]
        records.append({
            "id": getattr(row, identifier),
            "label": f"CEP {int(row.zip_prefix):05d} · ID …{short_id}",
            "zip_prefix": int(row.zip_prefix),
            "city": row.city,
            "state": row.state,
            "lat": round(float(row.lat), 6),
            "lng": round(float(row.lng), 6),
        })
    return records
