from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.repository import load_eda_summary, load_module_registry
from backend.services.regression import load_regression_results, predict_delivery
from backend.services.regression_locations import list_cities, list_locations

router = APIRouter()


class RegressionPredictionRequest(BaseModel):
    purchase_timestamp: datetime
    estimated_delivery_date: date
    n_items: int = Field(ge=1, le=100)
    n_products: int = Field(ge=1, le=100)
    n_sellers: int = Field(ge=1, le=20)
    price_total: float = Field(ge=0, le=100_000)
    freight_total: float = Field(ge=0, le=20_000)
    weight_total_g: float = Field(ge=0, le=1_000_000)
    volume_total_cm3: float = Field(ge=0, le=100_000_000)
    distance_km: float = Field(ge=0, le=10_000)
    customer_state: str = Field(min_length=2, max_length=2)
    main_seller_state: str = Field(min_length=2, max_length=2)
    main_category: str = Field(min_length=1, max_length=120)


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "datasmart-olist-api", "version": "0.1.0"}


@router.get("/api/v1/eda/summary", tags=["eda"])
def eda_summary() -> dict:
    try:
        return load_eda_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No se encontró el resumen del EDA.") from exc


@router.get("/api/v1/modules", tags=["modules"])
def modules() -> list[dict]:
    return load_module_registry()


@router.get("/api/v1/regression/results", tags=["regression"])
def regression_results() -> dict:
    try:
        return load_regression_results()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No se encontraron los artefactos de regresión.") from exc


@router.get("/api/v1/regression/locations/cities", tags=["regression"])
def regression_location_cities(
    role: str = Query(pattern="^(customer|seller)$"),
    state: str = Query(min_length=2, max_length=2),
) -> list[str]:
    try:
        return list_cities(role, state)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No se encontró el catálogo de ubicaciones.") from exc


@router.get("/api/v1/regression/locations", tags=["regression"])
def regression_locations(
    role: str = Query(pattern="^(customer|seller)$"),
    state: str = Query(min_length=2, max_length=2),
    city: str = Query(min_length=1, max_length=120),
) -> list[dict]:
    try:
        return list_locations(role, state, city)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No se encontró el catálogo de ubicaciones.") from exc


@router.post("/api/v1/regression/predict", tags=["regression"])
def regression_predict(request: RegressionPredictionRequest) -> dict:
    try:
        return predict_delivery(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="El modelo de regresión no está disponible.") from exc
    except (AttributeError, ImportError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo ejecutar la predicción: {exc}") from exc
