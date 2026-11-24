from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.classification import explain_prediction, load_classification_results, predict_risk
from backend.services.monitoring import load_monitoring_report
from backend.services.repository import load_eda_summary, load_module_registry
from backend.services.regression import load_regression_results, predict_freight
from backend.services.regression_locations import list_cities, list_locations, resolve_state
from backend.services.sentiment import load_sentiment_results, predict_sentiment

router = APIRouter()


class RegressionPredictionRequest(BaseModel):
    purchase_timestamp: datetime
    n_items: int = Field(ge=1, le=100)
    n_products: int = Field(ge=1, le=100)
    n_sellers: int = Field(ge=1, le=20)
    price_total: float = Field(ge=0, le=100_000)
    weight_total_g: float = Field(ge=0, le=1_000_000)
    volume_total_cm3: float = Field(ge=0, le=100_000_000)
    distance_km: float = Field(ge=0, le=10_000)
    seller_id: str = Field(min_length=1, max_length=64)
    seller_lat: float = Field(ge=-35, le=6)
    seller_lng: float = Field(ge=-75, le=-30)
    customer_lat: float = Field(ge=-35, le=6)
    customer_lng: float = Field(ge=-75, le=-30)
    customer_state: str = Field(min_length=2, max_length=2)
    main_seller_state: str = Field(min_length=2, max_length=2)
    main_category: str = Field(min_length=1, max_length=120)


class SentimentPredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5_000, description="Reseña en portugués brasileño.")


class ClassificationPredictionRequest(BaseModel):
    """Un pedido tal como se observa al enviar la encuesta de satisfacción."""

    purchase_timestamp: datetime
    survey_timestamp: datetime | None = Field(default=None, description="Envío de la encuesta; por defecto compra + 20 días.")
    estimated_delivery_date: datetime | None = Field(default=None, description="Fecha prometida al cliente.")
    shipping_limit_date: datetime | None = Field(default=None, description="Fecha límite para despachar.")
    delivered_carrier_date: datetime | None = Field(default=None, description="Entrega al transportista, si ocurrió.")
    delivered_customer_date: datetime | None = Field(default=None, description="Entrega al cliente, si ocurrió.")
    approval_hours: float = Field(default=0.5, ge=0, le=24 * 30)
    n_items: int = Field(default=1, ge=1, le=100)
    n_products: int = Field(default=1, ge=1, le=100)
    n_sellers: int = Field(default=1, ge=1, le=20)
    n_categories: int = Field(default=1, ge=1, le=20)
    price_total: float = Field(ge=0, le=100_000)
    freight_total: float = Field(default=0, ge=0, le=10_000)
    weight_total_g: float | None = Field(default=None, ge=0, le=1_000_000)
    volume_total_cm3: float | None = Field(default=None, ge=0, le=100_000_000)
    photos_mean: float | None = Field(default=None, ge=0, le=50)
    description_length_mean: float | None = Field(default=None, ge=0, le=10_000)
    installments: int = Field(default=1, ge=1, le=36)
    n_payment_types: int = Field(default=1, ge=1, le=5)
    payment_type: str = Field(default="credit_card", min_length=1, max_length=40)
    customer_state: str = Field(min_length=2, max_length=2)
    seller_state: str = Field(min_length=2, max_length=2)
    distance_km: float | None = Field(default=None, ge=0, le=10_000)
    main_category: str = Field(default="sin_categoria", min_length=1, max_length=120)



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


@router.get("/api/v1/regression/locations/resolve-state", tags=["regression"])
def regression_location_state(
    lat: float = Query(ge=-35, le=6),
    lng: float = Query(ge=-75, le=-30),
) -> dict:
    try:
        return resolve_state(lat, lng)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No se encontró el catálogo de ubicaciones.") from exc


@router.post("/api/v1/regression/predict", tags=["regression"])
def regression_predict(request: RegressionPredictionRequest) -> dict:
    try:
        return predict_freight(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="El modelo de regresión no está disponible.") from exc
    except (AttributeError, ImportError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo ejecutar la predicción: {exc}") from exc


@router.get("/api/v1/classification/results", tags=["classification"])
def classification_results() -> dict:
    try:
        return load_classification_results()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No se encontraron los artefactos de clasificación.") from exc


@router.post("/api/v1/classification/predict", tags=["classification"])
def classification_predict(request: ClassificationPredictionRequest) -> dict:
    payload = request.model_dump()
    payload = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in payload.items()}
    try:
        return predict_risk(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="El modelo de clasificación no está disponible.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (AttributeError, ImportError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo ejecutar la predicción: {exc}") from exc


@router.post("/api/v1/classification/explain", tags=["classification"])
def classification_explain(request: ClassificationPredictionRequest,
                           method: str = Query(default="shap", pattern="^(shap|lime|perturbacion)$")) -> dict:
    payload = request.model_dump()
    payload = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in payload.items()}
    try:
        return explain_prediction(payload, prefer=method)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="El modelo de clasificación no está disponible.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (AttributeError, ImportError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo explicar la predicción: {exc}") from exc


@router.get("/api/v1/monitoring/report", tags=["monitoring"])
def monitoring_report(refresh: bool = Query(default=False, description="Recalcular el reporte y registrar la corrida.")) -> dict:
    try:
        return load_monitoring_report(refresh=refresh)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No hay reporte de monitoreo; ejecuta python -m src.monitoring.") from exc


@router.get("/api/v1/nlp/results", tags=["nlp"])
def nlp_results() -> dict:
    try:
        return load_sentiment_results()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No se encontraron los artefactos de sentimientos.") from exc


@router.post("/api/v1/nlp/predict", tags=["nlp"])
def nlp_predict(request: SentimentPredictionRequest) -> dict:
    try:
        return predict_sentiment(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="El modelo de sentimientos no está disponible.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (AttributeError, ImportError) as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo ejecutar la predicción: {exc}") from exc
