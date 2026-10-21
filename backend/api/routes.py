from fastapi import APIRouter, HTTPException

from backend.services.repository import load_eda_summary, load_module_registry

router = APIRouter()


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
