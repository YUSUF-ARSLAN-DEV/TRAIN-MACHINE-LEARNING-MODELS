from fastapi import APIRouter, HTTPException

from app.catalog import get_model, get_models

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models():
    return get_models()


@router.get("/{model_id}")
def read_model(model_id: str):
    model = get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Unknown model id")
    return model
