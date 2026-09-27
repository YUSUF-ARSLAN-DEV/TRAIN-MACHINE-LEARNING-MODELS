from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.bert_service import get_service

router = APIRouter(prefix="/api/bert", tags=["bert"])


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


@router.post("/predict")
def predict(payload: PredictRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        service = get_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    predictions = service.predict(text)
    top = predictions[0]
    return {
        "intent": top["label"],
        "confidence": top["score"],
        "top_k": predictions,
    }
