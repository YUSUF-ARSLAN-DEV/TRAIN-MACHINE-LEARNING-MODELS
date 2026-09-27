import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.services.resnet_service import get_service

router = APIRouter(prefix="/api/resnet", tags=["resnet"])


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not read image file") from exc

    try:
        service = get_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    predictions = service.predict(image)
    top = predictions[0]
    return {
        "label": top["label"],
        "confidence": top["score"],
        "top_k": predictions,
    }
