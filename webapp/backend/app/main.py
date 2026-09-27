from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import bert, models, resnet

app = FastAPI(title="ML Models Showcase API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(bert.router)
app.include_router(resnet.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
