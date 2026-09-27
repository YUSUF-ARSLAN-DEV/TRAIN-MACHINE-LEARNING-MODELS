# Showcase Web App

A FastAPI backend + React frontend that lets someone try each fine-tuned model
from the [Progress Tracker](../README.md) in a browser: one page per model,
one card per model on the home screen. Which models are wired up (vs. still
"coming soon") is driven entirely by `backend/app/catalog.py` — add a project
there once it's trained and it gets its own nav entry and page automatically.

## Status

| Model | Page | Backend |
|---|---|---|
| BERT intent classifier | live | loads `BERT/bert_intent_model` and serves real predictions |
| ResNet-18 / EuroSAT | live UI, backend returns 503 | `SOURCE_CODE/Resnet18.py` never calls `torch.save`, so there are no weights yet. The service is already wired to `SOURCE_CODE/resnet18_eurosat.pth` — once that file exists, this works with no further changes. |
| YOLO / LoRA-PEFT / multi-task | "coming soon" placeholder | not built yet |

## Run it

**Backend** (from `webapp/backend`, using the repo's existing `venv`):

```bash
../../venv/Scripts/pip install -r requirements.txt   # first time only
../../venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

**Frontend** (from `webapp/frontend`):

```bash
npm install   # first time only
npm run dev
```

Then open http://localhost:5173. The frontend expects the API at
`http://localhost:8000`; override with a `VITE_API_BASE` env var if needed.
