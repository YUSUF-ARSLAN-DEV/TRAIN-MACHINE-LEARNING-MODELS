"""Single source of truth for which models exist, their status, and how the
frontend should present each one. The frontend fetches this from GET /api/models
instead of hardcoding it, so adding a 6th project later means editing this list only.
"""

MODELS = [
    {
        "id": "bert-intent",
        "name": "Customer Support Intent Classifier",
        "shortName": "BERT",
        "task": "text-classification",
        "status": "ready",
        "stack": ["BERT", "Hugging Face Transformers"],
        "description": (
            "Fine-tuned bert-base-uncased on the Bitext customer support dataset. "
            "Type a customer message and it predicts one of 27 support intents "
            "(refunds, cancellations, account changes, etc.) with a confidence score."
        ),
        "inputType": "text",
        "endpoint": "/api/bert/predict",
    },
    {
        "id": "resnet18-eurosat",
        "name": "Satellite Land-Use Classifier",
        "shortName": "ResNet-18",
        "task": "image-classification",
        "status": "ready",
        "stack": ["ResNet-18", "PyTorch", "torchvision"],
        "description": (
            "Transfer-learned ResNet-18 (frozen ImageNet backbone, fine-tuned final layer) "
            "on the EuroSAT Sentinel-2 dataset. Upload a satellite image tile and it "
            "predicts the land-use class (forest, river, residential, industrial, etc.)."
        ),
        "inputType": "image",
        "endpoint": "/api/resnet/predict",
    },
    {
        "id": "yolo-detection",
        "name": "Object Detector",
        "shortName": "YOLO",
        "task": "object-detection",
        "status": "planned",
        "stack": ["YOLO", "Ultralytics"],
        "description": (
            "Detection fine-tuning with YOLO — will locate and label multiple objects "
            "within an uploaded image, drawing bounding boxes with class + confidence."
        ),
        "inputType": "image",
        "endpoint": None,
    },
    {
        "id": "llm-lora",
        "name": "LLM Adapter",
        "shortName": "LoRA",
        "task": "text-generation",
        "status": "planned",
        "stack": ["LoRA", "PEFT"],
        "description": (
            "Parameter-efficient fine-tuning (LoRA/PEFT) of a base LLM for a "
            "domain-specific task — will take a prompt and return the adapted model's response."
        ),
        "inputType": "text",
        "endpoint": None,
    },
    {
        "id": "multitask-serving",
        "name": "Multi-Task Model",
        "shortName": "Multi-Task",
        "task": "multi-task",
        "status": "planned",
        "stack": ["Multi-task fine-tuning", "Serving pipeline"],
        "description": (
            "A single model fine-tuned across multiple tasks at once, served behind a "
            "unified inference pipeline — the capstone project tying the track together."
        ),
        "inputType": "text",
        "endpoint": None,
    },
]


def get_models():
    return MODELS


def get_model(model_id: str):
    for m in MODELS:
        if m["id"] == model_id:
            return m
    return None
