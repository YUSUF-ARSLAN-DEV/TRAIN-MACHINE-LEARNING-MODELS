"""Serves the EuroSAT ResNet-18 classifier.

Weights: RESNET/src/resnet18_eurosat.pth (saved by RESNET/src/Resnet18.py).
If the file is missing, FileNotFoundError is raised and the router returns a 503
that the UI shows as "model not trained yet".
"""

from pathlib import Path
from functools import lru_cache

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_PATH = REPO_ROOT / "RESNET" /"src"/ "resnet18_eurosat.pth"

# Alphabetical order, matching torchvision.datasets.EuroSAT's ImageFolder class order
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


class ResNetEuroSatService:
    def __init__(self, weights_path: Path):
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No trained ResNet-18 weights found at {weights_path}. "
                "Run training and save the model to this path to enable this model."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 10)
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5):
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)[0]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.shape[-1]))

        return [
            {"label": CLASS_NAMES[int(idx)], "score": float(score)}
            for score, idx in zip(top.values.tolist(), top.indices.tolist())
        ]


@lru_cache(maxsize=1)
def get_service() -> ResNetEuroSatService:
    return ResNetEuroSatService(WEIGHTS_PATH)
