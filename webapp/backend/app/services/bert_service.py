"""Loads the trained BERT intent classifier once and serves predictions.

Points at BERT/bert_intent_model, the directory the training script
(BERT/BERT.py) already fine-tuned and saved — nothing here retrains or
modifies that model.
"""

from pathlib import Path
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO_ROOT / "BERT" / "bert_intent_model"


class BertIntentService:
    def __init__(self, model_dir: Path):
        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"No trained BERT model found at {model_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, text: str, top_k: int = 5):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.shape[-1]))

        results = [
            {"label": self.id2label[int(idx)], "score": float(score)}
            for score, idx in zip(top.values.tolist(), top.indices.tolist())
        ]
        return results


@lru_cache(maxsize=1)
def get_service() -> BertIntentService:
    return BertIntentService(MODEL_DIR)
