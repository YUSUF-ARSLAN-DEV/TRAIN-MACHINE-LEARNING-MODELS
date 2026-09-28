# Train Machine Learning Models

A self-directed learning track for building practical ML **fine-tuning** skills across five
projects, spanning computer vision, NLP, object detection, LLM adaptation, and model serving.

Each project is a focused, end-to-end exercise: load real data, adapt a pre-trained model,
train, evaluate, and (later in the track) serve it.

---

## Progress Tracker

| # | Project | Focus | Stack | Status |
|---|---------|-------|-------|--------|
| 1 | Satellite image classification | Transfer learning on a CNN | ResNet-18 · PyTorch | ✅ Complete — 93% accuracy |
| 2 | Text classification | Fine-tuning a transformer encoder | BERT · Hugging Face | ⬜ Planned |
| 3 | Object detection | Detection fine-tuning | YOLO | ⬜ Planned |
| 4 | LLM adaptation | Parameter-efficient fine-tuning | LoRA · PEFT | ⬜ Planned |
| 5 | Multi-task model + serving | Multi-task fine-tuning and deployment | Serving pipeline | ⬜ Planned |

---

## Project 1 — ResNet-18 on EuroSAT ✅

Fine-tuned a pre-trained **ResNet-18** to classify Sentinel-2 satellite imagery by land use.

- **Dataset:** [EuroSAT](https://github.com/phelber/EuroSAT) — ~27,000 Sentinel-2 images,
  64×64 px, 10 land-use classes (e.g. forest, residential, river, industrial).
- **Approach:** Frozen ImageNet backbone; replaced the final fully-connected layer with a new
  `Linear(512, 10)` and trained **only that layer** (linear probing).
- **Preprocessing:** resize to 224×224, ImageNet mean/std normalization.
- **Training:** 80/20 train/test split, batch size 32, Adam (lr 1e-3),
  cross-entropy loss, 10 epochs.
- **Hardware:** NVIDIA RTX 5070 (CUDA).
- **Result:** **~93% test accuracy.**

Code: [`Resnet18.py`](./Resnet18.py) — training and evaluation loops.
`chec.py` is a quick CUDA-availability check.

### Run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch torchvision pillow
python Resnet18.py             # downloads EuroSAT on first run
```

---

## Tech Stack

| Area | Tools |
|------|-------|
| Language | Python 3 |
| Deep learning | PyTorch, torchvision |
| NLP (upcoming) | Hugging Face Transformers, Datasets, PEFT |
| Detection (upcoming) | YOLO (Ultralytics) |
| Serving (upcoming) | FastAPI / TorchServe |
| Compute | NVIDIA RTX 5070, CUDA |
| Tooling | venv, Git |

---

## Repository Layout

```
.
├── Resnet18.py      # Project 1 — training + evaluation
├── chec.py          # CUDA sanity check
├── EUROSAT/         # dataset (gitignored, auto-downloaded)
├── venv/            # virtual environment (gitignored)
└── README.md
```

---

*Independent learning project — not affiliated with any coursework or employer.*
Project 2 — BERT Intent Classification ✅
Fine-tuned a pre-trained BERT encoder to classify customer-support messages by intent.

Dataset: Bitext Customer Support LLM Chatbot Training Dataset — ~27,000 templated customer-support instructions, 27 intent classes (e.g. cancel_order, track_refund, contact_human_agent).

Approach: bert-base-uncased + AutoModelForSequenceClassification head (Linear(768, 27)). Full fine-tune (encoder + head), not just linear probing.

Preprocessing: tokenize with padding=False + truncation=True; dynamic per-batch padding via DataCollatorWithPadding.

Splitting: raw train split into train/val/test using shuffled row indices; label mapping built once from train and reused across all splits.

Training: Trainer + TrainingArguments — AdamW, lr 2e-5, 3 epochs, batch size 16, eval/save per epoch.

Metrics: accuracy + macro-F1 via sklearn.

Hardware: NVIDIA RTX 5070 (CUDA), ~2.5 min train.

Result: ~99.7% val accuracy / macro-F1. Model saved to OUTPUT_DIR with id2label/label2id set, reloadable via pipeline.

⚠️ Evaluation caveat
The validation number is inflated by template leakage. Bitext instructions are template-generated near-duplicates (e.g. "cancel order {{Order Number}}" variants). A random row-level split scatters near-twin phrasings across train and val, so the model is partly recognizing seen patterns rather than generalizing. On a manually written, non-template test of 50 sentences, real-world accuracy dropped to ~73% (avg confidence ~85%) — the honest estimate. The correct fix is a grouped split by template skeleton (strip {{slots}}, group by normalized phrasing, keep each group on one side). Left as future work; noted here so the val number isn't misread.

Run it
bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch transformers datasets scikit-learn accelerate
python BERT.py                 # trains, saves, then runs interactive inference
Files:

BERT.py — main pipeline (load → split → tokenize → train → save → inference)

BERT_SUPPORT.py — split, tokenize, mapping, Trainer args, metrics

config.py — shared constants (model name, split %, paths)

