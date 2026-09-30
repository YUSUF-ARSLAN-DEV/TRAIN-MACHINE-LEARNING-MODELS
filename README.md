# Train Machine Learning Models

A self-directed learning track for building practical ML **fine-tuning** skills across five
projects, spanning computer vision, NLP, object detection, LLM adaptation, and model serving.
A FastAPI + React showcase web app lets each finished model be tried in the browser.

Each project is a focused, end-to-end exercise: load real data, adapt a pre-trained model,
train, evaluate, and (later in the track) serve it. The full roadmap is in [`plan.txt`](./plan.txt).

---

## Progress Tracker

| # | Project | Focus | Stack | Status |
|---|---------|-------|-------|--------|
| 1 | Satellite image classification | Transfer learning on a CNN | ResNet-18 · PyTorch | ✅ Complete — 93% accuracy |
| 2 | Text classification | Fine-tuning a transformer encoder | BERT · Hugging Face | ✅ Complete — 99.7% val / ~73% real-world |
| 3 | Object detection | Detection fine-tuning | YOLOv8 · Ultralytics | 🟡 Trained — mAP50 0.991; ONNX export + write-up pending |
| 4 | LLM adaptation | Parameter-efficient fine-tuning | LoRA · PEFT | ⬜ Planned |
| 5 | Multi-task model + serving | Multi-task fine-tuning and deployment | FastAPI · Docker · MLflow | ⬜ Planned |
| – | Showcase web app | Try each model in a browser | FastAPI · React · Vite | 🟡 BERT live; ResNet blocked on weights; YOLO next |

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

Code: [`RESNET/src/Resnet18.py`](./RESNET/src/Resnet18.py) — training and evaluation loops.

> ⚠️ The script does not yet call `torch.save()`, so no weights exist for the web app to load.

### Run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch torchvision pillow
python RESNET/src/Resnet18.py   # downloads EuroSAT on first run
```

---

## Project 2 — BERT Intent Classification ✅

Fine-tuned a pre-trained BERT encoder to classify customer-support messages by intent.

- **Dataset:** Bitext Customer Support LLM Chatbot Training Dataset — ~27,000 templated
  customer-support instructions, 27 intent classes (e.g. `cancel_order`, `track_refund`,
  `contact_human_agent`).
- **Approach:** `bert-base-uncased` + `AutoModelForSequenceClassification` head
  (`Linear(768, 27)`). Full fine-tune (encoder + head), not just linear probing.
- **Preprocessing:** tokenize with `padding=False, truncation=True`; dynamic per-batch padding
  via `DataCollatorWithPadding`.
- **Splitting:** shuffled row-index split into train/val/test (70/15/15); label mapping built
  once from train and reused across splits.
- **Training:** `Trainer` + `TrainingArguments` — AdamW, lr 2e-5, 3 epochs, eval/save per epoch.
- **Metrics:** accuracy + macro-F1 (sklearn).
- **Hardware:** NVIDIA RTX 5070 (CUDA), ~2.5 min train.
- **Result:** ~99.7% val accuracy / macro-F1. Trained weights are saved and reloadable via `pipeline`.

**⚠️ Evaluation caveat.** The validation number is inflated by template leakage. Bitext
instructions are template-generated near-duplicates, so a random row-level split scatters
near-twin phrasings across train and val. On a manually written, non-template test of 50
sentences, real-world accuracy dropped to **~73%** (avg confidence ~85%) — the honest estimate.
The proper fix is a grouped split by template skeleton (strip `{{slots}}`, group by normalized
phrasing, keep each group on one side). Left as future work.

Files: [`BERT/BERT.py`](./BERT/BERT.py) (main pipeline),
[`BERT/BERT_SUPPORT.py`](./BERT/BERT_SUPPORT.py) (split, tokenize, mapping, metrics),
[`BERT/config.py`](./BERT/config.py) (constants and hand-written test sentences).

```bash
pip install -r BERT/requirements.txt
python BERT/BERT.py            # trains, saves, then runs interactive inference
```

---

## Project 3 — YOLOv8 Retail Shelf Detection 🟡

Fine-tuned a COCO-pretrained **YOLOv8n** to detect products on retail shelves.

- **Dataset:** Roboflow Universe *robust-shelf-monitoring* (CC BY 4.0), YOLO format,
  7 classes: `ariel, boost, ghee, harpic, oil, pickle, tea` — see [`YOLO/data.yaml`](./YOLO/data.yaml).
- **Training:** `yolov8n.pt`, 50 epochs, imgsz 640, batch 16, default augmentation
  (run saved in `runs/detect/train-6`; the earlier `train`…`train-5` folders are aborted attempts).
- **Result (final epoch, validation):**

  | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|
  | 0.994 | 0.992 | **0.991** | **0.778** |

  mAP50 is near-saturated (easy dataset); mAP50-95 is the more informative number.
- **Saved artifacts:** `runs/detect/train-6/` has the PR/F1 curves, confusion matrix,
  `results.csv` and `weights/best.pt`.
- **Inference:** [`YOLO/yolo.py`](./YOLO/yolo.py) runs `best.pt` on a test image.

**Still to do:** compare a larger backbone / augmentation tweaks, export to ONNX and verify with
`onnxruntime`, and finish this write-up.

Note: the image dataset folders (`YOLO/train`, `valid`, `test`) are gitignored — download the
dataset from Roboflow to retrain.

---

## Showcase Web App 🟡

A FastAPI backend + React (Vite) frontend in [`webapp/`](./webapp) — one page per model.
Which models appear is driven by `webapp/backend/app/catalog.py`; see
[`webapp/README.md`](./webapp/README.md) for run instructions.

| Model | State |
|---|---|
| BERT intent classifier | ✅ live |
| ResNet-18 / EuroSAT | UI built; backend returns 503 until weights are saved (and the service's weights path is corrected to `RESNET/src/`) |
| YOLO | next — after ONNX export |
| LoRA / Multi-task | placeholders |

---

## Projects 4 & 5 — Planned ⬜

- **Project 4:** QLoRA fine-tune of an LLM with `peft` + `trl` + `bitsandbytes`.
- **Project 5:** multi-task model, FastAPI endpoint, Docker, MLflow versioning, drift
  monitoring, A/B testing, CI/CD.

Details in [`plan.txt`](./plan.txt).

---

## Tech Stack

| Area | Tools |
|------|-------|
| Language | Python 3, JavaScript |
| Deep learning | PyTorch, torchvision |
| NLP | Hugging Face Transformers, Datasets, scikit-learn |
| Detection | YOLOv8 (Ultralytics) |
| Web app | FastAPI, React, Vite |
| Upcoming | PEFT/TRL/bitsandbytes, Docker, MLflow, GitHub Actions |
| Compute | NVIDIA RTX 5070, CUDA |
| Tooling | venv, Git |

---

## Repository Layout

```
.
├── RESNET/src/Resnet18.py   # Project 1 — training + evaluation
├── BERT/                            # Project 2 — BERT.py, BERT_SUPPORT.py, config.py
├── YOLO/                            # Project 3 — yolo.py, data.yaml
├── runs/detect/train-6/             # Project 3 — trained YOLO run + best.pt
├── webapp/                          # Showcase app (backend/ FastAPI, frontend/ React)
├── plan.txt                         # Full roadmap and progress
├── EUROSAT/, venv/                  # dataset + virtualenv (gitignored)
└── README.md
```

---

*Independent learning project — not affiliated with any coursework or employer.*
