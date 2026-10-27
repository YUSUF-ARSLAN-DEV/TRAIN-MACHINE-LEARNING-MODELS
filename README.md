# Train Machine Learning Models

A self-directed learning track for building practical ML **fine-tuning** skills across five
projects, spanning computer vision, NLP, object detection, LLM adaptation, and model serving.
A FastAPI + React showcase web app lets each finished model be tried in the browser.

Each project is a focused, end-to-end exercise: load real data, adapt a pre-trained model,
train, evaluate, and (later in the track) serve it.

---

## Progress Tracker

| # | Project | Focus | Stack | Status |
|---|---------|-------|-------|--------|
| 1 | Satellite image classification | Transfer learning on a CNN | ResNet-18 · PyTorch | ✅ Complete — 93% accuracy |
| 2 | Text classification | Fine-tuning a transformer encoder | BERT · Hugging Face | ✅ Complete — 99.7% val / ~73% real-world |
| 3 | Object detection | Detection fine-tuning | YOLOv8 · Ultralytics | 🟡 In progress — trained (mAP50 0.547) + ONNX export; NMS + write-up pending |
| 4 | LLM adaptation | Parameter-efficient fine-tuning | LoRA · PEFT | ⬜ Planned |
| 5 | Multi-task model + serving | Multi-task fine-tuning and deployment | FastAPI · Docker · MLflow | ⬜ Planned |
| – | Showcase web app | Try each model in a browser | FastAPI · React · Vite | 🟡 BERT + ResNet live; YOLO next |

---

## How to run

Everything below runs from the **repo root**, in a virtual environment (the pins were compiled with Python 3.14;
the pinned PyTorch is the CUDA 12.8 build, so training expects an NVIDIA GPU).

```bash
python -m venv venv
venv\Scripts\activate             # Windows
pip install -r requirements.txt   # exact pinned versions
pip install -e .                  # makes the shared `common` package (set_seed) importable
```

| Project | Command | What it does |
|---|---|---|
| 1 · ResNet-18 | `python RESNET/src/Resnet18.py` | downloads EuroSAT on first run, trains, saves weights to `RESNET/src/`, prints test accuracy |
| 2 · BERT | `python BERT/BERT.py` | trains, saves to `BERT/bert_intent_model/`, then runs interactive inference |
| 3 · YOLO | `python YOLO/scripts/yolo.py` | runs the ONNX model on the test images and draws boxes into `YOLO/TEST_RUN_1/` (needs the dataset, see Project 3) |
| Web app | see [`webapp/README.md`](./webapp/README.md) | FastAPI backend + React frontend |

**Reproducibility:** every project calls `set_seed()` from `common/seed.py` with the seed in its `config.py`, and the
dependency versions are pinned. Running ResNet twice gives the identical accuracy.
Hyperparameters (epochs, batch size, learning rate, image size, split ratio) live in each project's `config.py`.

**Changing dependencies:** edit `requirements.in` (direct dependencies only), then run
`pip-compile requirements.in -o requirements.txt`. Never edit `requirements.txt` by hand.

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

Trained weights are saved to `RESNET/src/resnet18_eurosat.pth` and served by the web app.

See [How to run](#how-to-run).

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

See [How to run](#how-to-run).

---

## Project 3 — YOLOv8 Object Detection on Open Images 🟡

Fine-tuned a COCO-pretrained **YOLOv8n** to detect 8 generic retail/product classes, exported it
to ONNX, and is hand-writing the inference post-processing.

- **Dataset:** Open Images V7 via FiftyOne, 8 classes: Bottle, Box, Tin can, Snack, Drink, Milk,
  Personal care, Cosmetics (see [`YOLO/data.yaml`](./YOLO/data.yaml)). ~75k training images.
  `YOLO/scripts/image_download.py` downloads; `YOLO/scripts/convert_labels.py` converts the raw
  CSV annotations to YOLO `.txt` labels (`class x_center y_center w h`, normalized).
- **Training:** `yolov8n.pt`, 20 epochs, imgsz 640, batch 48 (run: `YOLO/openimages_v8n_e20/`,
  with PR/F1 curves, confusion matrix, `results.csv`, `weights/best.pt`).
- **Result (test split):**

  | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|
  | 0.600 | 0.461 | **0.547** | **0.438** |

- **ONNX export:** `best.pt` -> `best.onnx`, verified with `onnxruntime`: input `[1, 3, 640, 640]`,
  output `[1, 12, 8400]`.

### Manual ONNX post-processing ([`YOLO/scripts/yolo.py`](./YOLO/scripts/yolo.py))

The raw output is `[batch, 12, 8400]`: 8400 candidate predictions, each with 12 values
(4 box + 8 class scores). Steps implemented so far:

1. Resize to 640x640, scale to 0-1, transpose to channels-first, add a batch dimension.
2. Take `outputs[0][0]` (first image) and transpose to `(8400, 12)`.
3. Split: `boxes = pred[:, :4]` (center-format `x, y, w, h`), `scores = pred[:, 4:]` (8 classes).
4. `argmax` / `max` over the classes gives each candidate's class id and confidence.
5. Keep candidates with confidence > 0.25. On a test image this leaves ~80 boxes.

Those ~80 boxes are mostly **duplicates of the same few objects**: class scores are independent
sigmoids, so many neighbouring grid cells fire on one object. Removing them is the next step.

**Still to do:** center -> corner box conversion, hand-written IoU + NMS, draw boxes on the image,
read the PR curve, compare a larger backbone (`yolov8s/m`) and augmentation, then wire YOLO into the web app.

Earlier experiment: a first run on a Roboflow shelf-monitoring set (7 Nigerian brands, mAP50 0.991)
looked near-perfect but forced unseen brands into the nearest class (closed-set failure), which
motivated switching to the broader Open Images data. Its plots and weights were removed from the repo to keep it small.

Note: `YOLO/dataset/` is gitignored; re-run `image_download.py` and `convert_labels.py` to recreate it.

---

## Showcase Web App 🟡

A FastAPI backend + React (Vite) frontend in [`webapp/`](./webapp) — one page per model.
Which models appear is driven by `webapp/backend/app/catalog.py`; see
[`webapp/README.md`](./webapp/README.md) for run instructions.

| Model | State |
|---|---|
| BERT intent classifier | ✅ live |
| ResNet-18 / EuroSAT | ✅ live |
| YOLO | next — after NMS + drawing works |
| LoRA / Multi-task | placeholders |

---

## Projects 4 & 5 — Planned ⬜

- **Project 4:** QLoRA fine-tune of an LLM with `peft` + `trl` + `bitsandbytes`.
- **Project 5:** multi-task model, FastAPI endpoint, Docker, MLflow versioning, drift
  monitoring, A/B testing, CI/CD.

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
├── YOLO/                            # Project 3 — data.yaml, scripts/ (download, convert, yolo.py)
│   └── openimages_v8n_e20/          #   trained run, weights/best.pt + best.onnx
├── webapp/                          # Showcase app (backend/ FastAPI, frontend/ React)
├── common/seed.py                   # shared set_seed() used by all projects
├── pyproject.toml, requirements.in/.txt  # packaging + pinned dependencies
├── JOURNAL.md                       # what broke, why, how it was fixed
├── EUROSAT/, venv/                  # dataset + virtualenv (gitignored)
└── README.md
```

---

*Independent learning project — not affiliated with any coursework or employer.*
