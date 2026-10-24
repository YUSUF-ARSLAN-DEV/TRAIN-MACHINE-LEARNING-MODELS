# train-machine-learning-models

Fine-tuning track: ResNet-18 on EuroSAT, BERT intent classification (Bitext), YOLOv8 retail detection with ONNX export,
plus a FastAPI + React webapp that serves the models.

Goal until 20 Nov 2026: reproducible seeded runs logged to MLflow, honest test-set numbers for all three models,
a model registry with a promotion gate, Dockerized and load-tested serving, drift monitoring for BERT.

## How we work: three levels

I'm on a one-month plan. I don't hand-write everything; I own the decisions and can explain every line.

**1. I write it myself (use the learning loop below).** Small, but it is the judgment an interviewer will probe:
- BERT grouped split by template skeleton (`skeleton()` in `BERT/BERT_SUPPORT.py`)
- IoU and non-maximum suppression in `YOLO/scripts/yolo.py`
- the promotion rule: when a candidate model beats production
- the drift threshold and what counts as drift

**2. You draft, I own it.** MLflow logging and registry calls, Dockerfiles, CI workflows, pytest tests, load-test scripts.
- Before writing, say in 3-5 lines what you'll build and why.
- After writing, explain each block: what it does and what breaks without it.
- Leave one small change for me to make by hand, and tell me which.

**3. Delegate.** React UI in `webapp/frontend/`, README formatting, boilerplate config. Just do it and tell me what changed.

If you're unsure which level something is, ask.

## Learning loop (level 1 only)

1. I explain the task in my own words; you push back on gaps before any code.
2. I write a draft; you don't offer your version first.
3. You review it against my intent: logic errors vs syntax I don't know yet. Explain the fix, scoped to the broken piece.
4. I rewrite from the explanation, not by copy-paste.
5. Add what I missed to `JOURNAL.md` under "syntax I don't own yet"; point out repeats.

## Rules

- No rabbit holes: solve today's task only. If I've been stuck 45 minutes, suggest the simplest workaround.
  Interesting-but-not-needed ideas go to `LATER.md`, not into the code.
- Every reported number comes from a seeded run logged to MLflow, evaluated on the test split (never val alone).
  Don't write a number in the README or anywhere else that no run produced.
- Watch for leakage: near-duplicate rows across splits, tuning on the test set, test data in training.
  If you see it, stop and tell me.
- Don't retrain on the full dataset just to check something; say what it would cost first (GPU is an RTX 5070).
- Interview prep mode: if I say "interview prep", switch off all of this and don't help.