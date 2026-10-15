"""Convert the raw Open Images V7 download into a YOLO-format dataset.

Input : TRAIN_IMAGES/open-images-v7/<split>/data/*.jpg + <split>/labels/detections.csv
Output: dataset/images/<split>/*.jpg (hardlinks, no extra disk) + dataset/labels/<split>/*.txt

Each .txt row is:  class x_center y_center width height   (all normalized 0-1)
Open Images already stores boxes normalized as XMin/XMax/YMin/YMax, so no image
size is needed. Class order follows the `names` list in data.yaml.

Run from anywhere:  python YOLO/convert_labels.py
"""
import csv
import os
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE / "TRAIN_IMAGES" / "open-images-v7"
OUT = HERE / "dataset"
CLASS_CSV = HERE / "TRAIN_IMAGES" / "class-descriptions-boxable.csv"
CLASS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"

SPLITS = {"train": "train", "validation": "val", "test": "test"}  # source name -> YOLO name
SKIP_GROUP_BOXES = True  # IsGroupOf=1 boxes cover a crowd, not one object

names = yaml.safe_load((HERE / "data.yaml").read_text())["names"]

if not CLASS_CSV.exists():
    urllib.request.urlretrieve(CLASS_URL, CLASS_CSV)

# MID (e.g. /m/09j2d) -> class index, only for the classes we care about
with open(CLASS_CSV, newline="", encoding="utf-8") as f:
    name_to_mid = {row[1]: row[0] for row in csv.reader(f)}
missing = [n for n in names if n not in name_to_mid]
if missing:
    raise SystemExit(f"Class names not found in Open Images: {missing}")
mid_to_idx = {name_to_mid[n]: i for i, n in enumerate(names)}


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


for src_split, yolo_split in SPLITS.items():
    img_dir = SRC / src_split / "data"
    det_csv = SRC / src_split / "labels" / "detections.csv"
    out_img = OUT / "images" / yolo_split
    out_lbl = OUT / "labels" / yolo_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    present = {p.stem: p for p in img_dir.glob("*.jpg")}
    boxes = defaultdict(list)
    with open(det_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] not in present:
                continue
            idx = mid_to_idx.get(row["LabelName"])
            if idx is None:
                continue
            if SKIP_GROUP_BOXES and row["IsGroupOf"] == "1":
                continue
            x0, x1 = float(row["XMin"]), float(row["XMax"])
            y0, y1 = float(row["YMin"]), float(row["YMax"])
            boxes[row["ImageID"]].append(
                f"{idx} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"
            )

    kept = 0
    for image_id, img_path in present.items():
        rows = boxes.get(image_id)
        if not rows:  # no box for our classes -> would be a pure background image; skip
            continue
        link_or_copy(img_path, out_img / img_path.name)
        (out_lbl / f"{image_id}.txt").write_text("\n".join(rows) + "\n")
        kept += 1
    print(f"{yolo_split}: {kept}/{len(present)} images kept, "
          f"{sum(len(v) for v in boxes.values())} boxes")
