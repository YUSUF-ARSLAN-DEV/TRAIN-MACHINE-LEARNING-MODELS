from pathlib import Path

SEED = 42

# paths (anchored to the YOLO folder so the script works from any working directory)
YOLO_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = YOLO_DIR / "data.yaml"
INITIAL_WEIGHTS = YOLO_DIR / "INITIAL_WEIGHTS" / "yolov8n.pt"
RUN_NAME = "openimages_v8n_e20"
BEST_PT = YOLO_DIR / RUN_NAME / "weights" / "best.pt"
BEST_ONNX = YOLO_DIR / RUN_NAME / "weights" / "best.onnx"
TEST_IMAGES_DIR = YOLO_DIR / "dataset" / "images" / "test"
OUTPUT_DIR = YOLO_DIR / "TEST_RUN_1"

# training
BATCH_SIZE = 48
WORKERS = 2
EPOCHS = 20
IMGSZ = 640 # the training size and the size every image is resized to at inference

# inference / post-processing
CONF_THRESHOLD = 0.25 # boxes whose best class score is below this are dropped
NMS_IOU_THRESHOLD = 0.45 # same-class boxes overlapping a keeper by more than this are duplicates

# same order as data.yaml
CLASS_NAMES = ["Bottle", "Box", "Tin can", "Snack", "Drink", "Milk", "Personal care", "Cosmetics"]
