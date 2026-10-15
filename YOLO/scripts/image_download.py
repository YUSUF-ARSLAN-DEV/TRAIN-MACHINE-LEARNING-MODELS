import fiftyone as fo
import fiftyone.zoo as foz
classes = ["Bottle", "Box", "Tin can", "Snack", "Drink", "Milk", "Personal care", "Cosmetics"]
fo.config.dataset_zoo_dir = "TRAIN_IMAGES/"
ds = foz.load_zoo_dataset("open-images-v7", split="train", label_types=["detections"], classes=classes, max_samples=50000)
