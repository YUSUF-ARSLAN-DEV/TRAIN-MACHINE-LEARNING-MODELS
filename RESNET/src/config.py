import torch
from pathlib import Path 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path=Path(__file__).resolve().parent/"resnet18_eurosat.pth"
SEED=42

# training
num_epochs = 10
learning_rate = 0.001

# preprocessing
transform_h = 224
transform_w = 224

# data
data_root = "EUROSAT"
train_ratio = 0.8 # the test ratio is 1 - train_ratio

# loaders (the test loader never shuffles, so only train has a switch)
load_batch_size = 32
load_workers = 4
pin_memory_switch = True
train_shuffle = True
