import torch 
num_epochs = 10 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path="resnet18_eurosat.pth"