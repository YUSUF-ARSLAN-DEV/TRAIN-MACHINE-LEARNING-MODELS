from torchvision import models  , transforms ,datasets ,nn 
from torch.utils.data import random_split , DataLoader 
import torch.nn as nn 
from PIL import Image 

VisionModel  = models.resnet18(weights = models.ResNet18_Weights.DEFAULT ) 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# downloading the data set 

transform  = transforms.Compose ([
    transforms.Resize((224,224)) , 
    transforms.ToTensor() , 
    transforms.Normalize(mean,std) 

]

)

EURO_DATA_SET = datasets.EuroSAT(root="EUROSAT" , download=True , transform=transform)

train_set , test_set = random_split(EURO_DATA_SET , [0.8,0.2] )

train_loader = DataLoader(train_set , batch_size=32, shuffle = True  )  # splits the data set into batches of 50 and loads them 
test_loader = DataLoader(test_set,batch_size=32 , shuffle = True ) # the batches are fed to eh moddle different in each epoch 


# Freeze the prexisting layers , before we create a new final linaer layer with 512 inputs and 10 output neurons 

for parameter in VisionModel.parameters() : 
    parameter.required_grad = False 

VisionModel.fc = nn.Layer(512,10) 

