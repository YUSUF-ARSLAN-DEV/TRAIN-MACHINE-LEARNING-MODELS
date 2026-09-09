from torchvision import models  , transforms ,datasets ,nn 
from torch.utils.data import random_split , DataLoader 
from torch import nn , optim 
import torch as torch 
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

entropy_loss = nn.CrossEntropyLoss() # entroppy_loss is a callable method now 


optimizer = optim.Adam(VisionModel.fc.parameters() , lr=0.001)

num_epochs = 10 
def training_loop() : 
    for i in range (0,num_epochs) :
        for images , labels   in train_loader : 
            # passing the batch 
            predictions =VisionModel(images ) # passing a batch to the model 
            loss = entropy_loss(predictions ,labels )
            loss.backward()
            optimizer.step() 
            loss.zero_grad()

correct = 0  # n of correct images 
total = 0 # total number of iamges 

VisionModel.eval() # swithces the model to evaluation mode 
with torch.no_grad():
    for images , labels in test_loader :  # - labels have 32 values too 
        predictions = VisionModel(images) # this returns 32 rows of 10 scores each 
        _ , predicted_values = torch.max(predictions,1) # 32 values 
        # the above is a batch operation since we passd 32 images in each batch and 10 values are outputed for each we are asking for 
        # the top value in each of these 32 lists so we end up with 32 prediction 
        correct += (predicted_values == labels ).sum().item() 
        total = labels.size(0) # labels is 1d tensor so we want the dimension  of columsn 


print(f"The Accuracy of the model is:\n\n {correct/total}* 100")


         
    