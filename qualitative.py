import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import UrbanSound8KDataset


print("cuda is available: ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#Load the data
#region data
train_loader_LMC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_train.pkl', 'LMC'),
    batch_size=32, shuffle=True,
    num_workers=8, pin_memory=True)

test_loader_LMC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_test.pkl', 'LMC'),
    batch_size=32, shuffle=False,
    num_workers=8, pin_memory=True)

train_loader_MC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_train.pkl', 'MC'),
    batch_size=32, shuffle=True,
    num_workers=8, pin_memory=True)

test_loader_MC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MC'),
    batch_size=32, shuffle=False,
    num_workers=8, pin_memory=True)

train_loader_MLMC = torch.utils.data.DataLoader(
      UrbanSound8KDataset('UrbanSound8K_train.pkl', 'MLMC'),
      batch_size=32, shuffle=True,
      num_workers=8, pin_memory=True)

test_loader_MLMC = torch.utils.data.DataLoader(
     UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MLMC'),
     batch_size=32, shuffle=False,
     num_workers=8, pin_memory=True)

#endregion data

# Define the network classes
#region NetworkClasses
class LMC_Net(nn.Module):
    #Initialisation method
    def __init__(self):
        super().__init__()

        ##1st layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3,3),
            padding = 1 
        )

        self.norm1 = nn.BatchNorm2d(num_features=32)
        
        ##2nd layer
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (3,3),
            # dilation=2,
            padding = 1
        )

        self.norm2 = nn.BatchNorm2d(num_features = 32)

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),padding=1)

        ##3rd layer
        self.conv3 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3,3),
            # dilation=2,
            padding = 1
        )

        self.norm3 = nn.BatchNorm2d(num_features = 64)

        ##4th layer
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            stride = (2,2),  
            # dilation=2,
            padding = 1
        )
        
        self.norm4 = nn.BatchNorm2d(num_features = 64)

        ##5th layer
        self.fc1 = nn.Linear(in_features=15488,out_features=1024)#no dilation
        # self.fc1 = nn.Linear(in_features=46080,out_features=1024)#dilation of 2,2,2
        
        #6th layer
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024 ,out_features=10)

    def forward(self,x):
        ##1
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        ##2
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool1(x) #pooling
        x = self.dropout1(x)

        ##3
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)

        ##4
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.dropout2(x)

        #Flatten
        x = torch.flatten(x,start_dim = 1)

        ##5
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.dropout3(x)

        ##6
        x = self.fc2(x)

        return x

class MLMC_Net(nn.Module):
    #Initialisation method
    def __init__(self):
        super().__init__()
        #define layers here

       ##1st layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3,3),
            padding = 1 
        )

        self.norm1 = nn.BatchNorm2d(num_features=32)
        

        ##2nd layer
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (3,3),
            # dilation=2,
            padding = 1 
        )

        self.norm2 = nn.BatchNorm2d(num_features = 32)

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),padding=1)

        ##3rd layer
        self.conv3 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3,3),
            # dilation=2,
            padding = 1
        )

        self.norm3 = nn.BatchNorm2d(num_features = 64)

        ##4th layer
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            stride = (2,2),
            # dilation=2,
            padding = 1
        )
        
        self.norm4 = nn.BatchNorm2d(num_features = 64)

        ##5th layer
        self.fc1 = nn.Linear(in_features=26048,out_features=1024)#no dilation
        # self.fc1 = nn.Linear(in_features=80640,out_features=1024)#dilation of 2,2,2
        
        #6th layer
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024 ,out_features=10)

    def forward(self,x):
        ##1
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        ##2
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool1(x) #pooling
        x = self.dropout1(x)

        ##3
        x = self.conv3(x)

        x = self.norm3(x)
        x = F.relu(x)

        ##4
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.dropout2(x)

        #Flatten
        x = torch.flatten(x,start_dim = 1)

        ##5
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.dropout3(x)

        ##6
        x = self.fc2(x)

        return x

#endregion NetworkClasses


#Save image function
def saveImage(inputJ, outputName):
    inputJ = inputJ.cpu()
    inputJ = inputJ.numpy()
    input2d = np.squeeze(inputJ, axis=0)
    input2d = np.squeeze(input2d, axis=0)
    plt.imshow(input2d)
    plt.savefig(outputName)

#Create, load and setup the saved models
LMC_model = LMC_Net().to(device)
MC_model = LMC_Net().to(device)

LMC_model.load_state_dict(torch.load('modelsBase/lmc.pt'))
MC_model.load_state_dict(torch.load('modelsBase/mc.pt'))

LMC_model.eval()
MC_model.eval()


# Find images
foundOutput1 = False
foundOutput21 = False
foundOutput22 = False
foundOutput3 = False
foundOutput4 = False

softmax = nn.Softmax(dim=0)
#for each batch
for i,(input,target,filenames) in enumerate(test_loader_LMC):
    for i2,(input2,target2,filenames2) in enumerate(test_loader_MC):
        if(i == i2):
            #for each image
            for j in range(0,input.shape[0]):
                #Get image and target
                im_LMC = input[j].to(device)
                im_LMC.unsqueeze_(0)
                im_MC = input2[j].to(device)
                im_MC.unsqueeze_(0)

                targ = target[j].to(device)
                targ.unsqueeze_(0)

                #Get logits
                LMC_logits = LMC_model(im_LMC)
                MC_logits  = MC_model(im_MC)

                #Get predictions
                LMC_prediction = torch.argmax(LMC_logits,dim=-1).to(device)
                MC_prediction = torch.argmax(MC_logits,dim = -1).to(device)
                TSCNN_prediction = torch.argmax((softmax(MC_logits)+softmax(LMC_logits)),dim=-1)

                #Check if correct
                LMC_isCorrect = (torch.equal(LMC_prediction, targ))
                MC_isCorrect = (torch.equal(MC_prediction, targ))
                TSCNN_isCorrect = (torch.equal(TSCNN_prediction, targ))

                #Output image when appropriate

                ###output 1 LMC and MC correct
                if((LMC_isCorrect) and (MC_isCorrect) and (foundOutput1 == False)):
                    saveImage(im_LMC,'outsBase/out1-LMC.png')
                    saveImage(im_MC,'outsBase/out1-MC.png')
                    foundOutput1 = True
                ###output 2.1 LMC correct and MC incorrect
                if(LMC_isCorrect and (not MC_isCorrect) and (foundOutput21 == False)):
                    saveImage(im_LMC,'outsBase/out21-LMC.png')
                    saveImage(im_MC,'outsBase/out21-MC.png')
                    foundOutput21 = True
                ###output 2.2 LMC incorrect and MC correct
                if((not LMC_isCorrect) and MC_isCorrect and (foundOutput22 == False)):
                    saveImage(im_LMC,'outsBase/out22-LMC.png')
                    saveImage(im_MC,'outsBase/out22-MC.png')
                    foundOutput22 = True
                ###output 3 TSCNN correct, LMC incorrect, MC incorrect
                if(TSCNN_isCorrect and (not LMC_isCorrect) and (not MC_isCorrect) and (foundOutput3 == False)):
                    saveImage(im_LMC,'outsBase/out3-LMC.png')
                    saveImage(im_MC,'outsBase/out3-MC.png')
                    foundOutput3 = True
                ###output 4 all incorrect
                if((not LMC_isCorrect ) and (not MC_isCorrect ) and (not TSCNN_isCorrect) and (foundOutput4 == False)):
                    saveImage(im_LMC,'outsBase/out4-LMC.png')
                    saveImage(im_MC,'outsBase/out4-MC.png')
                    foundOutput4 = True
                
            if(foundOutput1 and foundOutput21 and foundOutput22 and foundOutput3 and foundOutput4):
                print("found all: breaking")
                break

            break
            
if(foundOutput1 == False):  print("Output 1  not found: No case where both LMC and MC are correct")
if(foundOutput21 == False): print("Output 21 not found: No case where LMC is correct, and MC is incorrect")
if(foundOutput22 == False): print("Output 22 not found: No case where LMC is incorrect, and MC is correct")
if(foundOutput3 == False):  print("Output 3  not found: No case where both LMC and MC are incorrect, and TSCNN is correct")
if(foundOutput4 == False):  print("Output 4  not found: No case where all are incorrect")
