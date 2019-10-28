import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import UrbanSound8KDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

#region data
train_loader_LMC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_train.pkl', 'LMC'),
    batch_size=32, shuffle=True,
    num_workers=8, pin_memory=True)

test_loader_LMC = torch.utils.data.DataLoader(
     UrbanSound8KDataset('UrbanSound8K_test.pkl', 'LMC'),
     batch_size=32, shuffle=False,
     num_workers=8, pin_memory=True)

# train_loader_MC = torch.utils.data.DataLoader(
#       UrbanSound8KDataset('UrbanSound8K_train.pkl', 'MC'),
#       batch_size=32, shuffle=True,
#       num_workers=8, pin_memory=True)

# test_loader_MC = torch.utils.data.DataLoader(
#      UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MC'),
#      batch_size=32, shuffle=False,
#      num_workers=8, pin_memory=True)

# train_loader_MLMC = torch.utils.data.DataLoader(
#       UrbanSound8KDataset('UrbanSound8K_train.pkl', 'MLMC'),
#       batch_size=32, shuffle=True,
#       num_workers=8, pin_memory=True)

# test_loader_MLMC = torch.utils.data.DataLoader(
#      UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MLMC'),
#      batch_size=32, shuffle=False,
#      num_workers=8, pin_memory=True)
#endregion data


class LMC_Net(nn.Module):
    #Initialisation method
    def __init__(self):
        super().__init__()
        #define layers here

        ##1st layer
        self.conv1 = nn.Conv2d(
            #presuming 1 channel input image?????
            in_channels=1,
            out_channels=32,
            kernel_size=(3,3),
            # stride=(2,2),
            padding = 1  ##RUBBISH
        )

        ##### randomly initialise params??????

        self.norm2 = nn.BatchNorm2d(num_features=32)
        
        #relu

        ##2nd layer
        self.dropout3 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (3,3),
            #stride = (2,2)
            padding = 1  ##RUBBISH
        )

        self.norm5 = nn.BatchNorm2d(num_features = 32)
        
        #relu

        #maxpooling
        self.pool6 = nn.MaxPool2d(kernel_size=(2,2),padding=1)

        ##3rd layer
        self.conv7 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3,3),
            #stride = (2,2)
            padding = 1 ##RUBBISH
        )

        self.norm8 = nn.BatchNorm2d(num_features = 64)

        #relu

        ##4th layer
        self.dropout9 = nn.Dropout(p=0.5)

        self.conv10 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            #stride = (2,2)
            padding = 1  ##RUBBISH
        )
        
        self.norm11 = nn.BatchNorm2d(num_features = 64)

        #relu

        #[32, 64, 43, 21]
        ##5th layer
        self.fc12 = nn.Linear(in_features=57792,out_features=1024)
        #Sigmoid
        
        #6th layer
        self.dropout13 = nn.Dropout(p=0.5)
        self.fc14 = nn.Linear(in_features=1024 ,out_features=10  )
        #Softmax

    def forward(self,x):
        #define forward pass here
        ##1
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.relu(x)
        ##2
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = self.pool6(x)
        ##3
        x = self.conv7(x)
        x = self.norm8(x)
        x = F.relu(x)
        ##4
        x = self.dropout9(x)
        x = self.conv10(x)
        x = self.norm11(x)
        x = F.relu(x)

        #Flatten
        x = torch.flatten(x,start_dim = 1)

        ##5
        x = self.fc12(x)
        x = torch.sigmoid(x)
        ##6
        x = self.dropout13(x)
        x = self.fc14(x)
        x = F.softmax(x,dim=1)     #loss function is cross entropy loss, which needs raw logits, so we do not want to apply softmax here
        return x

model = LMC_Net().to(device)
item = next(iter(train_loader_LMC))
print(item[0].shape)
print(model(item[0]).shape)
print()

optimiser = optim.SGD(
    params=model.parameters(),
    lr=0.001,
    momentum = 0.9,
    weight_decay=1e-5#L2 regularization -> what value????
    )

criterion = nn.CrossEntropyLoss()




#####TRAINING LOOP#######
for epoch in range(0,1):
    #for each batch (input is 32 images)
    for i,(input,target,filenames) in enumerate(train_loader_LMC):
        #training loop for single batch
        logits = model(input)
        loss = criterion(logits,target)
        print(i,loss)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

# For each batch- ignore filenames, this is only useful in testing to combine audio segments
#for i,(input,target,filename) in enumerate(train_loader):
#   training code

#for i, (input,target,filename) in enumerate(val_loader):
#   validation code




#region notes

###optimizer: SGD; L2 reg; learn rate 0.001; momentum 0.9
###loss function: cross entropy
#nb: don't use dropout in testing

#result = LMC_model().to(device)
#to instantiate: LMC_model = LMC_Net()
#to do forward pass: output = LMC_model(input)

#####QUESTIONS:
# 1.) stride and padding for the convolutional layers: e.g. how is the image size maintained across the 
# 2.) check location of dropout layers
# 3.) fully connected layers? paper says one layer, seems to be 2? if 1 layer, how can the output of conv layers be flatened to 1024????
# 4.) what L2 regularization value to use? - use common values
# 5.) how do you do the audio segment combination in testing: how de we interpret the filenames thing?
# 6.) How many epochs?

#endregion notes