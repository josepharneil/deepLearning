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

train_loader_LMC = torch.utils.data.DataLoader(
      UrbanSound8KDataset('UrbanSound8K_train.pkl', 'LMC'),
      batch_size=32, shuffle=True,
      num_workers=8, pin_memory=True)

# test_loader_LMC = torch.utils.data.DataLoader(
#      UrbanSound8KDataset('UrbanSound8K_test.pkl', 'LMC'),
#      batch_size=32, shuffle=False,
#      num_workers=8, pin_memory=True)

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

class LMC_Net(nn.Module):
    #Initialisation method
    def __init__(self):
        super().__init__()
        #define layers here
        self.conv1 = nn.Conv2d(
            #presuming 1 channel input image?????
            in_channels=1,
            out_channels=32,
            kernel_size=(3,3),
            stride=(2,2)
        )



    def forward(self,x):
        #define forward pass here
        x = self.conv1(x)
        return x






model = LMC_Net()
item = next(iter(train_loader_LMC))
print(item[0].shape)
print(model(item[0]).shape)

#result = LMC_model().to(device)
#to instantiate: LMC_model = LMC_Net()
#to do forward pass: output = LMC_model(input)

