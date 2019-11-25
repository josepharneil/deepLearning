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

summary_writer = SummaryWriter('logs',flush_secs=5)

print("cuda is available: ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

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

# train_loader_MLMC = torch.utils.data.DataLoader(
#       UrbanSound8KDataset('UrbanSound8K_train.pkl', 'MLMC'),
#       batch_size=32, shuffle=True,
#       num_workers=8, pin_memory=True)

# test_loader_MLMC = torch.utils.data.DataLoader(
#      UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MLMC'),
#      batch_size=32, shuffle=False,
#      num_workers=8, pin_memory=True)


# from PIL import Image
for i,(input,target,filenames) in enumerate(train_loader_LMC):
    input2d = np.squeeze(input[i], axis=0)
    plt.imshow(input2d.numpy())
    plt.show()
    # break


for i,(input,target,filenames) in enumerate(train_loader_MC):
    input2d = np.squeeze(input[12], axis=0)
    plt.imshow(input2d.numpy())
    plt.show()
    break