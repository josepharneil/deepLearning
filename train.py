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

train_loader = torch.utils.data.DataLoader(
      UrbanSound8KDataset('UrbanSound8K_train.pkl', 'LMC'),
      batch_size=32, shuffle=True,
      num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
     UrbanSound8KDataset('UrbanSound8K_test.pkl', 'LMC'),
     batch_size=32, shuffle=False,
     num_workers=8, pin_memory=True)