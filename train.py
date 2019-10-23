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