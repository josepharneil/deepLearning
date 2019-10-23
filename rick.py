import torch
from torch.utils import data
import numpy as np
import pickle


dataset = pickle.load(open('UrbanSound8K_train.pkl', 'rb'))
LM = dataset[1]['features']['logmelspec']
print(LM.shape)
Chroma = dataset[1]['features']['chroma']
print(Chroma.shape)
SpectralContrast = dataset[1]['features']['spectral_contrast']
print(SpectralContrast.shape)
Tonnetz = dataset[1]['features']['tonnetz']
print(Tonnetz.shape)
feature = np.concatenate((LM,Chroma,SpectralContrast,Tonnetz))
print(feature.shape)
