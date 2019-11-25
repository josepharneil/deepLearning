import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        # Log-mel + CST
        if self.mode == 'LMC':
            # Edit here to load and concatenate the neccessary features to 
            # create the LMC feature
            LM               = self.dataset[index]['features']['logmelspec']
            Chroma           = self.dataset[index]['features']['chroma']
            SpectralContrast = self.dataset[index]['features']['spectral_contrast']
            Tonnetz          = self.dataset[index]['features']['tonnetz']

            # Concat all of above
            feature = np.concatenate((LM,Chroma,SpectralContrast,Tonnetz))

            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        # MFCC + CST
        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to 
            # create the MC feature
            MFCC             = self.dataset[index]['features']['mfcc']
            Chroma           = self.dataset[index]['features']['chroma']
            SpectralContrast = self.dataset[index]['features']['spectral_contrast']
            Tonnetz          = self.dataset[index]['features']['tonnetz']

            feature = np.concatenate((MFCC,Chroma,SpectralContrast,Tonnetz))

            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        # MFCC + LM + CST
        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to 
            # create the MLMC feature
            MFCC             = self.dataset[index]['features']['mfcc']
            LM               = self.dataset[index]['features']['logmelspec']
            Chroma           = self.dataset[index]['features']['chroma']
            SpectralContrast = self.dataset[index]['features']['spectral_contrast']
            Tonnetz          = self.dataset[index]['features']['tonnetz']

            feature = np.concatenate((MFCC,LM,Chroma,SpectralContrast,Tonnetz))

            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
       
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)
