import torch
from torch.utils.data import Dataset
import numpy as np

class SRDataset(Dataset):
    def __init__(self, lr_path, hr_path, augment=True):
        self.lr_imgs = np.load(lr_path)
        self.hr_imgs = np.load(hr_path)
        self.augment = augment
    
    def __len__(self):
        return len(self.lr_imgs)
    
    def __getitem__(self, idx):
        lr = self.lr_imgs[idx]
        hr = self.hr_imgs[idx]
        
        if self.augment:
            if np.random.rand() < 0.5:
                lr = np.fliplr(lr)
                hr = np.fliplr(hr)
            if np.random.rand() < 0.5:
                lr = np.flipud(lr)
                hr = np.flipud(hr)
            k = np.random.randint(0, 4)
            lr = np.rot90(lr, k)
            hr = np.rot90(hr, k)
        
        lr = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        hr = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        return lr, hr