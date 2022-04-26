# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as transforms
import torchtoolbox.transform as tt_transform
from torch.utils.data import Dataset, DataLoader 
from PIL import Image

# %%
IMG_HEIGHT=256
IMG_WIDTH=256
class Nuclie_dataset(Dataset):
    def __init__(self, folder_path, train:bool, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.train = train

        indices = np.random.permutation(len(os.listdir(folder_path)))
        train_length = int(len(os.listdir(folder_path))*0.8)
        if train:
            indices = indices[:train_length]
            self.folders = np.array(os.listdir(folder_path))[indices]
        else:
            indices = indices[train_length:]
            self.folders = np.array(os.listdir(folder_path))[indices]
        
    def __getitem__(self, idx):
        image_folder = os.path.join(self.folder_path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.folder_path, self.folders[idx], 'masks/')
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = self.get_mask(mask_folder, IMG_HEIGHT, IMG_WIDTH)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.folders)
    
    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((256, 256, 3), dtype=np.uint8)
        for mask_name in os.listdir(mask_folder):
            mask_ = cv2.imread(os.path.join(mask_folder, mask_name))
            mask_ = tt_transform.Resize((256, 256))(mask_)   # Mask Size가 이미지마다 다름
            mask += mask_

#        mask = transform.Resize((IMG_HEIGHT, IMG_WIDTH))(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255
        return mask

