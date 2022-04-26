# %%
import argparse
import os
import cv2
import random
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import Nuclie_dataset
from model import Unet_block, UNet
from utils import BCEDiceLoss, AverageMeter, count_params, iou_score
from init_weights import init_weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
IMG_HEIGHT=256
IMG_WIDTH=256
train_transform = Compose([
    transforms.Resize(IMG_HEIGHT, IMG_WIDTH),
    OneOf([
    transforms.HorizontalFlip(),
    transforms.VerticalFlip(),
    transforms.RandomRotate90(),],
    p=1),
#    transforms.Cutout(),
    OneOf([
        transforms.HueSaturationValue(),
        transforms.RandomBrightness(),
        transforms.RandomContrast(),
    ], p=1),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

val_transform = Compose([
    transforms.Resize(IMG_HEIGHT, IMG_WIDTH),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

base_path = '../data/stage1_train/'

random_seed = 38
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dataset = Nuclie_dataset(base_path, train=True, transform=train_transform)
val_dataset = Nuclie_dataset(base_path, train=False, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# %%
model = UNet(1, 3).to(device)
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.000068)
# %%
def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss':AverageMeter(),
                  'iou' :AverageMeter()}

    model.train()
    # pbar = tqdm(total=len(train_loader))

    for inputs, labels in train_loader:
        inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
        labels = torch.tensor(labels, device=device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log
        iou = iou_score(outputs, labels, threshold=0.8)
        avg_meters['loss'].update(loss.item(), n=inputs.size(0))
        avg_meters['iou'].update(iou, n=inputs.size(0))

        log = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                ])
    #     pbar.set_postfix(log)
    #     pbar.update(1)
    # pbar.close()
    return log, model

def validation(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    
    model.eval()
    with torch.no_grad():
#        pbar = tqdm(total=len(val_loader))
        for inputs, labels in val_loader:
            inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
            labels = torch.tensor(labels, device=device, dtype=torch.float32)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            iou = iou_score(outputs, labels, threshold=0.8)

            avg_meters['loss'].update(loss.item(), n=inputs.size(0))
            avg_meters['iou'].update(iou, n=inputs.size(0))

            log = OrderedDict([
                        ('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                    ])
#            pbar.set_postfix(log)
#            pbar.update(1)
#        pbar.close()
    return log

# %%
epochs=10
best_iou = 0
for epoch in range(1, epochs+1):
    train_log, model = train(train_loader, model, criterion, optimizer)
    val_log =  validation(val_loader, model, criterion)
    print(f'{epoch}Epoch')
    print(f'train loss:{train_log["loss"]:.3f} |train iou:{train_log["iou"]:.3f}')
    print(f'val loss:{val_log["loss"]:.3f} |val iou:{val_log["iou"]:.3f}\n')
    valid_iou = val_log['iou']
    if best_iou < valid_iou:
        best_iou = valid_iou
#        torch.save(model.state_dict(), f'../results/unet/best_model.pth')

    torch.save(model.state_dict(), f'../results/unet/{epoch}epoch.pth')
# %%
def visualize(image, label, seg_image):
    f, ax = plt.subplots(1, 3, figsize=(20, 8))
    ax[0].imshow(image)
    ax[1].imshow(label, cmap='gray')
    ax[2].imshow(seg_image, cmap='gray')

    ax[0].set_title('Original Image')
    ax[1].set_title('Ground Truth')
    ax[2].set_title('UNet')

    ax[0].title.set_size(25)
    ax[1].title.set_size(25)
    ax[2].title.set_size(25)

    f.tight_layout()
    plt.show()

# %%
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
model.load_state_dict(torch.load('../results/unet/2epoch.pth'))
model.eval()

val_iter = iter(val_loader)

# %%
image, label = next(val_iter)
seg_image = model(image.to(device))
image = torch.sigmoid(image)
seg_image = torch.sigmoid(seg_image)
seg_image[seg_image<0.7]=0
seg_image[seg_image>=0.7]=1
image, label = image[0].permute(1, 2, 0).numpy(), label[0].permute(1, 2, 0).numpy().squeeze()
seg_image = seg_image[0].permute(1, 2, 0).cpu().detach().numpy().squeeze()

visualize(image, label, seg_image)
# %%
