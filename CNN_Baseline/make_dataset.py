import os
import shutil
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import math
import random
import h5py  
import glob
from sklearn.model_selection import train_test_split


class Dataset_ds(Dataset):
    def __init__(self, filepath_list, transform=None):
        self.filepath_list = filepath_list
        self.length = len(filepath_list)
        self.transform = transform
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        target_dir = self.filepath_list[idx].split('/')[-1]
        # Load cropped scar image
        img_path = self.filepath_list[idx]
        image = [None, None]
        image[0] = Image.open(f'{img_path}/{target_dir}_xz.png')
        image[1] = Image.open(f'{img_path}/{target_dir}_yz.png')
        image0 = torch.tensor(np.expand_dims(image[0], axis=0))
        image1 = torch.tensor(np.expand_dims(image[1], axis=0))
        # load label array
        pid = None
        label = 0
        with open(f'{img_path}/{target_dir}_pid.txt', 'r', encoding='utf-8') as file:
            pid = file.read()
        if pid=='nuecc':
            label=0
        elif pid=='numucc':
            label=1
        else:
            label=2
        label = torch.tensor(label)      
        return image0, image1, label

    def read_h5(self, filepath):
        f = h5py.File(filepath,'r') 
        images = f['images'][:]
        labels = f['pids'][:]
        f.close()
        return images, labels


def classifier_dataloader_cropped2(batch_size, shuffle):
    '''
    Round 2 data, larger dataset.
    '''
    # Set Image Transform
    transform = transforms.Compose([
        transforms.ToTensor(),            # Transform to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

    class_paths = glob.glob("/baldig/physicsprojects2/dikshans/datasets/bigDataset/*/*")
    train_paths , test_paths = train_test_split(class_paths, test_size=0.05, random_state=77, shuffle=True)
    train_dataset = Dataset_ds(train_paths, transform=transform)
    test_dataset = Dataset_ds(test_paths, transform=transform)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Build dataset success. ")
    
    return train_loader, test_loader


if __name__ == '__main__':
    pass