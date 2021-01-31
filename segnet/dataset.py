import os
import cv2 
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision


class SegDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.transform = transform
        self.img = sorted(glob.glob(os.path.join(image_path, "*.jpg")))
        self.mask = sorted(glob.glob(os.path.join(mask_path, "*.png")))

        self.img_path = image_path
        self.mask_path = mask_path

    def __len__(self):
        return len(self.mask) // 7 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve the image correspoding to the mask
        file_name = str(os.path.basename(self.mask[idx]))[:17]
        image_path = self.img_path + file_name + ".jpg"

        # Read the image and mask        
        ####################################################################
        # Default size is 720 x 1280
        ####################################################################
        image = cv2.imread(image_path)
        mask = cv2.imread(self.mask[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        mask = np.expand_dims(mask,2)

        # Transforms images and mask 
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask).squeeze(0)


        return image, mask



