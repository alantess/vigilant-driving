import os
import glob
import torch
import numpy as np
import cv2
import torchvision
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    def __init__(self,camera_path, bg_path, fg_path, disparity_path, transform=None,val_set=False):
        self.val_set = val_set
        self.transform = transform
        self.camera = sorted(glob.glob(os.path.join(camera_path, "*.jpg")))
        self.bg_mask = sorted(glob.glob(os.path.join(bg_path, "*.png")))
        self.fg_mask = sorted(glob.glob(os.path.join(fg_path, "*.png")))
        self.disparity = sorted(glob.glob(os.path.join(disparity_path, "*.png")))

    def __len__(self):
        if self.val_set:
            return len(self.camera) // 4
        else:
            return len(self.camera)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        """
        Default Camera shape: 960x3180x3
        Grayscale Image shape: 960x3130
        """
        camera = cv2.imread(self.camera[idx])
        bg_mask = cv2.imread(self.bg_mask[idx])
        fg_mask = cv2.imread(self.fg_mask[idx])
        disparity = cv2.imread(self.disparity[idx])

        if self.transform:
            normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            # norm_pos = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.2,1,1))
            grayscale = torchvision.transforms.Grayscale()
            camera = normalize(self .transform(camera))
            bg_mask = grayscale(self.transform(bg_mask)).squeeze(0)
            fg_mask = grayscale(self.transform(fg_mask)).squeeze(0)
            disparity = normalize(self.transform(disparity))
            disparity = grayscale(disparity)

        # disparity[disparity < -1.5] = 3.0
        # disparity[disparity < -0.5] = 4.0
        fg_mask[fg_mask>0.5] = 1.0
        bg_mask[bg_mask>0.5] = 1.0



        return camera, bg_mask, fg_mask, disparity



