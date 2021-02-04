import os
import glob
import torch
import numpy as np
import cv2
import torchvision
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    def __init__(self,camera_path, bg_path, fg_path, disparity_path, transform=None):
        self.transform = transform
        self.camera = sorted(glob.glob(os.path.join(camera_path, "*.jpg")))
        self.bg_mask = sorted(glob.glob(os.path.join(bg_path, "*.png")))
        self.fg_mask = sorted(glob.glob(os.path.join(fg_path, "*.png")))
        self.disparity = sorted(glob.glob(os.path.join(disparity_path, ".png")))

    def __len__(self):
        return len(self.camera)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        camera = cv2.imread(self.camera[idx])
        bg_mask = cv2.imread(self.bg_mask[idx], cv2.IMREAD_GRAYSCALE)
        fg_mask = cv2.imread(self.fg_mask[idx], cv2.IMREAD_GRAYSCALE)
        disparity = cv2.imread(self.disparity[idx])

        if self.transform:
            camera = self.transform(camera)
            bg_mask = self.transform(bg_mask)
            fg_mask = self.transform(fg_mask)
            disparity = self.transform(disparity)

        bg_mask[bg_mask>0.5] = 1.0
        fg_mask[fg_mask>0.5] = 1.0

        return camera, bg_mask, fg_mask, disparity




