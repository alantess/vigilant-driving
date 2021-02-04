import torch
import numpy as np
from dataset import DepthDataset

if __name__ == '__main__':
    # PATHS NEEDED FOR DATASET
    camera_path = "D:\\dataset\\depth_perception\\train\\stereo_train_001\\camera_5"
    fg_path = "D:\\dataset\\depth_perception\\train\\stereo_train_001\\fg_mask"
    bg_path = 'D:\\dataset\\depth_perception\\train\\stereo_train_001\\bg_mask'
    disparity_path = "D:\\dataset\\depth_perception\\train\\stereo_train_001\\disparity"

    # Variables Needed for training
    SEED = 112
    BATCH = 3
    PIN_MEM = True
    WORKERS = 2
    EPOCHS = 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True


