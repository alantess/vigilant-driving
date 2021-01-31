import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from dataset import SegDataset
from network import UNet
from train import train_model, test_model

if __name__ == "__main__":
    # NEEDED VARIABLES FOR TRAINING
    BATCH_SIZE = 2
    EPOCHS = 100
    SEED = 23
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    # Check for GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # PATHS TO DATASETS 
    image_path = "/mnt/d/dataset/LanesBerkleyBDK/images/bdd100k/images/100k/train/" 
    mask_path = "/mnt/d/dataset/LanesBerkleyBDK/map/bdd100k/drivable_maps/labels/train/"

    # PREPROCESSING FOR IMAGES 
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)), 
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]) 

    """
    Shape of images is (batch, 3, 256, 256)
    Shape of masks is (batch, 1 , 256, 256)
    """
    # LOSS FUNCTION 
    loss_fn = nn.MSELoss()


    # LOADING THE DATASET INTO TRAINLOADER
    trainset = SegDataset(image_path, mask_path, transform=preprocess)
    train_loader = DataLoader(trainset, BATCH_SIZE, num_workers=2 , pin_memory=True)

    # Load the model & Optimizer
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

    # Train Model 
    train_model(model,optimizer,train_loader, loss_fn, device, EPOCHS)

    # Test model
    # test_model(model, train_loader, loss_fn, device)
    



