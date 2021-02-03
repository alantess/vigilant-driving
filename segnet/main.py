import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from dataset import SegDataset
from network import SegNet
from train import train_model, test_model

if __name__ == "__main__":
    # NEEDED VARIABLES FOR TRAINING
    BATCH_SIZE = 16
    EPOCHS = 15
    IMAGE_SIZE = 256
    NUM_WORKERS = 4
    PIN_MEM = True
    SEED = 99
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    # Check for GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # PATHS TO DATASETS 
    image_path = "/mnt/d/dataset/LanesBerkleyBDK/images/bdd100k/images/100k/train/" 
    mask_path = "/mnt/d/dataset/LanesBerkleyBDK/map/bdd100k/drivable_maps/color_labels/train/"



    # PREPROCESSING FOR IMAGES 
    preprocess = transforms.Compose([

        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        # transforms.Normalize((0.485, 0.456, 0.406),(0.485, 0.456, 0.406)),
        ]) 

    """
    Shape of images is (batch,CLASSES , 256, 256)
    Shape of masks is (batch,  256, 256)
    """
    # LOSS FUNCTION 
    loss_fn = nn.CrossEntropyLoss()


    # LOADING THE DATASET INTO TRAINLOADER
    trainset = SegDataset(image_path, mask_path, transform=preprocess)
    train_loader = DataLoader(trainset, BATCH_SIZE, num_workers=NUM_WORKERS , pin_memory=PIN_MEM, shuffle=True)

    # Load the model & Optimizer
    model = SegNet() 
    optimizer = torch.optim.Adam(model.parameters(), lr =1e-5)


    #
    # Train Model 
    train_model(model,optimizer,train_loader, loss_fn, device, EPOCHS, True)

    # Test model
    # test_model(model, train_loader, loss_fn, device)




