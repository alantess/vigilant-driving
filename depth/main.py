import torch
import torch.nn as nn
import numpy as np
from dataset import DepthDataset
from network import *
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader



def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, load_model=False):
    """
    :param model: Input model 
    :param train_loader: Training Set  
    :param val_loader: Validation Set 
    :param optimizer: Optimizer  
    :param loss_fn: Loss function 
    :param device: GPU or CPU 
    :param epochs: Training iteration 
    :param load_model: Loads saved model 
    :return: None 
    """
    scaler = torch.cuda.amp.GradScaler()
    best_score = np.inf
    model = model.to(device)

    if load_model:
        model.load()
        print("Model Loaded.")

    print("Starting...")
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        val_loop = tqdm(val_loader)
        total_loss = 0
        for i , (image, _ , _,target) in enumerate(loop):
            # Set input and target to device
            image = image.to(device, dtype=torch.float32)
            target = target.to(device,dtype=torch.float32)
            # clear gradients
            for p in model.parameters():
                p.grad = None

            # Forward Pass
            with torch.cuda.amp.autocast():
                pred = model(image)
                loss = loss_fn(pred, target)

            # Backward Pass 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Check performance on validation set
        print("VALIDATION:")
        with torch.no_grad():
            val_loss = 0
            for j , (input, _ , _, y) in enumerate(val_loop):
                input = input.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)

                with torch.cuda.amp.autocast():
                    prediction = model(input)
                    v_loss = loss_fn(prediction, y)

                val_loss += v_loss.item()
                val_loop.set_postfix(val_loss=v_loss.item())
        
        # Saves model based on validation loss
        if val_loss < best_score:
            best_score = val_loss
            model.save()
            print("Model saved.")

        print(f"EPOCH {epoch}: {total_loss:.5f}")
        print(f"Validation Loss: {val_loss}")

if __name__ == '__main__':
    # PATHS NEEDED FOR TRAINING SET
    camera_path = "D:\\dataset\\depth_perception\\train\\stereo_train_001\\camera_5"
    fg_path = "D:\\dataset\\depth_perception\\train\\stereo_train_001\\fg_mask"
    bg_path = 'D:\\dataset\\depth_perception\\train\\stereo_train_001\\bg_mask'
    disparity_path = "D:\\dataset\\depth_perception\\train\\stereo_train_001\\disparity"

    # PATH NEEDED FOR VALIDATION SET
    val_camera_path = "D:\\dataset\\depth_perception\\val\\stereo_train_002\\camera_5"
    val_fg_path = "D:\\dataset\\depth_perception\\val\\stereo_train_002\\fg_mask"
    val_bg_path = 'D:\\dataset\\depth_perception\\val\\stereo_train_002\\bg_mask'
    val_disparity_path = "D:\\dataset\\depth_perception\\val\\stereo_train_002\\disparity"

    # Variables Needed for training
    SEED = 99
    BATCH = 3
    SIZE = 256
    PIN_MEM = True
    WORKERS = 2
    EPOCHS = 7
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    # Preprocessing images
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(0.2),
        transforms.Resize((SIZE,SIZE)),

    ])
    """
    Returns Camera, background mask, foreground mask (CARS), and disparity
    """
    # Train Loader
    trainset = DepthDataset(camera_path, bg_path, fg_path, disparity_path, preprocess)
    train_loader = DataLoader(trainset, batch_size=BATCH, num_workers=WORKERS, pin_memory=PIN_MEM)

    # Validatin Loader
    valset = DepthDataset(val_camera_path,val_bg_path,val_fg_path,val_disparity_path,preprocess,True)
    val_loader = DataLoader(valset, batch_size=BATCH,num_workers=WORKERS,pin_memory=PIN_MEM)

    # Model
    model = URes()

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),  lr=1e-5)

    # Uncomment the line below to begin training.
    # train(model,train_loader,val_loader,optimizer,loss_fn,device,EPOCHS,True)
