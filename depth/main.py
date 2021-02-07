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
            image = image.to(device, dtype=torch.float32)
            target = target.to(device,dtype=torch.float32)

            for p in model.parameters():
                p.grad = None


            with torch.cuda.amp.autocast():
                pred = model(image)
                loss = loss_fn(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())


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

        if val_loss < best_score:
            best_score = val_loss
            model.save()
            print("Model saved.")

        print(f"EPOCH {epoch}: {total_loss:.5f}")
        print(f"Validation Loss: {val_loss}")




def imshow(img,mask):
    # resize = transforms.Resize((960,3180))
    # img = resize(img)
    # mask = resize(mask)
    img = img / 2 + 0.5
    plt.title("WORKING")
    plt.imshow(img.permute(1,2,0), cmap='gray')
    plt.imshow(mask, cmap='CMRmap', alpha=0.5)
    plt.show()

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
    BATCH = 12
    SIZE = 256
    PIN_MEM = True
    WORKERS = 2
    EPOCHS = 40
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
    model = GTRes()

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),  lr=1e-5)

    # Training
    train(model,train_loader,val_loader,optimizer,loss_fn,device,EPOCHS)
    # Training for u_net

    # img,_,_,y = next(iter(train_loader))
    # with torch.no_grad():
    #     out = model(img)
    #     print(y.size())
    #     print(out.size())
    # print(y.size())
    # model.load()
    # model.eval()
    #
    # with torch.no_grad():
    #     output = model(img).argmax(1)
    # imshow(img[3], output[3].detach().numpy())