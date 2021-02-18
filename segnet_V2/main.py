import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from dataset import SegDataset
from network import SegNetV2


def train(model, device, train_loader, val_loader, loss_fn, optimizer, epochs, load_model=False):
    global total_loss
    if load_model:
        model.load()
        print("Model Loaded")

    model.to(device)
    best_score = np.inf
    scaler = torch.cuda.amp.GradScaler()

    print("Starting...")
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Training Epoch #{epoch}")
        val_loop = tqdm(val_loader, desc=f"Validation Epoch #{epoch}")
        total_loss = 0
        for i, (img, mask) in enumerate(loop):
            img = img.to(device, dtype=torch.float32)  # NxCxHxW
            mask = mask.to(device, dtype=torch.long)  # NxHxW

            for p in model.parameters():
                p.grad = None

            # Forward Pass
            with torch.cuda.amp.autocast():
                pred = model(img)  # NxClassxHxW
                loss = loss_fn(pred, mask)

            # Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        with torch.no_grad():
            """
            Checks model performance on the validation set
            """
            val_loss = 0
            for j, (input, target) in enumerate(val_loop):
                input = input.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.long)

                with torch.cuda.amp.autocast():
                    prediction = model(input)
                    v_loss = loss_fn(prediction, target)

                val_loss += v_loss.item()
                val_loop.set_postfix(loss=v_loss.item())

        # # Save model if it does well on the validation set
        if val_loss < best_score:
            best_score = val_loss
            model.save()
            print("Model saved.")
        #
        # print(f'Training Loss {total_loss:.3f} \t Validation Loss {val_loss:.3f}')
        #

def imshow(img, mask):
    plt.ion()
    for i in range(img.size(0)):
        plt.imshow(img[i].permute(1, 2, 0), cmap='gray')
        plt.imshow(mask[i], cmap='gist_ncar', alpha=0.2)
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    # Directories
    img_dir = "D:\\dataset\\comma10k\\imgs"
    mask_dir = "D:\\dataset\\comma10k\\masks"
    val_imgs_dir = "D:\\dataset\comma10k\\val_imgs"
    val_masks_dir = "D:\\dataset\\comma10k\\val_masks"

    # Needed for training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 22
    EPOCHS = 20
    IMAGE_SIZE = 256
    NUM_WORKERS = 4
    PIN_MEM = True
    SEED = 89
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # Preprocess dataset
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])

    # Load the datasets needed.
    # Shape: 3 x 256 x 256 image
    trainset = SegDataset(img_dir, mask_dir, transform=preprocess)
    train_loader = DataLoader(trainset, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEM, shuffle=True)

    valset = SegDataset(val_imgs_dir, val_masks_dir, transform=preprocess)
    val_loader = DataLoader(valset, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEM)

    # Load Model
    model = SegNetV2()
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Train the model
    # train(model, device, train_loader, val_loader, loss_fn, optimizer, EPOCHS,True)

    img, mask = next(iter(val_loader))
    model.load()
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        model.to(device)

        out = model(img).argmax(1)

        out = out.detach().cpu()
        print(out.size())
        print(mask.size())
        imshow(img.cpu(),out)
        # imshow(img,out)

