import torch 
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, optimizer, data_loader,loss_fn,device, epochs,load_model=False):
    # Allows for gradient scaling
    scaler = torch.cuda.amp.GradScaler()
    best_score = np.inf
    # Set model to device
    model = model.to(device)
    # Load model if neededj
    if load_model:
        print("Model Loaded.")
        model.load()

    # Begin training
    print("Starting...")
    for epoch in range(epochs):
        total_loss = 0
        for i , data in enumerate(tqdm(data_loader)):
            # get the image and mask and send to device
            image, mask = data
            image = image.to(device)
            mask = mask.to(device)

            # Zero Grad
            for p in model.parameters():
                p.grad = None

            # Forward Pass
            with torch.cuda.amp.autocast():
                output = model(image)
                loss = loss_fn(output, mask)
            
            # Backwards Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # compute losses for iteration 
            total_loss += loss.item()
    
    # If the loss is the lowest then save the model. 
        if total_loss < best_score:
            print("Saving...")
            best_score = total_loss
            model.save()

        print(f"EPOCH {epoch}: TOTAL LOSS: {total_loss:.5f}")

    print("Finished.")


def test_model(model, data_loader,loss_fn, device):
    scaler = torch.cuda.amp.GradScaler()
    model = model.to(device)
    model.load()

    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            image, mask = data 
            image = image.to(device)
            mask = mask.to(device)

            with torch.cuda.amp.autocast():
                pred = model(image)
                loss = loss_fn(pred, mask)

            total_loss += loss.item()
            break
    print(f"Total loss: {total_loss:.5f}")




def display(image, mask):
    image = image.cpu()
    mask = mask.detach().cpu()

    image = image / 2 + 0.5
    mask = mask / 2  + 0.5
    plt.imshow(image.permute(1,2,0), cmap='jet')
    plt.imshow(mask.permute(1,2,0), cmap='hot', alpha=0.9)
    plt.show()


