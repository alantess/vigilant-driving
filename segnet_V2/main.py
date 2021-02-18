import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from dataset import SegDataset
from network import SegNetV2
import cv2


def train(model, device, train_loader, val_loader, loss_fn, optimizer, epochs, load_model=False):
    """
    :param model: Model for predictions
    :param device: GPU or CPU
    :param train_loader: Training set
    :param val_loader: Validation Set
    :param loss_fn: loss function
    :param optimizer: Optimizer
    :param epochs: amount of epochs
    :param load_model: Loads a pretrained model
    :return: None
    """
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

        print(f'Training Loss {total_loss:.3f} \t Validation Loss {val_loss:.3f}')


def visualize(directory, model, transform, device):
    """
    Takes video and turn them into images.
    Saves the images into a folder.
    :param directory:Where the images saved
    :param model:Model for prediction
    :param transform:Convert images to tensor
    :param device: GPU or CPU
    :return: None
    """
    model.load()
    model.to(device)
    model.eval()
    with torch.no_grad():
        video = cv2.VideoCapture(directory)
        success, image = video.read()
        count = 0
        while success:
            img = transform(image)
            img = img.to(device, dtype=torch.float32).unsqueeze(0)
            pred = model(img).argmax(1)
            imshow(img, pred, count)
            success, image = video.read()
            count += 1



def imshow(img, mask, count=0):
    """
    Saves image to a folder
    :param img: Image
    :param mask: Mask
    :param count: Frame count
    :return:
    """
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    resize = transforms.Resize((360,640))
    save_dir = "D:\\VIDEOS\\segnet_V2\\img" + str(count) + ".png"
    img, mask = img.cpu(), mask.detach().cpu()
    img, mask = resize(img), resize(mask)
    img = invTrans(img)
    plt.axis('off')
    plt.imshow(img[0].permute(1, 2, 0), cmap='gray')
    plt.imshow(mask[0], cmap='coolwarm', alpha=0.4)
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)
    # plt.show()


if __name__ == '__main__':
    # Directories
    img_dir = "D:\\dataset\\comma10k\\imgs"
    mask_dir = "D:\\dataset\\comma10k\\masks"
    val_imgs_dir = "D:\\dataset\comma10k\\val_imgs"
    val_masks_dir = "D:\\dataset\\comma10k\\val_masks"
    video_directory = "D:\\VIDEOS\\default_driving.mp4"

    # Needed for training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 2
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

    # Video Prepreocess
    video_preprocess =transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
    train(model, device, train_loader, val_loader, loss_fn, optimizer, EPOCHS)

    # Visualize
    visualize(video_directory, model, video_preprocess, device)



