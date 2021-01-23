import torch as T
import cv2
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from gtrxl_torch.gtrxl_torch import GTrXL
from sklearn.preprocessing import MinMaxScaler
from torch import optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models


class CNNGRU(nn.Module):
    def __init__(self, lr=0.00065):
        super(CNNGRU, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.gru = nn.GRU(128, 256, 4)
        self.out = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)
        out, h0 = self.gru(x)
        out = self.out(out).mean(1)
        return out


class CNNTransformer(nn.Module):
    def __init__(self, lr=0.00045):
        super(CNNTransformer, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(1000, 256)
        self.tranformer = GTrXL(256, 4, 3)
        self.fc2 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(512, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(self.fc1(x))
        x = self.tranformer(x)
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.out(x).mean(0)
        return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_labels(dir):
    df = pd.read_csv(dir)
    scaler = MinMaxScaler()
    scaler.fit(df.values)
    data = scaler.transform(df.values)
    return scaler , data


def vid_to_tensor(dir, transform):
    image_tensor = T.zeros((20400, 3, 256, 256))
    video = cv2.VideoCapture(dir)
    success, image = video.read()
    count = 0
    while success:
        image_tensor[count] = transform(image)
        success, image = video.read()
        count += 1

    return image_tensor

def train(train_dir,labels_dir,  transform,criterion, batch, EPOCHS =1,gru = True):
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    best_loss = np.inf
    if gru:
        model = CNNGRU().to(device)
        save_path = "models/gru.pt"
    else:
        model = CNNTransformer().to(device)
        save_path = "models/transformer.pt"

    print("Model Parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    scaler, labels = get_labels(labels_dir)
    labels = T.tensor(labels, dtype=T.float, device=device)
    
    print("starting...")
    for epoch in range(EPOCHS):
        total_loss , running_loss = 0.0, 0.0
        image_tensor = T.zeros((batch, 3,256,256), device=device)
        video = cv2.VideoCapture(train_dir)
        success, image = video.read()
        index,i,count  = 0,0,0
        counter = 0
        while success: 
            image_tensor[count] = transform(image) 
            count += 1

            if count % batch == 0: 
                i += 1
                index = i * batch
                # Set labels
                y = labels[index-batch:index] 
                
                # # zeros out gradients
                for p in model.parameters():
                    p.grad = None
                #
                # Make a predictions and get the loss
                pred = model(image_tensor)
                loss = criterion(pred,y)
                loss.backward()
                # Optimizer step
                model.optimizer.step()

                running_loss += loss.item()
                total_loss += running_loss
                if i % 255 == 254: 
                    running_loss /= 255
                    print(f"[{epoch}/{i+1}] \tLoss {running_loss:.3f} ")

                image_tensor = T.zeros((batch, 3,256,256), device=device)
                count = 0

            success, image = video.read()


        print("Total loss: ", total_loss )

        if total_loss < best_loss:
            print("Saving...")
            best_loss = total_loss

    # imshow(torchvision.utils.make_grid(image_tensor.cpu()))



if __name__ == '__main__':
    train_video = "/mnt/d/pytorch/speedchallenge/data/train.mp4"
    train_labels = "/mnt/d/pytorch/speedchallenge/data/train.txt"

    SEED = 98
    T.manual_seed(SEED)
    np.random.seed(SEED)
    T.backends.cudnn.benchmark = True
    best_loss = np.inf
    BATCH_SIZE = 16
    EPOCHS = 200
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")

    criterion = nn.MSELoss()

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ])

    

    train(train_video,train_labels,transform, criterion, 20)





