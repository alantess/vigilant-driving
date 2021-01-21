from gtrxl_torch.gtrxl_torch import GTrXL
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules import transformer
import torchvision.transforms as transforms

"""
1 Image is the source and the other is the target
the source can be represented as as 32 x 32 image
while the target can be a 48 x 48 image

DIMS = S  x N  x E
Source Dims = 4 x 32 x 768


Src goes through GTRxL and MLP HEAD
We get a classification

# Test this on the markov model dataset I have
"""


def make_patches(img: Tensor, patch_size: int) -> Tensor:
    """
    HW = Resolution of the image
    P = Resolution of the patch
    Calculate patch (S) = HW / P^2

    :param img: Batch x Channels x Height x Width 
    :return:Tensor: S x N x E
    """
    n, c = img.size(0), img.size(1)
    height, width = img.size(2), img.size(3)
    assert (height == width), "Height and width must be the same size"

    n_patches = (height * width) // pow(patch_size, 2)
    flatten_dim = c * pow(patch_size, 2)

    flatten_img = T.zeros(size=(n_patches, n, flatten_dim), dtype=T.float)
    n_row = math.sqrt(n_patches)

    for i in range(n):
        row, col = 0, 0
        for j in range(n_patches):

            if j % n_row == 0:
                row += patch_size
                col = patch_size
            else:
                col += patch_size

            image_patch = img[i, :, row - patch_size:row, col - patch_size: col].reshape(-1)
            flatten_img[j][i] = image_patch

    return flatten_img


def load_dataset(train_path, test_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader


class MarkovNet(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, n_classes, h_dims, dropout=0.5, lr=0.00025):
        super(MarkovNet, self).__init__()
        self.transformer = GTrXL(d_model, n_heads, n_layers, activation="gelu")
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, h_dims)
        self.out = nn.Linear(h_dims, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.transformer(x)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.out(x)
        return x


def calculate_d_model(trainloader, patches):
    img, _ = next(iter(trainloader))
    img = make_patches(img, patches)
    return img.size(2)


def train(train_dir, test_dir, epochs, device, nheads=4, t_layers=3, h_dims=256,patch_dims=16, train=True):
    trainloader, testloader = load_dataset(train_dir, test_dir, BATCH_SIZE)
    d_model = calculate_d_model(trainloader, 16)
    n_classes = 2
    best_score = -np.inf

    criterion = nn.CrossEntropyLoss()
    model = MarkovNet(d_model,nheads, t_layers, n_classes, h_dims).to(device)
    print("Train loader size: ", len(trainloader))
    print("Model Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if train:
        print("Train mode initiated.")
        for epoch in range(epochs):
            running_loss = 0.0
            for i , data in enumerate(trainloader,0):
                inputs, labels = data
                img = make_patches(inputs,patch_dims)
                img, labels = img.to(device), labels.to(device)

                for p in model.parameters():
                    p.grad = None

                output = model(img).mean(0)

                loss = criterion(output, labels)
                loss.backward()
                model.optimizer.step()

                running_loss += loss.item()

                if i % 69 == 68:
                    print(f"|{epoch}|{i}|\tloss: {running_loss/69:.3f}")
                    running_loss = 0.0
        print("FINISHED.")
    else:
        print("Test mode initiated.")






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    EPOCHS = 100
    BATCH_SIZE = 32
    SEED = 75
    T.backends.cudnn.benchmark = True
    T.manual_seed(SEED)
    np.random.seed(SEED)

    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    train_dir = '../../dataset/ltc_classifer'
    test_dir = '../../dataset/ltc_test_classifier'

    train(train_dir,test_dir,EPOCHS,device)


