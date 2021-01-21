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

def make_patches(img:Tensor, patch_size:int) -> Tensor:
    """
    HW = Resolution of the image
    P = Resolution of the patch
    Calculate patch (S) = HW / P^2

    :param img: Batch x Channels x Height x Width 
    :return:Tensor: S x N x E
    """
    n,c = img.size(0), img.size(1)
    height, width = img.size(2), img.size(3)
    assert (height == width), "Height and width must be the same size"

    n_patches = (height * width) // pow(patch_size,2)
    flatten_dim = c * pow(patch_size,2)


    flatten_img = T.zeros(size = (n_patches, n, flatten_dim), dtype=T.float)
    n_row = math.sqrt(n_patches)

    for i in range(n):
        row, col = 0, 0
        for j in range(n_patches):
            if j % n_row == 0:
                row += patch_size
                col = patch_size
            else:
                col += patch_size

            image_patch = img[i,:, row - patch_size:row, col - patch_size: col].reshape(-1)
            flatten_img[j][i] = image_patch

    return flatten_img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = T.randn((2,3,32,32))
    dim = make_patches(image,16)
    print(dim.size())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
