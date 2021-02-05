import os
import torch
import torch.nn as nn
from torchvision import models

class CarSegNet(nn.Module):
    def __init__(self, chkpt="models"):
        super(CarSegNet, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet101(False, num_classes=2)
        self.file = os.path.join(chkpt, "deeplab_weights_cars")


    def forward(self,x):
        return self.model(x)['out']

    def save(self):
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))