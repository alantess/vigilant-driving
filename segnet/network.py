import torch
import os
import torch.nn as nn
import torchvision


class SegNet(nn.Module):
    def __init__(self, chkpt_dir="models"):
        super(SegNet,self).__init__()
        self.file = os.path.join(chkpt_dir, "deeplab_weights_driving")
        self.base_model = torchvision.models.segmentation.deeplabv3_resnet101(False, num_classes=3)

    def forward(self,x):
        return self.base_model(x)['out']

    def save(self):
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))
    

