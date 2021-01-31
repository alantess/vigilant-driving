import torch
import os
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, chkpt_dir="models"):
        super(UNet,self).__init__()
        self.file = os.path.join(chkpt_dir, "u_net_weights_driving")
        self.u_net = model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                    in_channels=3, out_channels=1, init_features=32, pretrained=False) 


    def forward(self,x):
        return self.u_net(x)

    def save():
        torch.save(self.state_dict(), self.file)

    def load():
        self.load_state_dict(torch.load(self.file))
    

