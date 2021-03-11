import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models


class VideoResNet(nn.Module):
    def __init__(self, timesteps, lr=1e-5, scaler=None, chkpt_dir="models"):
        super(VideoResNet, self).__init__()
        self.base_model = models.video.r3d_18(pretrained=True)
        self.gru = nn.GRU(400, 256, 3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, timesteps)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.file = os.path.join(chkpt_dir, 'video_resnet_weights')
        self.timesteps = timesteps
        self.scaler = scaler

    def forward(self, x):
        x = self.base_model(x).unsqueeze(1)
        x, h0 = self.gru(x)
        x = x.view(-1)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.out(x)
        if self.scaler:
            x = x.reshape(self.timesteps, 1)
            x = self.scaler.inverse_transform(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))
