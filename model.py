import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np
from data import ChallengeDataset
import pandas as pd

class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResBlock, self).__init__()
        padding = int(np.ceil(0.5 * (kernel_size - 1)))

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

        if stride == 2:
            # Make channel and spatial dimension of input match the output
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels))
        else:
            # No adjustment required
            self.skip = nn.Identity()
    
    def forward(self, x):
        return self.block(x) + self.skip(x)



class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        # 3, 4, 6, 3
        self.res_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1),
            # ResBlock(64, 64, stride=1),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1),
            # ResBlock(128, 128, stride=1),
            # ResBlock(128, 128, stride=1),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
            # ResBlock(256, 256, stride=1),
            # ResBlock(256, 256, stride=1),
            # ResBlock(256, 256, stride=1),
            # ResBlock(256, 256, stride=1),
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
            # ResBlock(512, 512, stride=1),
            nn.AvgPool2d(kernel_size=(10, 10)),
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

        # self.apply(weights_init)
                

    def forward(self, x):
        return self.res_net(x)


    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            t.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            t.nn.init.constant_(m.bias.data, 0)