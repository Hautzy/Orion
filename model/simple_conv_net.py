import torch
from torch import nn


class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()
        ms = 16
        self.t1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=ms, kernel_size=(4 , 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t2 = nn.Sequential(
            nn.Conv2d(in_channels=ms, out_channels=ms*4, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(ms*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t3 = nn.Sequential(
            nn.Conv2d(in_channels=ms*4, out_channels=1, kernel_size=(5, 5)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        return x  # output of discriminator