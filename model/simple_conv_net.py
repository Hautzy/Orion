import torch
from torch import nn


class SimpleCnn(nn.Module):
    def __init__(self, n_hidden_layers=3):
        super(SimpleCnn, self).__init__()

        cnn = []
        n_in_channels = 2
        n_out_channels = 32
        kernel_size = 7

        # building up kind of hidden layers thingi here
        for i in range(n_hidden_layers):
            conv = nn.Conv2d(in_channels=n_in_channels, out_channels=n_out_channels, kernel_size=kernel_size,
                             padding=int(kernel_size / 2))
            batch_norm = nn.BatchNorm2d(n_out_channels)
            leaky_relu = nn.LeakyReLU(0.2, inplace=True)

            cnn.append(conv)
            cnn.append(batch_norm)
            cnn.append(leaky_relu)
            n_in_channels = n_out_channels
        self.hidden_layers = nn.Sequential(*cnn)

        n_out_channels = int(n_in_channels / 2)
        self.layer_01 = nn.Sequential(
            nn.Conv2d(in_channels=n_in_channels, out_channels=n_out_channels, kernel_size=4, stride=2),
            nn.BatchNorm2d(n_out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        n_in_channels = n_out_channels

        n_out_channels = int(n_in_channels / 2)
        self.layer_02 = nn.Sequential(
            nn.Conv2d(in_channels=n_in_channels, out_channels=n_out_channels, kernel_size=5, stride=2),
            nn.BatchNorm2d(n_out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        n_in_channels = n_out_channels

        n_out_channels = 1
        self.layer_03 = nn.Sequential(
            nn.Conv2d(in_channels=n_in_channels, out_channels=n_out_channels, kernel_size=3),
            nn.BatchNorm2d(n_out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.layer_01(x)
        x = self.layer_02(x)
        x = self.layer_03(x)
        return x  # output of discriminator
