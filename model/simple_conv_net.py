import torch
import config as c
from torch import nn


class SimpleCnn(nn.Module):
    def __init__(self, n_hidden_layers=1):
        super(SimpleCnn, self).__init__()

        cnn = []
        n_in_channels = 2
        n_out_channels = 8
        kernel_size = 7

        # building up kind of hidden layers thingi here
        for i in range(n_hidden_layers):
            n_out_channels *= 2
            conv = nn.Conv2d(in_channels=n_in_channels, out_channels=n_out_channels, kernel_size=kernel_size, bias=True,
                             padding=int(kernel_size / 2))
            relu = nn.ReLU()

            cnn.append(conv)
            cnn.append(relu)
            n_in_channels = n_out_channels

        self.hidden_layers = nn.Sequential(*cnn)
        self.last = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                    kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2))


    def forward(self, x, metas):
        x = self.hidden_layers(x)
        x = self.last(x)

        cropped_output = torch.zeros(size=(len(metas), 1, c.MAX_CROP_SIZE, c.MAX_CROP_SIZE)).to(c.device)
        for i, meta in enumerate(metas):
            (st_x, st_y), (en_x, en_y) = meta.get_coordinates()
            cropped_output[i, 0, :meta.height, :meta.width] = x[i, 0, st_y:en_y, st_x:en_x]

        return cropped_output
