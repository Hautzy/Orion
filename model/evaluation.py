import os
import torch
from torch import nn

import config as c
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.pixel_normalizer import get_latest_total_mean


def evaluate_model(model, data_loader):
    mses = np.zeros(shape=len(data_loader))
    mse = nn.MSELoss().to(c.device)
    total_pixel_mean = get_latest_total_mean()

    with torch.no_grad():
        for i, (X, y, metas) in enumerate(data_loader):
            X = X.to(c.device)
            output = model(X, metas)
            # testing with real pixel values
            # y += total_pixel_mean
            y *= c.MAX_PIXEL_VALUE

            # output += total_pixel_mean
            cpu_output = output.cpu().detach()
            cpu_output *= c.MAX_PIXEL_VALUE

            mses[i] = mse(y, cpu_output)
            del X, output
    return np.mean(mses)