import torch
from torch import nn

import config as c
import numpy as np


def evaluate_model(model, data_loader):
    mses = np.zeros(shape=len(data_loader))
    mse = nn.MSELoss().to(c.device)

    with torch.no_grad():
        for i, (X, y, meta) in enumerate(data_loader):
            output = model(X)
            mses[i] = mse(y, output)
    return np.mean(mses)