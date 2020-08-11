import torch
from torch import nn

import config as c
import numpy as np


def evaluate_model(model, data_loader):
    mses = np.zeros(shape=len(data_loader))
    mse = nn.MSELoss().to(c.device)

    with torch.no_grad():
        for i, (X, y, metas) in enumerate(data_loader):
            X = X.to(c.device)
            y = y.to(c.device)
            output = model(X, metas)
            mses[i] = mse(y, output)
            del X, y
    return np.mean(mses)