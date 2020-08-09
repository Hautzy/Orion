import torch
from torch import nn

import numpy as np
import config as c
from dataset.crop_data_loader import create_data_loaders
from model.evaluation import evaluate_model
from model.simple_conv_net import SimpleCnn


def train():
    # data
    device = c.device
    train_loader, test_loader, validation_loader = create_data_loaders()

    # hyper params
    learning_rate = 1e-3
    weight_decay = 1e-5
    max_epochs = 2
    max_batches_per_epoch = len(train_loader)
    validation_set_per_batch = 1

    model = SimpleCnn().to(device)
    mse = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # stats
    current_batch = 0
    losses = list()

    for current_epoch in range(max_epochs):
        for X, y, meta in train_loader:
            outputs = model(X)
            optimizer.zero_grad()
            loss = mse(outputs, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            current_batch += 1
            print(
                f'Epoch[{current_epoch + 1}/{max_epochs}] Batch [{current_batch}/{max_batches_per_epoch}], Loss: {loss.item() * c.MAX_PIXEL_VALUE}')

            if current_batch % validation_set_per_batch == 0:
                current_validation_loss = evaluate_model(model, validation_loader)
                print(f'validation loss: {current_validation_loss * c.MAX_PIXEL_VALUE}')
        current_batch = 0
