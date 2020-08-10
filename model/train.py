import os
from pickle import load

import torch
from torch import nn

import numpy as np
import config as c
from dataset.crop_data_loader import create_data_loaders
from model.evaluation import evaluate_model
from model.simple_conv_net import SimpleCnn
from utils.logger import Logger


def train(experiment_name):
    exp_path = c.FOLDER_PATH_EXPERIMENTS + os.sep + experiment_name
    c.create_folder(exp_path)
    logger = Logger('EXP-' + experiment_name)
    # data
    device = c.device
    train_loader, test_loader, validation_loader = create_data_loaders()

    # hyper params
    learning_rate = 1e-3
    weight_decay = 1e-5
    max_epochs = 2
    max_batches_per_epoch = len(train_loader)
    validation_set_per_batch = 250
    log_per_batch = 10

    model = SimpleCnn().to(device)
    mse = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # stats
    current_batch = 0
    best_validation_loss = np.inf
    losses = list()
    validation_losses = list()

    for current_epoch in range(max_epochs):
        for X, y, meta in train_loader:
            outputs = model(X)
            optimizer.zero_grad()
            loss = mse(outputs, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            current_batch += 1
            if current_batch % log_per_batch == 0:
                logger.log(
                f'Epoch[{current_epoch + 1}/{max_epochs}] Batch [{current_batch}/{max_batches_per_epoch}], Loss: {loss.item() * c.MAX_PIXEL_VALUE}')

            if current_batch % validation_set_per_batch == 0:
                best_validation_loss = test_validation_set(best_validation_loss, exp_path, model, validation_loader, validation_losses, logger)
        current_batch = 0

    test_validation_set(best_validation_loss, exp_path, model, validation_loader, validation_losses, logger)

    best_model = torch.load(f'{exp_path}{os.sep}{c.BEST_MODEL}')
    validation_loss = evaluate_model(best_model, validation_loader).item()
    test_loss = evaluate_model(best_model, test_loader).item()

    logger.log(f'Best model validation loss: {validation_loss * c.MAX_PIXEL_VALUE}')
    logger.log(f'Best model test loss: {test_loss * c.MAX_PIXEL_VALUE}')

    with open(exp_path + os.sep + c.EXPERIMENT_RESULTS, 'w+') as fh:
        print(f'Best Model', file=fh)
        print(f'Test Loss: {test_loss}', file=fh)
        print(f'Validation Loss: {validation_loss}', file=fh)
        print(f'Training', file=fh)
        print(f"Losses: {losses}", file=fh)
        print(f"Validation Losses: {validation_losses}", file=fh)

    print('Finished Training')


def test_validation_set(best_validation_loss, exp_path, model, validation_loader, validation_losses, logger):
    current_validation_loss = evaluate_model(model, validation_loader)
    current_validation_loss = current_validation_loss.item()
    logger.log(f'validation loss: {current_validation_loss * c.MAX_PIXEL_VALUE}')
    validation_losses.append(current_validation_loss)
    if current_validation_loss <= best_validation_loss:
        best_validation_loss = current_validation_loss
        torch.save(model, f'{exp_path}{os.sep}{c.BEST_MODEL}')
    return best_validation_loss


