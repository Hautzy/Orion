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


def plot_experiment(exp_name):
    exp_path = '../' + c.FOLDER_PATH_MAIN_EXPERIMENTS + os.sep + exp_name

    with open(exp_path + os.sep + c.EXPERIMENT_RESULTS, 'r') as f:
        lines = f.readlines()
        losses_line = lines[4]
        validation_line = lines[5]

        losses_line = losses_line.replace(']', '').replace('Losses: [', '')
        losses = line_of_losses_to_arr(losses_line)
        x_length = losses.shape[0]
        x_losses = np.linspace(0, x_length, x_length)
        plt.figure()
        plt.plot(x_losses, losses)
        plt.savefig(exp_path + os.sep + 'losses.svg')

        plt.figure()
        validation_line = validation_line.replace(']', '').replace('Validation Losses: [', '')
        validation = line_of_losses_to_arr(validation_line)
        x_length = validation.shape[0]
        x_losses = np.linspace(0, x_length, x_length)
        plt.subplot
        plt.plot(x_losses, validation)
        plt.savefig(exp_path + os.sep + 'validation.svg')


def line_of_losses_to_arr(line):
    parts = line.split(',')
    nums = [float(part) for part in parts]
    return np.array(nums)


plot_experiment('exp_01')
plot_experiment('exp_02')
plot_experiment('exp_03')
plot_experiment('exp_04')
plot_experiment('exp_05')
