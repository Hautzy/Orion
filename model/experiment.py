import json
import os
from pickle import load, dump

import numpy as np
import torch
from torch import nn

import config as c
from dataset.crop_data_loader import create_data_loaders
from preprocessing.crop_meta import CropMeta
from preprocessing.sample import Sample
from utils.logger import Logger
import matplotlib.pyplot as plt

LOG_PER_BATCHES = 25
TEST_VALIDATION_SET_PER_BATCHES = 500
REPORT = 'report.json'
BEST_MODEL = 'best_model.pk'
PREDICTIONS = 'predictions.pk'


class Stats:
    def __init__(self):
        self.best_validation_loss = np.inf
        self.train_losses = list()
        self.validation_losses = list()
        self.best_test_loss = 0
        self.submission_test_loss = 0


class Experiment:

    def __init__(self, name):
        self.name = name
        self.path = f'{c.FOLDER_PATH_MAIN_EXPERIMENTS}{os.sep}{name}'
        self.create_experiment_folder()
        self.logger = Logger(f'EXPERIMENT-{name}')
        self.stats = Stats()

    def create_experiment_folder(self):
        c.create_folder(self.path)

    def train_model(self, model, params, max_epochs=1):
        train_loader, test_loader, validation_loader = create_data_loaders()

        # hyper params
        learning_rate = params['lr']
        weight_decay = params['wd']

        max_batches_per_epoch = len(train_loader)
        model = model.to(c.device)
        mse = nn.MSELoss().to(c.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        current_batch = 0
        for current_epoch in range(max_epochs):
            for X, y, crop_metas, _ in train_loader:
                X = X.to(c.device)
                y = y.to(c.device)
                outputs = model(X)
                optimizer.zero_grad()
                # so there are models which return 21x21 and those who return the full image
                loss = self.calculate_loss(outputs, X, y, crop_metas, mse)
                loss.backward()
                optimizer.step()

                self.stats.train_losses.append(loss.item())
                current_batch += 1
                if current_batch % LOG_PER_BATCHES == 0:
                    self.logger.log(f'Epoch[{current_epoch + 1}/{max_epochs}] '
                                    f'Batch [{current_batch}/{max_batches_per_epoch}], '
                                    f'Loss: {loss.item() * 1000}')
                if current_batch % TEST_VALIDATION_SET_PER_BATCHES == 0:
                    self.test_validation_set(model, validation_loader)
                del X, y, outputs
            current_batch = 0

        self.test_validation_set(model, validation_loader)

        del model, mse
        self.print_stats(test_loader)
        self.logger.log('Finished Training')

    def calculate_loss(self, outputs, X, y, crop_metas, mse):
        if outputs.shape[3] == c.MAX_CROP_SIZE:
            cropped_output = torch.zeros(size=(len(crop_metas), 1, c.MAX_CROP_SIZE, c.MAX_CROP_SIZE)).to(c.device)
            for i, meta in enumerate(crop_metas):
                (st_x, st_y), (en_x, en_y) = meta.get_coordinates()
                cropped_output[i, 0, :meta.height, :meta.width] = outputs[i, 0, st_y:en_y, st_x:en_x]
            del cropped_output
            return mse(outputs, y)
        else:
            images = torch.zeros(size=(len(crop_metas), 1, c.MAX_SAMPLE_HEIGHT, c.MAX_SAMPLE_WIDTH))
            for i, crop_meta in enumerate(crop_metas):
                (st_x, st_y), (en_x, en_y) = crop_meta.get_coordinates()
                images[i, 0, :, :] = X[i, 0, :, :].clone().detach()
                images[i, 0, st_y:en_y, st_x:en_x] = y[i, 0, :crop_meta.height, :crop_meta.width]
            images = images.to(c.device)
            loss = mse(images, outputs)
            del images
            return loss


    def print_stats(self, test_loader):
        best_model = torch.load(f'{self.path}{os.sep}{BEST_MODEL}').to(c.device)
        test_loss = self.evaluate_model(best_model, test_loader)

        self.stats.best_test_loss = test_loss

        submission_outputs = self.create_submission_predictions(best_model)
        self.submission_scoring(submission_outputs)

        self.logger.log(f'Best model validation loss: {self.stats.best_validation_loss}')
        self.logger.log(f'Best model test loss: {test_loss}')

        self.plot_loss(self.stats.train_losses, 'train_losses')
        self.plot_loss(self.stats.validation_losses, 'validation_losses')

        with open(f'{self.path}{os.sep}{REPORT}', 'w+') as f:
            json_stats = json.dumps(self.stats.__dict__)
            f.writelines(json_stats)
        del best_model

    def plot_loss(self, loss, name):
        loss = np.array(loss)
        length = loss.shape[0]
        x = np.linspace(0, length, length)
        plt.figure()
        plt.plot(x, loss)
        plt.savefig(f'{self.path}{os.sep}{name}.svg')

    def test_validation_set(self, model, validation_loader):
        current_validation_loss = self.evaluate_model(model, validation_loader)
        current_validation_loss = current_validation_loss.item()
        self.logger.log(f'Validation Loss: {current_validation_loss}')
        self.stats.validation_losses.append(current_validation_loss)
        if current_validation_loss <= self.stats.best_validation_loss:
            self.stats.best_validation_loss = current_validation_loss
            torch.save(model, f'{self.path}{os.sep}{BEST_MODEL}')


    def evaluate_model(self, model, data_loader):
        mses = np.zeros(shape=len(data_loader))
        mse = nn.MSELoss().to(c.device)
        # total_pixel_mean = get_latest_total_mean()

        with torch.no_grad():
            for i, (X, y, metas, _) in enumerate(data_loader):
                X = X.to(c.device)
                outputs = model(X)
                # cpu_output = output.cpu().detach()
                # y *= c.MAX_PIXEL_VALUE
                # cpu_output *= c.MAX_PIXEL_VALUE
                # TODO: reintroduce normalizing pixel values
                # y += total_pixel_mean
                # output += total_pixel_mean
                loss = self.calculate_loss(outputs, X, y, metas, mse)
                mses[i] = loss.item()
                del X
        return np.mean(mses)

    def submission_scoring(self, predictions=None):
        if predictions is None:
            with open(f'{self.path}{os.sep}{PREDICTIONS}', 'rb') as f:
                predictions = load(f)
        with open(c.FILE_SUBMISSION_TESTING_TARGETS, 'rb') as f:
            targets = load(f)

        mses = np.zeros(shape=len(predictions))
        ind = 0
        for target, prediction in zip(targets, predictions):
            mses[ind] = Experiment.submission_mse(target, prediction)
            ind += 1
        mean_loss = np.mean(mses)
        self.stats.submission_test_loss = mean_loss
        self.logger.log(f'Mean submission loss: {mean_loss}')

    @staticmethod
    def submission_mse(target_array, prediction_array):
        prediction_array, target_array = np.asarray(prediction_array, np.float64), np.asarray(target_array, np.float64)
        other_test = np.mean((prediction_array - target_array) ** 2)
        return other_test

    def create_submission_predictions(self, model):
        with open(c.FILE_SUBMISSION_TESTING_RAW_DATA, 'rb') as f:
            test_set = load(f)

        crop_sizes = test_set['crop_sizes']
        crop_centers = test_set['crop_centers']
        images = test_set['images']
        model = model.to(c.device)
        #total_pixel_mean = get_latest_total_mean()
        outputs = list()

        for ind in range(len(images)):
            crop_size = crop_sizes[ind]
            crop_center = crop_centers[ind]
            image = images[ind]
            image = np.array(image, dtype='float64')
            image /= c.MAX_PIXEL_VALUE
            #TODO: image -= total_pixel_mean

            crop_meta = CropMeta(crop_size, crop_center)

            t_X = torch.from_numpy(image)
            (st_x, st_y), (en_x, en_y) = crop_meta.get_coordinates()
            t_y = t_X[st_y:en_y, st_x:en_x].clone().detach()
            t_X[st_y:en_y, st_x:en_x] = c.PADDING_VALUE
            t_map = torch.zeros(t_X.shape)
            t_map[st_y:en_y, st_x:en_x] = c.MAP_POS

            sample = Sample(t_X, t_y, t_map, crop_meta)

            X = torch.zeros(size=(1, 2, c.MAX_SAMPLE_WIDTH, c.MAX_SAMPLE_HEIGHT)).to(c.device)
            y = torch.zeros(size=(1, 1, c.MAX_CROP_SIZE, c.MAX_CROP_SIZE)).to(c.device)
            X_shape = sample.X.shape
            X[0, 0, :X_shape[0], :X_shape[1]] = sample.X
            X[0, 1, :X_shape[0], :X_shape[1]] = sample.map
            y_shape = sample.y.shape
            y[0, 0, :y_shape[0], :y_shape[1]] = sample.y

            output = model(X)
            cpu_output = output.cpu().detach().numpy()
            del output
            #cpu_output += total_pixel_mean
            cpu_output *= c.MAX_PIXEL_VALUE

            if cpu_output.shape[2] == c.MAX_CROP_SIZE:
                outputs.append(cpu_output[0, 0, :crop_meta.height, :crop_meta.width])
            else:
                outputs.append(cpu_output[0, 0, st_y:en_y, st_x:en_x])
        self.logger.log(f'Tested {ind+1} submission samples')
        del model
        with open(f'{self.path}{os.sep}{PREDICTIONS}', 'wb') as f:
            dump(outputs, f)
        return outputs
